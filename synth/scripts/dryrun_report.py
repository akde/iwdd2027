"""Tables for an integration dry run of the generator (synth/reports/dryrun_v3.md); read-only on every batch dir.
python synth/scripts/dryrun_report.py dry DIR [DIR ...] [--legs] [--workers 16] > TABLES.md
python synth/scripts/dryrun_report.py retro OUT.csv [--legs] [--workers 16] > TABLES.md
dry: DIR = a `run_batch.py --plan-only` batch dir (scenarios/, legcheck.jsonl) with its log at DIR.log. Planning and
  rejections (log; gate = leg_overlap and apart estimate, screen = postqc.check strict; the crouch rule of D41 by mode,
  gender and object), scenes / plates vs a scene-uniform target, families, gender x object, night, distractors (per
  label), light per source clip against synth/data/plate_light.csv (D40), timing laws and actor size vs the
  accepted pairs of s15-s17, release v2 and the real contest law, then per planned member: the strict screen re-run,
  the structure check, a bar check of the projected walkers and object, and the leg gate (gate records, or with
  --legs a fresh leg_overlap.py run in Blender without rendering, compared with them); a GPU time estimate.
retro: the 439 accepted pairs of release v1.2 + s15-s17 through every strict rule of a new batch (postqc.check
  strict with the vehicle masks, and with --legs the leg gate qc.leg_verdict and the apart estimate as
  run_batch.screen_gate) -> OUT.csv, one row per pair; release v2 is not changed."""
import argparse, ast, collections, json, pathlib, re, sys, tempfile
import concurrent.futures as cf
import numpy as np, pandas as pd
from scipy.stats import ks_2samp

SYN = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SYN / "gen"), str(SYN / "scripts")]
import scenario as S, qc, degrade, postqc, run_batch
from camera import Camera, group_calib
from tracks import track, walker_keys

D = SYN / "data"
GATE = {"walk-in through object", "standing in object", "walk-out through object", "distractor through object",
        "leg screen failed", "object never apart (estimate)", "apart screen failed", qc.CROUCH_WHY}
LARGE = {"compost_bag_02", "cement_bag", "industrial_pastic_container"}
Q = [0.1, 0.25, 0.5, 0.75, 0.9]
_OBST = None


def md(df, nd=3):
    """DataFrame -> markdown table (index included)."""
    df = df.reset_index()
    f = lambda v: f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
                     + ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)]) + "\n"


def load(paths):
    """{group_id: (group, members positive first, group dict identical in every member file)} of scenario files."""
    out = collections.defaultdict(list)
    for p in paths:
        x = json.loads(p.read_text()); out[x["group"]["group_id"]].append(x)
    return {g: (xs[0]["group"], sorted((x["member"] for x in xs), key=lambda m: -m["y"]),
                all(x["group"] == xs[0]["group"] for x in xs)) for g, xs in out.items()}


def old_pairs(which):
    """Accepted pairs of s15-s17 ('s15-s17'), release v2 ('v2') or release v1.2 + s15-s17 ('all439')."""
    a = pd.read_csv(D / "synth_manifest.csv").assign(bd="data")
    b = pd.read_csv(D / "batch_s34/synth_manifest.csv").assign(bd="data/batch_s34")
    if which == "s15-s17":
        acc = a[a.qc]
    elif which == "v2":
        v2 = pd.read_csv(SYN / "release_v2_manifest.csv")
        acc = v2.assign(bd=v2.batch.map({"data": "data", "batch_s34": "data/batch_s34"}))
    else:
        acc = pd.concat([a[a.qc], b[b.qc & ~b.pair_id.isin(a.pair_id)]])
    acc = acc.drop_duplicates("pair_id")
    return {r.pair_id: r.bd for r in acc.itertuples()}


def old_groups(which):
    pairs = old_pairs(which)
    return load([p for pid, bd in pairs.items() for p in sorted((SYN / bd / "scenarios").glob(f"{pid}_*.json"))])


def rel_size(gr):
    cam = Camera(group_calib(gr), gr["render_size"])
    f, h, _ = cam.person_px(np.array([gr["path"]["X"]]), gr["actor"]["height_m"])
    return float((f[0, 1] - h[0, 1]) / cam.H)


def features(G, src):
    """One row per group and one per member."""
    g, m = [], []
    for gid, (gr, ms, same) in G.items():
        o = gr["object"]; pos = ms[0]
        g.append(dict(src=src, gid=gid, mode=S.mode_of(gr), scene=gr["scene_id"], plate=gr["plate_id"], fold=gr["fold"],
                      meva=gr["source"] != "contest", neg=gr["families"][1], gender=gr["actor"]["gender"],
                      obj="carrier_bag" if o.get("object_type") == "plastic_bag" else o["asset"], large=o["asset"] in LARGE,
                      night=gr["light"]["night"], sun=gr["light"]["sun"], rel=rel_size(gr), n_dis=len(gr["distractors"]),
                      onset=pos["t_onset_exact"], d=gr["duration"], t_event=gr["t_event"],
                      after=gr["duration"] - pos["t_onset_exact"], same=same,
                      frames=sum(int(np.floor(gr["duration"] * gr["render_fps"] + 1e-6)) for _ in ms)))
        for x in ms:
            m.append(dict(src=src, gid=gid, mode=S.mode_of(gr), outcome=x["outcome"], y=x["y"], n_dis=len(gr["distractors"]),
                          t_dec_null=x["t_dec"] is None, before_walkout=gr["duration"] < x["events"]["t_walk_out"]))
    return pd.DataFrame(g), pd.DataFrame(m)


def log_stats(d):
    """Per group id: mode, planned, draws, reason counter (one count per rejected draw and reason), from DIR.log."""
    rows = {}
    for line in pathlib.Path(str(d) + ".log").read_text().splitlines():
        m = re.match(r"(g\d+) planned (\S+) (\[.*?\]) draws=(\d+) ?(\{.*\})?", line)
        k = re.match(r"(g\d+) SKIPPED: \d+ draws (\[.*?\]|None) (car|on foot) failed the screen: (\{.*\})", line)
        if m:
            fam = ast.literal_eval(m.group(3))
            why = collections.Counter({r: v for key, v in ast.literal_eval(m.group(5) or "{}").items() for r in key.split("; ")})
            rows[m.group(1)] = dict(mode="car" if fam[0] == "unload_vehicle_leave" else "foot", planned=True,
                                    draws=int(m.group(4)), why=why)
        elif k:
            why = collections.Counter({r: v for key, v in ast.literal_eval(k.group(4)).items() for r in key.split("; ")})
            rows[k.group(1)] = dict(mode="car" if k.group(3) == "car" else "foot", planned=False, draws=run_batch.MAX_DRAWS, why=why)
    return rows


def gate_records(d):
    """legcheck.jsonl: per clip the list of records (one per gate round of its group; the last = the accepted draw)."""
    out = collections.defaultdict(list)
    p = d / "legcheck.jsonl"
    for line in (p.read_text().splitlines() if p.exists() else []):
        r = json.loads(line); out[r["clip"]].append(r)
    return out


def bar_px(cam, xy, h, width=0.7):
    """Render px of upright width x h m boxes at ground points xy inside the frame but outside the camera image."""
    foot, head, z = cam.person_px(xy, h)
    half = 0.5 * width * np.abs(foot[:, 1] - head[:, 1]) / h
    ov = lambda a0, a1, c0, c1: np.clip(np.minimum(a1, c1) - np.maximum(a0, c0), 0, None)
    x0, y0, x1, y1 = cam.box
    bx0, bx1, by0, by1 = foot[:, 0] - half, foot[:, 0] + half, head[:, 1], foot[:, 1]
    return np.where(z > 0.5, ov(bx0, bx1, 0, cam.W) * ov(by0, by1, 0, cam.H) - ov(bx0, bx1, x0, x1) * ov(by0, by1, y0, y1), 0.0)


def bars(gr, ms):
    """Bar check of the projected scene (F2, D31): px in black bars of the actor at the spot (t_arrive..t_walk_out),
    the object's projected rest box, the car-mode place S; distractor feet in a bar; walkers crossing a bar (walk-in /
    walk-out through a bar side is allowed: the compositor hides it as a real person leaves the image)."""
    cam = Camera(group_calib(gr), gr["render_size"])
    r = dict(bar_plate=bool(cam.bars.any()), spot=0.0, obj=0.0, S=0.0, dis_feet=0, cross=0.0)
    if not r["bar_plate"]:
        return r
    h, d = gr["actor"]["height_m"], gr["duration"]
    for m in ms:
        ev = m["events"]
        ts = np.arange(ev["t_arrive"], min(ev["t_walk_out"], d), 0.05)
        r["spot"] = max(r["spot"], float(bar_px(cam, track(m["root_keys"], ts), h).max(initial=0)))
        r["cross"] = max(r["cross"], float(bar_px(cam, track(m["root_keys"], np.arange(0, d, 0.05)), h).max()))
    (c, th, hh), dz = S.group_rest_box(gr), S.object_height(gr["object"]["asset"], gr["object"].get("scale", 1.0))
    P = c + np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) * hh @ S.rot(th).T
    uv, z = cam.project(np.r_[np.c_[P, np.zeros(4)], np.c_[P, np.full(4, dz)]])
    ov = lambda a0, a1, c0, c1: max(0.0, min(a1, c1) - max(a0, c0))
    (u0, v0), (u1, v1), (x0, y0, x1, y1) = uv.min(0), uv.max(0), cam.box
    r["obj"] = ov(u0, u1, 0, cam.W) * ov(v0, v1, 0, cam.H) - ov(u0, u1, x0, x1) * ov(v0, v1, y0, y1)
    if "vehicle" in gr:
        r["S"] = float(bar_px(cam, np.array([gr["path"]["E"]]), 1.7)[0])
    for Dd in gr["distractors"]:
        w = track(walker_keys(Dd["path"], Dd["t0"], Dd["speed"]), np.arange(0, d, 0.05))
        foot, _, z = cam.person_px(w, 1.7)
        fr = (z > 0.5) & (foot[:, 0] >= 0) & (foot[:, 0] < cam.W) & (foot[:, 1] >= 0) & (foot[:, 1] < cam.H)
        r["dis_feet"] += int((fr & ~cam.in_image(foot)).any())
        r["cross"] = max(r["cross"], float(bar_px(cam, w, S.structures.distractor_height(Dd)).max()))
    return r


def screen_one(item):
    """The strict screen (as run_batch) and the structure check of one group."""
    global _OBST
    if _OBST is None:
        _OBST = json.loads((D / "obstacles.json").read_text())
    gid, (gr, ms, _) = item
    cam = Camera(group_calib(gr), gr["render_size"])
    sh = postqc.structure_hit(gr, ms, cam)
    return gid, postqc.check(gr, ms, _OBST, vehicle_masks=True, strict=True), sh, bars(gr, ms)


def screen_all(G, workers):
    with cf.ProcessPoolExecutor(workers) as ex:
        return {gid: (w, sh, b) for gid, w, sh, b in ex.map(screen_one, G.items(), chunksize=4)}


def apart_s(gr, rec):
    """Seconds of rest frames with the released object seen apart (leg_overlap --apart record), as screen_gate."""
    W, H = gr["render_size"]
    sc = degrade.sample_params(np.random.default_rng(gr["degrade_seed"]), W, H, gr["render_fps"])["height"] / H
    return sum(px * sc * sc >= qc.MIN_OBJ_PX for _, px, _, _ in rec["apart"]) / gr["render_fps"]


def legs(files, workers):
    """Fresh leg_overlap.py records (Blender, no rendering; positives with --apart) -> {clip: record}."""
    with tempfile.TemporaryDirectory() as tmp:
        return run_batch.blender_screen([(f, f.stem.endswith("_leave")) for f in files], pathlib.Path(tmp), workers)


def leg_rows(G, rec):
    rows = []
    for gid, (gr, ms, _) in G.items():
        for i, m in enumerate(ms):
            r = rec.get(f"{gid}_{m['outcome']}")
            ok = r is not None and "error" not in r
            rows.append(dict(gid=gid, clip=f"{gid}_{m['outcome']}", mode=S.mode_of(gr), gender=gr["actor"]["gender"],
                             outcome=m["outcome"], ok=ok,
                             verdict=(";".join(qc.leg_verdict(r, obj=gr["object"]["asset"])) if ok
                                      else "leg screen failed"),
                             apart=apart_s(gr, r) if ok and i == 0 and "apart" in r else np.nan,
                             rest_gap=float(np.linalg.norm(np.array(r["rest"]) - S.group_rest_box(gr)[0]))
                             if ok and r.get("rest") else np.nan,
                             **({p: v for p, v in r["phases"].items()} if ok else {})))
    return pd.DataFrame(rows)


def light_ok(gr, row):
    """Does the group's light follow its plate's plate_light.csv row (D40)? Source 'table', the row's condition, night
    iff the condition is night, sun iff it is sun; a sun group's azimuth within S.AZ_JIT of the annotation (circular)
    and its elevation within S.EL_JIT of it (clipped to 5-85), or inside S.EL_LAW when none was annotated."""
    L = gr["light"]
    if (row is None or L.get("source") != "table" or L.get("condition") != row["condition"]
            or L["night"] != (row["condition"] == "night") or L["sun"] != (row["condition"] == "sun")):
        return False
    if not L["sun"]:
        return True
    daz = abs((L["sun_az"] - float(row["sun_az_deg"]) + 180.0) % 360.0 - 180.0)
    el = float(row["sun_el_deg"]) if row["sun_el_deg"] else None
    lo, hi = (max(5.0, el - S.EL_JIT), min(85.0, el + S.EL_JIT)) if el is not None else S.EL_LAW
    return daz <= S.AZ_JIT + 1e-9 and lo - 1e-9 <= L["sun_el"] <= hi + 1e-9


def light_table(G, light):
    """Per source clip of plate_light.csv: groups, car-mode groups, sun share, the sun groups' azimuth range against the
    annotated azimuth, and groups whose light follows the row (light_ok) with the group dict identical in both members."""
    rows = []
    for gid, (gr, ms, same) in G.items():
        r, L = light.get(gr["plate_id"]), gr["light"]
        rows.append(dict(clip=r["clip"] if r else "(no row)", condition=r["condition"] if r else "", car=S.mode_of(gr) == "car",
                         sun=L["sun"], night=L["night"], az=L["sun_az"] if L["sun"] else np.nan,
                         ann_az=float(r["sun_az_deg"]) if r and r["sun_az_deg"] else np.nan, ok=light_ok(gr, r) and same))
    df = pd.DataFrame(rows)
    return df.groupby("clip").agg(condition=("condition", "first"), groups=("ok", "size"), car=("car", "sum"),
                                  night=("night", "mean"), sun=("sun", "mean"), az_min=("az", "min"), az_max=("az", "max"),
                                  ann_az=("ann_az", "first"), consistent=("ok", "sum")).sort_values("groups", ascending=False)


def gate_table(GR):
    """One row per screened draw (the positive's gate record): mode, gender, object, any leg reason, the crouch reason
    (qc.CROUCH_WHY, D41), the apart estimate; cls = the object if it has a set-down (S.SETDOWN), else other rigid / soft."""
    gt = pd.DataFrame([dict(mode="car" if r["car"] else "foot", gender=r["gender"], obj=r["object"], large=r["object"] in LARGE,
                            leg=bool(set(r["rejected"]) & (GATE - {"object never apart (estimate)", "apart screen failed"})),
                            crouch=qc.CROUCH_WHY in r["rejected"], apart="object never apart (estimate)" in r["rejected"])
                       for k, rs in GR.items() if k.endswith("_leave") for r in rs])
    if len(gt):
        gt["cls"] = np.where(gt.obj.isin(list(S.SETDOWN)), gt.obj,
                             np.where(gt.obj.isin(qc.SOFT_OBJECTS), "other soft", "other rigid"))
    return gt


def qtab(df, col, by, sets, extra=None):
    """Quantiles of df[col] per (by, src) for the sources in sets (extra: another sample for every by, as src 'real'),
    with the KS p value of each against sets[0]."""
    rows = []
    for k, x in df.groupby(by):
        v0 = x[x.src == sets[0]][col].dropna()
        for s, v in [(s, x[x.src == s][col].dropna()) for s in sets] + ([("real", pd.Series(extra))] if extra is not None else []):
            if len(v):
                rows.append(dict(**{by: k}, src=s, n=len(v), **{f"p{int(q * 100)}": float(v.quantile(q)) for q in Q},
                                 KS_p=ks_2samp(v0, v).pvalue if s != sets[0] else np.nan))
    return pd.DataFrame(rows).set_index([by, "src"])


def render_rate():
    """s15-s17 as rendered (all drawn groups; first scenario file -> last clip, by file times): frames per minute
    per mode (s15 car mode, s16 + s17 on foot)."""
    out = {}
    for mode, pre in (("car", ["g015"]), ("foot", ["g016", "g017"])):
        fr = mins = 0.0
        for p in pre:
            sc = sorted((D / "scenarios").glob(f"{p}*.json"))
            G = load(sc)
            fr += sum(len(ms) * int(np.floor(gr["duration"] * gr["render_fps"] + 1e-6)) for gr, ms, _ in G.values())
            mins += (max(f.stat().st_mtime for f in (D / "clips").glob(f"{p}*.mp4")) - min(f.stat().st_mtime for f in sc)) / 60
        out[mode] = (fr, mins)
    return out


def dry(a):
    dirs = [pathlib.Path(x) for x in a.dirs]
    G = {}
    for d in dirs:
        G.update(load(sorted((d / "scenarios").glob("*.json"))))
    LS = {k: v for d in dirs for k, v in log_stats(d).items()}
    GR = {k: v for d in dirs for k, v in gate_records(d).items()}
    t = S.Tables()
    g, m = features(G, "dry")
    print(f"# Tables: {len(dirs)} dry runs, {len(G)} groups planned\n")
    # 1. planning and rejections
    print("## 1. Planning, draws, rejections\n")
    rows = []
    for mode in ("foot", "car"):
        L = {k: v for k, v in LS.items() if v["mode"] == mode}
        why = sum((v["why"] for v in L.values()), collections.Counter())
        draws = sum(v["draws"] for v in L.values())
        gated = sum(len(GR.get(f"{k}_leave", [])) for k in L)
        gate_rej = sum(bool(r["rejected"]) for k in L for r in GR.get(f"{k}_leave", []))
        rows.append(dict(mode=mode, groups=len(L), planned=sum(v["planned"] for v in L.values()),
                         skipped=sum(not v["planned"] for v in L.values()), draws=draws, per_group=draws / max(1, len(L)),
                         no_plan=why["no plan"], screen_rejected=draws - gated - why["no plan"], gated=gated,
                         gate_rejected=gate_rej))
    print(md(pd.DataFrame(rows).set_index("mode")))
    rows = []
    for mode in ("foot", "car"):
        L = [v for v in LS.values() if v["mode"] == mode]
        draws = sum(v["draws"] for v in L)
        for r, n in sum((v["why"] for v in L), collections.Counter()).most_common():
            rows.append(dict(mode=mode, reason=r, kind="gate" if r in GATE else "sampler" if r == "no plan" else "screen",
                             draws=n, share_of_draws=n / draws))
    print("Rejected draws by reason (a draw can have several reasons):\n")
    print(md(pd.DataFrame(rows).set_index(["mode", "reason"])))
    gt = gate_table(GR)
    if len(gt):
        print("Gate rejections of screened draws by mode x gender (leg = any leg reason, crouch = the crouch-phase rule "
              "of D41, apart = the apart estimate):\n")
        print(md(gt.groupby(["mode", "gender"]).agg(draws=("leg", "size"), leg=("leg", "mean"), crouch=("crouch", "mean"),
                                                    apart=("apart", "mean"), large_draws=("large", "sum"),
                                                    large_leg=("leg", lambda x: x[gt.loc[x.index, "large"]].mean()))))
        print("Crouch rejections of screened draws by mode, by object class x gender (objects with a set-down, "
              f"S.SETDOWN {S.SETDOWN}; other rigid / soft: qc.crouch_th {qc.CROUCH_TH_RIGID} / {qc.CROUCH_TH_SOFT} m):\n")
        print(md(gt.groupby("mode").agg(draws=("crouch", "size"), crouch=("crouch", "mean"))))
        print(md(gt.groupby(["cls", "gender"]).agg(draws=("crouch", "size"), crouch=("crouch", "mean"))))
    # 2. scenes and plates
    print("## 2. Scenes and plates\n")
    usable = {"foot": set(t.plates.scene_id), "car": set(t.plates[t.plates.plate_id.map(t.still_car)].scene_id)}
    rows = []
    for mode, x in g.groupby("mode"):
        vc = x.scene.value_counts()
        rows.append(dict(mode=mode, groups=len(x), scenes_usable=len(usable[mode]), scenes_used=len(vc),
                         target=1 / len(vc), max_share=vc.iloc[0] / len(x), min_share=vc.iloc[-1] / len(x),
                         plates_used=x.plate.nunique(), max_plate_share=x.plate.value_counts().iloc[0] / len(x),
                         night=x.night.mean(), meva=x.meva.mean()))
    print(md(pd.DataFrame(rows).set_index("mode")))
    sc = g.groupby(["scene", "mode"]).size().unstack(fill_value=0)
    for mode in sc.columns:
        sc[f"{mode}_share"] = sc[mode] / sc[mode].sum()
        sc[f"{mode}_target"] = np.where(sc[mode] > 0, 1 / (sc[mode] > 0).sum(), 0.0)
    sc["all_share"] = sc[[c for c in ("foot", "car") if c in sc]].sum(1) / len(g)
    sc["fold"] = g.groupby("scene").fold.first()
    print("Per scene (target = uniform over the scenes the mode uses; the scene cap is 2x that):\n")
    print(md(sc.sort_values("all_share", ascending=False)))
    print("Scenes with usable plates but no group: foot", sorted(usable["foot"] - set(g[g["mode"] == "foot"].scene)),
          "| car", sorted(usable["car"] - set(g[g["mode"] == "car"].scene)), "\n")
    seed = g.gid.str[1:5]
    print("Max groups per plate in one run (MAX_USES", S.MAX_USES, "):", int(g.groupby([seed, "plate"]).size().max()),
          "| fold shares:", g.fold.value_counts(normalize=True).round(3).to_dict(), "\n")
    # 3. families, gender x object, night, distractors
    print("## 3. Families, gender x object, night, distractors\n")
    print("Car-mode share", round((g["mode"] == "car").mean(), 3), "| negative families (share within mode; NEG_WEIGHTS",
          S.NEG_WEIGHTS, "):\n")
    print(md(g.groupby("mode").neg.value_counts(normalize=True).rename("share").to_frame()))
    print("Gender x object (share within gender; n per gender in the last row):\n")
    ct = pd.crosstab(g.obj, g.gender, normalize="columns")
    ct.loc["n"] = g.gender.value_counts()
    ct.loc["large bags"] = g.groupby("gender").large.mean()
    print(md(ct))
    print("Night share: batch", round(g.night.mean(), 4), "|", g.groupby("mode").night.mean().round(4).to_dict(),
          "| night scenes on foot", g[g.night].scene.value_counts().to_dict(), "\n")
    lt = light_table(G, t.light)
    print(f"Light (plate_light.csv, D40 / D42): sun in {g.sun.mean():.4f} of the groups ({g[~g.night].sun.mean():.4f} of day "
          f"groups); groups whose light follows their plate's row, identical in both members: {int(lt.consistent.sum())} of "
          f"{len(g)}. Per source clip (az in deg; a sun group must be within {S.AZ_JIT} of ann_az):\n")
    print(md(lt, 2))
    print("Distractors per group (share with >= 1, mean):", g.groupby("mode").n_dis.agg(
        lambda x: f"{(x > 0).mean():.3f} / {x.mean():.3f}").to_dict(), "| groups whose member files have identical group "
        "dicts:", int(g.same.sum()), "of", len(g), "\n")
    print("Per label (members with >= 1 distractor):\n")
    print(md(m.groupby(["mode", "y"]).n_dis.agg(members="size", share=lambda x: (x > 0).mean(), mean="mean")))
    print("Distractor groups by scene (on foot):", g[(g["mode"] == "foot") & (g.n_dis > 0)].scene.value_counts().to_dict(), "\n")
    # 4. timing laws and size vs s15-s17, release v2, real
    print("## 4. Timing laws and actor size vs s15-s17, release v2 and the real law\n")
    refs = {s: features(old_groups(s), s) for s in ("s15-s17", "v2")}
    ga = pd.concat([g] + [v[0] for v in refs.values()]); ma = pd.concat([m] + [v[1] for v in refs.values()])
    sets = ["dry", "s15-s17", "v2"]
    for col in ("onset", "d", "t_event", "after"):
        print(f"{col} (positives, s):\n")
        print(md(qtab(ga, col, "mode", sets), 2))
    idx = pd.read_csv(SYN.parent / "cache/index.csv"); rp = idx[idx.y == 1]
    print("Real contest law: T p10/50/90", np.round(idx["T"].quantile([.1, .5, .9]).values, 1), "| onset g of positives",
          np.round(rp.g.quantile([.1, .5, .9]).values, 1), "| after onset < 2 s", round(float(((rp["T"] - rp.g) < 2).mean()), 3),
          "< 3 s", round(float(((rp["T"] - rp.g) < 3).mean()), 3), "\n")
    print("After onset < 2 s / < 3 s (positives):",
          {f"{k[0]}/{k[1]}": f"{(x < 2).mean():.3f} / {(x < 3).mean():.3f}" for k, x in ga.groupby(["mode", "src"]).after}, "\n")
    print("t_dec null and clip ends before the walk-out, by mode x outcome:\n")
    print(md(ma.groupby(["mode", "outcome", "src"]).agg(n=("y", "size"), t_dec_null=("t_dec_null", "mean"),
                                                         before_walkout=("before_walkout", "mean"))))
    print("Actor relative size at the spot (person px height / image height):\n")
    print(md(qtab(ga, "rel", "mode", sets, t.real_rel_h)))
    # 5. per planned member: screen, structures, bars, legs
    print("## 5. Every planned member: strict screen, structures, bars, leg gate\n")
    sc = screen_all(G, a.workers)
    bad = {k: v[0] for k, v in sc.items() if v[0]}
    print("Strict screen re-run (postqc.check strict, vehicle masks): groups with a reason", len(bad), bad or "", "\n")
    print("Structure check (postqc.structure_hit): groups in conflict", sum(bool(v[1]) for v in sc.values()),
          "| no raster", sum(v[1] is None for v in sc.values()), "\n")
    b = pd.DataFrame([dict(gid=k, mode=S.mode_of(G[k][0]), **v[2]) for k, v in sc.items()])
    bb = b[b.bar_plate]
    print(f"Bars: {len(bb)} groups on plates with bars ({bb.merge(g, on='gid').plate.nunique()} plates); px in a bar: actor at "
          f"the spot max {bb['spot'].max() if len(bb) else 0:.0f}, object rest box max {bb['obj'].max() if len(bb) else 0:.0f}, car place S max "
          f"{bb['S'].max() if len(bb) else 0:.0f}; distractors with feet in a bar {int(bb.dis_feet.sum())}; groups with a walker "
          f"crossing a bar (> 30 px, allowed) {int((bb.cross > 30).sum())}, of them with sun "
          f"{int((bb.merge(g, on='gid').query('cross > 30').sun).sum())}\n")
    if a.legs:
        files = [f for d in dirs for f in sorted((d / "scenarios").glob("*.json"))]
        rec = legs(files, a.workers)
        lr = leg_rows(G, rec)
        same = sum(GR.get(c) is not None and GR[c][-1]["hits"] == rec[c]["hits"]
                   and GR[c][-1].get("apart") == rec[c].get("apart") for c in lr["clip"] if c in rec)
        print(f"Leg gate, fresh leg_overlap.py run: {len(lr)} members, records {int(lr.ok.sum())}, failing qc.leg_verdict "
              f"{int((lr.verdict != '').sum())} {lr[lr.verdict != '']["clip"].tolist()[:20]}, positives with the object apart "
              f"< {qc.MIN_APART_S} s {int((lr.apart < qc.MIN_APART_S).sum())}; hits and apart identical to the gate "
              f"records {same} of {len(lr)}")
    else:
        lr = leg_rows(G, {c: rs[-1] for c, rs in GR.items()})
        print(f"Leg gate, accepted draws' gate records: {len(lr)} members, failing {int((lr.verdict != '').sum())}")
    print(f"; rest point vs scenario.rest_box median {lr.rest_gap.median() * 1000:.1f} mm, max {lr.rest_gap.max() * 1000:.1f}"
          " mm\n")
    ph = [p for p in ("stride", "stand", "crouch_in", "crouch_out", "settle", "turn", "pause", "crouch2", "walk_out") if p in lr]
    print("Members with leg contact >= 0.05 m by phase (crouch_in, crouch_out, settle, crouch2 are gated at "
          f"qc.crouch_th: rigid {qc.CROUCH_TH_RIGID} m, soft bags {qc.CROUCH_TH_SOFT} m, D41):\n")
    print(md(lr.groupby(["mode", "outcome", "gender"]).agg(n=("clip", "size"), **{p: (p, lambda x: int((x >= qc.LEG_TH).sum()))
                                                                                  for p in ph})))
    # 6. GPU estimate
    print("## 6. GPU time and expected yield\n")
    rr = render_rate()
    old = pd.concat([v[0] for v in refs.values()]).drop_duplicates("gid")
    for mode, (fr, mins) in rr.items():
        print(f"s15-s17 {mode}: {fr:.0f} member frames in {mins:.0f} min = {fr / mins:.0f} frames/min; dry run frames per "
              f"group {g[g['mode'] == mode].frames.mean():.0f} (accepted old pairs {old[old['mode'] == mode].frames.mean():.0f})")
    est = sum(600 * (g["mode"] == mode).mean() * g[g["mode"] == mode].frames.mean() / (fr / mins) for mode, (fr, mins) in rr.items())
    print(f"\nGPU estimate for 600 groups at the dry run's mix: {est:.0f} min ({est / 60:.1f} h)\n")


def retro(a):
    G = old_groups("all439")
    v2 = set(old_pairs("v2")); ex = pd.read_csv(D / "exclude_pairs.csv").set_index("pair_id")
    sc = screen_all(G, a.workers)
    rec = legs([p for pid, bd in old_pairs("all439").items() for p in sorted((SYN / bd / "scenarios").glob(f"{pid}_*.json"))],
               a.workers) if a.legs else {}
    lr = leg_rows(G, rec).groupby("gid").agg(legs=("verdict", lambda x: ";".join(sorted({r for v in x for r in v.split(";") if r}))),
                                             apart=("apart", "max")) if rec else None
    rows = []
    for gid, (gr, ms, _) in G.items():
        why, sh, _ = sc[gid]
        lg = lr.loc[gid, "legs"] if lr is not None else ""
        ap = lr.loc[gid, "apart"] if lr is not None else np.nan
        reasons = list(why) + ([r for r in lg.split(";") if r]) + (["object never apart (estimate)"] if ap < qc.MIN_APART_S else [])
        rows.append(dict(pair_id=gid, plate_id=gr["plate_id"], scene_id=gr["scene_id"], release_v2=gid in v2,
                         excluded_d29=gid in ex.index, d29_source=ex.source.get(gid, ""), screen=";".join(why),
                         structure_parts=";".join(sh or []), legs=lg, apart_s=ap, reasons=";".join(reasons),
                         flagged=bool(set(reasons) - {"distractor not approved"})))
    R = pd.DataFrame(rows).sort_values("pair_id")
    R.to_csv(a.out, index=False)
    print(f"# Retro: {len(R)} accepted pairs (release v2 {R.release_v2.sum()}, D29 excluded {R.excluded_d29.sum()})\n")
    for k, x in (("release v2", R[R.release_v2]), ("D29 excluded", R[R.excluded_d29])):
        c = collections.Counter(r for v in x.reasons for r in v.split(";") if r)
        print(f"{k}: {len(x)} pairs, flagged (any reason but 'distractor not approved') {int(x.flagged.sum())}, with any "
              f"reason {int((x.reasons != '').sum())}; by reason {dict(c.most_common())}\n")
    print("Release v2 pairs flagged, by reason:\n")
    for r in sorted({r for v in R[R.release_v2].reasons for r in v.split(";") if r}):
        ids = R[R.release_v2 & R.reasons.str.split(";").apply(lambda v: r in v)].pair_id.tolist()
        print(f"- {r} ({len(ids)}): {' '.join(ids)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dry"); p.add_argument("dirs", nargs="+")
    p = sub.add_parser("retro"); p.add_argument("out")
    for p in sub.choices.values():
        p.add_argument("--legs", action="store_true"); p.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    dry(a) if a.cmd == "dry" else retro(a)
