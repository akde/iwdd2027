"""Batch: scenarios -> renders -> clean composites -> shared degradation -> QC -> clips, labels, manifest.
python synth/gen/run_batch.py --groups N --seed S --data-dir DIR [--workers 3] [--samples 16] [--keep-renders]
       [--plan-only] [--reuse [--force]] [--plates PLATE_ID ...] [--families place_leave unload_vehicle_leave]
       [--car-share 0.6] [--night-share 0.1] [--scene-cap 2] [--light-fallback]
Per group (1 positive + 1 negative, DECISIONS.md D14), in the batch directory DIR (required, a new one per batch: a DIR
with a synth_manifest.csv, e.g. synth/data with s15-s17, is refused without --reuse; plate data, calibration and masks
are always read from synth/data and synth/calib):
  DIR/scenarios/<gid>_<outcome>.json        scenario (gen/scenario.py)
  DIR/renders/<gid>/<outcome>/              passes (gen/build_scene.py; PNGs deleted after compositing)
  DIR/clean/<gid>_<outcome>.mp4             clean composite (gen/composite.py)
  DIR/clips/<gid>_<outcome>.mp4             degraded clip (gen/degrade.py, one parameter draw per group)
  DIR/labels/<gid>_<outcome>.json           contest label keys + synthetic fields
  DIR/synth_manifest.csv                    one row per member, accepted and rejected (qc column)
Before rendering, each group is screened (postqc.check, strict; then, in rounds over all drawn groups, screen_gate in
Blender without rendering: every member's and distractor's legs against the resting object, qc.leg_verdict (F1, D33;
crouching members at the object's crouch depth, D41),
and the positive's object seen apart from the actor (F5, D32)) and redrawn with the same families, car mode, actor
gender and night flag until it passes; after MAX_DRAWS failed draws its group id is skipped (no scenario, no render).
The gate's leg_overlap records are appended to DIR/legcheck.jsonl (the last line per clip is the accepted draw's).
After rendering, run_group drops a group if a member fails the post-render QC (group-level reasons, such as the
positive's object never seen apart from the actor, are given to both members). --plan-only stops
after writing the scenarios. --reuse resumes a batch (D38): the scenarios already in DIR are kept, groups its manifest
holds in full are skipped, renders and composites an interrupted group left are reused (--force: redo finished groups,
re-compositing renders kept with --keep-renders). A render has a time limit; a group that fails is retried once from
scratch, then written to the manifest as failed (qc False, 'failed: ...'); after MAX_FAIL_STREAK failures in a row the
batch stops; the exit code is 1 if any group failed. --plates restricts sampling to the given plates (pilots). The
families default to both, so --car-share (default 0.6, the batch default since D34) is the share of groups in car mode
(unload_vehicle_leave). Only plates with vetted structure masks are sampled
(scenario.Tables, structures.usable: fail-closed; the excluded plates are printed). Night (D35): in a mode whose usable
night plates span >= 2 scenes (on foot; car mode has one night plate; with --plates one scene is enough, D38) each
group is a night group with probability
--night-share (default 0.10, the real share), else a day group, and draws only plates of its kind; with 60% car mode
the batch is then about 5% night (--night-share 0.25 gives 10% of the batch, at 25% of the on-foot groups).
--scene-cap: a scene takes a group only while it stays <= scene-cap x the mean count over the scenes its mode uses in
the batch (scenario.Tables.scenes; 0 = off).
Distractors (F3, D36; in car mode too since D38): only on plates with >= scenario.MIN_LINES approved lines of
synth/data/distractor_lines.json (the count is printed). Light (D40): only plates with a usable row of
synth/data/plate_light.csv (synth/scripts/plate_light.py) are sampled, the sun following the plate's real shadows;
--light-fallback also samples the others with the old random sun. Blender runs in a memory-capped scope (DECISIONS.md
D10)."""
import argparse, collections, json, math, pathlib, subprocess, sys, shutil, threading, traceback
import concurrent.futures as cf
import numpy as np, pandas as pd

GEN = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(GEN))
import scenario as S, composite, degrade, qc
from camera import Camera, group_calib, plate_calib
from tracks import moving

SYN = GEN.parent
D = SYN / "data"  # shared plate data; the batch's own outputs go to the batch directory (--data-dir)
MAX_DRAWS = 20
BLENDER = pathlib.Path.home() / "opt/blender-4.2.23-linux-x64/blender"
# failures (D38, review R3): a render gets max(RENDER_MIN_S, RENDER_S_PER_FRAME x frames) s (s15-s17: about 0.4 s
# per frame and process with 5 workers); a gate process GATE_S_PER_FILE s per scenario (about 1-3 s), at least
# GATE_MIN_S; a failed group is retried once from scratch, then written to the manifest (qc False, 'failed: ...'); after
# MAX_FAIL_STREAK failed groups in a row the batch stops; any failure makes the exit code 1
RENDER_MIN_S, RENDER_S_PER_FRAME, GATE_MIN_S, GATE_S_PER_FILE, MAX_FAIL_STREAK = 1200, 2.0, 600, 30, 5
EVENT_TYPE = {"static": "static dumping", "dynamic": "dynamic dumping"}


def render(sc_path, out, samples):
    cmd = ["systemd-run", "--user", "--scope", "-p", "MemoryMax=10G", "-p", "MemorySwapMax=0", "--quiet",
           str(BLENDER), "-b", "-P", str(GEN / "build_scene.py"), "--", str(sc_path), str(out), "--samples", str(samples)]
    gr = json.loads(pathlib.Path(sc_path).read_text())["group"]
    limit = max(RENDER_MIN_S, RENDER_S_PER_FRAME * math.floor(gr["duration"] * gr["render_fps"] + 1e-6))
    log = out / "blender.log"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=limit)
    except subprocess.TimeoutExpired as e:  # the child is killed; a hung Blender must not block its worker
        log.write_text(str(e.stdout or "")[-20000:] + "\n" + str(e.stderr or "")[-20000:])
        raise RuntimeError(f"blender timed out after {limit:.0f} s: see {log}")
    log.write_text(r.stdout[-20000:] + "\n" + r.stderr[-20000:])
    if r.returncode != 0 or "RENDERED" not in r.stdout:
        raise RuntimeError(f"blender failed ({r.returncode}): see {log}")


def labels(gr, m, deg_meta, area, qc_row):
    y = m["y"]
    return {"label": "true" if y else "false", "timestamp": m["timestamp"] if y else -1,
            "metadata": {"video_id": f"{gr['group_id']}_{m['outcome']}.mp4", "resolution": deg_meta["resolution"],
                         "frame_rate": deg_meta["frame_rate"], "duration": deg_meta["duration"],
                         "event_type": EVENT_TYPE[S.EVENT[m["family"]]] if y else "normal", "time_of_day": "night" if gr["light"]["night"] else "day",
                         "lighting": "low" if gr["light"]["dim"] else "high"},
            "t_onset_exact": m["t_onset_exact"], "t_release": m["events"].get("t_release"), "t_dec": m["t_dec"],
            "family": m["family"], "outcome": m["outcome"], "pair_id": gr["group_id"], "plate_id": gr["plate_id"],
            "scene_id": gr["scene_id"], "fold": gr["fold"], "seed": gr["seed"], "object_px_area_at_rest": area,
            "object": gr["object"]["asset"], "avatar": gr["actor"]["avatar"], "divergence": m["divergence"],
            "events": m["events"], "qc": qc_row}


def run_group(gr, ms, samples, keep, reuse=False, bd=D, force=False):
    """Render, composite, QC and label one group (ms: positive first) -> its manifest rows. reuse (resume): keep the
    renders and, unless force, the composites (clean clip and masks.npz) an interrupted run left."""
    gid = gr["group_id"]
    rows, ok, clean = [], True, {}
    plates = pd.read_csv(D / "plates.csv").set_index("plate_id")
    cal = group_calib(gr)
    W, H = gr["render_size"]
    rng = np.random.default_rng(gr["degrade_seed"])
    p = degrade.sample_params(rng, W, H, gr["render_fps"])
    if not reuse:  # the group's look is removed once, then fixed by the positive and read by every member (before
        (bd / f"renders/{gid}/look.json").unlink(missing_ok=True)  # C5 it was removed per member: review S2, D38)
    for m in ms:  # positive first: it fixes the group's look (composite.make_look)
        sc_path = bd / f"scenarios/{gid}_{m['outcome']}.json"
        rd = bd / f"renders/{gid}/{m['outcome']}"
        if not reuse:  # derived state of an earlier run of the same group id must not leak in
            shutil.rmtree(rd, ignore_errors=True)
        rd.mkdir(parents=True, exist_ok=True)
        cp = bd / f"clean/{gid}_{m['outcome']}.mp4"; cp.parent.mkdir(exist_ok=True)
        clean[m["outcome"]] = cp
        if reuse and not force and cp.exists() and (rd / "masks.npz").exists():
            continue  # composited before the interruption (masks.npz is written after the clip is closed)
        if not (reuse and (rd / "meta.json").exists() and any(rd.glob("rgba_*.png"))):
            render(sc_path, rd, samples)
        composite.main(sc_path, rd, cp)
    # QC (group level)
    scale = p["height"] / H
    pr = plates.loc[gr["plate_id"]]
    free = qc.person_free(gr["video"], gr["plate_id"], gr["plate_start"], gr["plate_start"] + gr["duration"], cal)
    pos = ms[0]
    frames, times = degrade.decode(clean[pos["outcome"]])
    # actor standing at the spot: the last frame before the bend (idle pose, cross-fade from walking finished)
    k = min(len(frames) - 1, int(np.floor(pos["events"]["t_event"] * gr["render_fps"])) - 1)
    cam = Camera(cal, (W, H))
    foot = cam.project(np.array([[*gr["path"]["X"], 0.0]]))[0][0]
    pmask = np.load(bd / f"renders/{gid}/{pos['outcome']}/masks.npz")
    hr = qc.mask_height_ratio(pmask, k, cam, gr["path"]["X"], gr["actor"]["height_m"])
    ks = qc.event_frames(pmask, pos, gr["render_fps"])
    vis = qc.visibility(frames, composite.plate_frames(pr.path, gr["plate_start"], [k_ / gr["render_fps"] for k_ in ks],
                                                       (W, H)), pmask["act_box"], ks)
    yr = qc.height_ratio(frames[k], plate_calib(gr["plate_id"]), foot, (W, H),  # real people's fit boxes (info only)
                         pmask["act_box"][k] if pmask["act_box"][k][0] >= 0 else None)
    apart = qc.object_apart_s(pmask, gr["render_fps"], scale)  # the positive's released object seen apart (D32)
    for m in ms:
        rd = bd / f"renders/{gid}/{m['outcome']}"
        masks = np.load(rd / "masks.npz")
        area = qc.object_area(masks, m, scale)
        view = qc.actor_in_view(masks, m, gr["render_fps"])
        why = []
        if area is not None and area < qc.MIN_OBJ_PX:
            why.append("object small")
        if view < 0.9:
            why.append("actor out of view")
        if not free:
            why.append("real person")
        if m["y"] == 0 and m["divergence"] is not None and m["divergence"] >= gr["duration"] - 0.5:
            why.append("diverges after the end")  # would duplicate the positive with the opposite label
        if m["outcome"] in ("pickup", "preexisting") and m["t_dec"] is None:
            why.append("outcome after the end")
        if vis < qc.MIN_VISIBILITY:
            why.append("actor not visible")
        if apart is not None and apart < qc.MIN_APART_S:
            why.append("object never apart")  # group level, read from the positive: both members are dropped
        if hr is None or abs(hr - 1) > qc.MAX_HEIGHT_ERR:
            why.append("height" if hr is not None else "actor not visible")
        rows.append(dict(m=m, area=area, view=view, why=why))
        ok &= not why
    out = []
    for r in rows:
        m = r["m"]; name = f"{gid}_{m['outcome']}"
        q = dict(ok=ok, why=";".join(r["why"]), object_px=r["area"], actor_view=r["view"], height_ratio=hr,
                 yolo_height_ratio=yr, person_free=free, visibility=vis, object_apart_s=apart)
        dm = dict(resolution="", frame_rate=0, duration=0)
        if ok:
            f, t = degrade.decode(clean[m["outcome"]])
            fr, tt = degrade.degrade_frames(f, t, p, np.random.default_rng(gr["degrade_seed"] + 1))
            cp = bd / f"clips/{name}.mp4"; cp.parent.mkdir(exist_ok=True)
            degrade.encode(fr, tt, cp, p["crf"], p["fps"])
            step = float(np.median(np.diff(tt))) if len(tt) > 1 else 1 / p["fps"]
            dm = dict(resolution=f"{p['width']}x{p['height']}", frame_rate=round(p["fps"], 3),
                      duration=round(float(tt[-1] + step), 3))
            lab = labels(gr, m, dm, r["area"], q)
            lp = bd / f"labels/{name}.json"; lp.parent.mkdir(exist_ok=True)
            lp.write_text(json.dumps(lab, indent=1))
        out.append(dict(clip=name, y=m["y"], family=m["family"], outcome=m["outcome"], pair_id=gid,
                        plate_id=gr["plate_id"], scene_id=gr["scene_id"], fold=gr["fold"], source=gr["source"],
                        avatar=gr["actor"]["avatar"], object=gr["object"]["asset"], timestamp=m["timestamp"],
                        t_onset_exact=m["t_onset_exact"], t_dec=m["t_dec"], duration=dm["duration"],
                        frame_rate=dm["frame_rate"], resolution=dm["resolution"],
                        time_of_day="night" if gr["light"]["night"] else "day",
                        lighting="low" if gr["light"]["dim"] else "high", crf=p["crf"], qc=ok, qc_why=q["why"],
                        object_px=r["area"], height_ratio=hr, yolo_height_ratio=yr, object_apart_s=apart))
    if not keep:
        for m in ms:
            for f in (bd / f"renders/{gid}/{m['outcome']}").glob("*.png"):
                f.unlink()
    return out


REST_TOL = 0.03  # m: rest point measured in Blender vs scenario.rest_box (a warning: the verdict is Blender's)
CARRY_TOL = 0.01  # m: carried object measured in Blender outside structures.carry_box's envelope (a warning, D38)


def carry_excess(r, m, scale):
    """How far (m) the carried object measured in Blender (leg_overlap.py 'carry': ground point forward / left in the
    actor's frame, bottom height; the crouch phases left out, as in the envelope's probe) lies outside the envelope of
    structures.carry_box for member m (walking or standing by tracks.moving; x the actor's scale); 0 inside or
    without records."""
    c = [x for x in r.get("carry") or [] if qc.leg_phase(r["events"], r["outcome"], x[0]) not in
         ("crouch_in", "crouch_out", "crouch2")]
    if not c:
        return 0.0
    c, st = np.array(c, float), S.structures
    w = moving(m["root_keys"], c[:, 0], st.CARRY_SETTLE)
    env = np.where(w[:, None, None], np.array([st.CARRY_F, st.CARRY_L, st.CARRY_TOP]), np.array(st.CARRY_STAND)) * scale
    ex = [env[:, 0, 0] - c[:, 1], c[:, 1] - env[:, 0, 1], env[:, 1, 0] - c[:, 2], c[:, 2] - env[:, 1, 1]]
    top, k = c[:, 3] + r["dims"][2], c[:, 3] > 1e-3  # the top only where the object is not on the ground
    if k.any():
        ex += [(env[:, 2, 0] - top)[k], (top - env[:, 2, 1])[k]]
    return float(max(0.0, max(e.max() for e in ex)))


def blender_screen(jobs, tmp, workers, keep=None):
    """leg_overlap.py on [(scenario path, apart)] in up to `workers` Blender processes (no rendering, CPU, each in a
    memory-capped scope; the --apart files, about 3x slower, get their share of processes) -> {clip: record}. A
    process is killed after max(GATE_MIN_S, GATE_S_PER_FILE x its files) s (its members without a record are then
    rejected by the caller: fail-closed redraw); its output goes to tmp/<name>_<i>.log, and the log of a process that
    left a member without a record is copied to the directory keep (D38, review R6)."""
    pos, neg = [p for p, a in jobs if a], [p for p, a in jobs if not a]
    kp = workers if not neg else 0 if not pos else max(1, min(workers - 1, round(workers * 2.0 * len(pos) / (2.0 * len(pos) + 0.7 * len(neg)))))
    procs = []
    for name, flag, files, k in (("apart", ["--apart"], pos, kp), ("legs", [], neg, max(1, workers - kp))):
        for i in range(min(k, len(files))):
            cmd = ["systemd-run", "--user", "--scope", "-p", "MemoryMax=6G", "-p", "MemorySwapMax=0", "--quiet",
                   str(BLENDER), "-b", "-P", str(SYN / "scripts/leg_overlap.py"), "--", str(tmp / f"{name}_{i}.jsonl"),
                   *flag, *map(str, files[i::k])]
            with (tmp / f"{name}_{i}.log").open("w") as lf:
                pr = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
            procs.append((pr, f"{name}_{i}", files[i::k]))
    for pr, name, files in procs:  # a watchdog kills a hung process (Popen.wait(timeout) would leave it running)
        dog = threading.Timer(max(GATE_MIN_S, GATE_S_PER_FILE * len(files)), getattr(pr, "kill", lambda: None))
        dog.start(); pr.wait(); dog.cancel()
    res = {}
    for f in tmp.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            r = json.loads(line); res[r["clip"]] = r
    for pr, name, files in procs:
        if keep is not None and any(pathlib.Path(f).stem not in res for f in files) and (tmp / f"{name}.log").exists():
            keep.mkdir(parents=True, exist_ok=True)
            shutil.copy(tmp / f"{name}.log", keep / f"{len(list(keep.glob('*.log'))):04d}_{name}.log")
    return res


def screen_gate(cands, bd, workers=8, legs=True):
    """Pre-render screen in Blender without rendering (leg_overlap.py) of drawn groups [(group, members), positive
    first] -> {group_id: reason}. The positive's released object must be seen apart from the actor for >=
    qc.MIN_APART_S s at the group's degraded scale (--apart: meshes projected and z-buffered, which reproduced the
    rendered object pixels and actor heights of old renders; F5 review, D32). legs: every member's legs and every
    distractor's against the resting object, qc.leg_verdict (F1, D33; the crouch phases at qc.crouch_th of the group's
    object, D41); the measured rest point is compared with
    scenario.group_rest_box (> REST_TOL m prints a warning), the carried object with structures.carry_box's envelope
    (carry_excess > CARRY_TOL m prints a warning, D38). Records go to bd/legcheck.jsonl with the verdict; the
    log of a Blender process that left a member without a record goes to bd/screen_logs/ (D38).
    Fail-closed: a member without a result rejects its group."""
    tmp = bd / "screen"; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    jobs = []
    for gr, ms in cands:
        for i, m in enumerate(ms if legs else ms[:1]):
            jobs.append((tmp / f"{gr['group_id']}_{m['outcome']}.json", i == 0))
            jobs[-1][0].write_text(json.dumps(dict(group=gr, member=m)))
    res = blender_screen(jobs, tmp, workers, bd / "screen_logs")
    bad = {}
    with (bd / "legcheck.jsonl").open("a") as log:
        for gr, ms in cands:
            gid, why = gr["group_id"], []
            r = res.get(f"{gid}_{ms[0]['outcome']}")
            if r is None or "error" in r or "apart" not in r:
                why.append("apart screen failed")
            else:
                W, H = gr["render_size"]
                scale = degrade.sample_params(np.random.default_rng(gr["degrade_seed"]), W, H, gr["render_fps"])["height"] / H
                n = sum(px * scale * scale >= qc.MIN_OBJ_PX for _, px, _, _ in r["apart"])
                if n / gr["render_fps"] < qc.MIN_APART_S:
                    why.append("object never apart (estimate)")
            for m in (ms if legs else []):
                r = res.get(f"{gid}_{m['outcome']}")
                if r is None or "error" in r:
                    why.append("leg screen failed"); continue
                r["verdict"] = qc.leg_verdict(r, obj=gr["object"]["asset"]); why += r["verdict"]
                if r["rest"] is not None:
                    r["rest_gap"] = round(float(np.linalg.norm(np.array(r["rest"]) - S.group_rest_box(gr)[0])), 4)
                    if r["rest_gap"] > REST_TOL:
                        print(r["clip"], f"WARNING rest point {r['rest_gap']:.3f} m from scenario.rest_box", flush=True)
                r["carry_excess"] = round(carry_excess(r, m, gr["actor"]["scale"]), 4)
                if r["carry_excess"] > CARRY_TOL:
                    print(r["clip"], f"WARNING carried object {r['carry_excess']:.3f} m outside structures.carry_box",
                          flush=True)
            for m in (ms if legs else ms[:1]):
                r = res.get(f"{gid}_{m['outcome']}")
                if r is not None:
                    log.write(json.dumps(dict(r, group_id=gid, family=m.get("family"), car="vehicle" in gr,
                                              gender=gr["actor"]["gender"] if "actor" in gr else None,
                                              object=gr["object"]["asset"] if "object" in gr else None,
                                              rejected=sorted(set(why)))) + "\n")
            if why:
                bad[gid] = "; ".join(sorted(set(why)))
    shutil.rmtree(tmp, ignore_errors=True)
    return bad


def apart_gate(cands, bd, workers=8):
    """screen_gate without the leg screen: {group_id: reason} for drawn groups whose positive shows its released
    object apart from the actor for < qc.MIN_APART_S s (estimate, D32)."""
    return screen_gate(cands, bd, workers, legs=False)


def draw(rng, t, gid, foot, car, fam=None, gender=None, night=None):
    """One group draw: car mode or on foot (families `foot`); fam = [positive, negative] families and the actor's
    gender to keep (the leg gate was gender-dependent before D35, D33); night: night or day plates only (None: any)."""
    if fam is None:
        return S.sample_vehicle_group(rng, t, gid, night=night) if car else S.sample_group(rng, t, gid, foot, night=night)
    w = [float(n == fam[1]) for n in S.NEG_OF[fam[0]]]
    return S.sample_vehicle_group(rng, t, gid, w, gender, night) if car else \
        S.sample_group(rng, t, gid, (fam[0],), w, gender, night)


def plan(a, rng, t, obst, bd, check, gate=None):
    """Scenarios of a.groups groups -> bd/scenarios; returns [(group, members)], positive first. A draw that the
    pre-render screen check(...) rejects is redrawn with the first draw's families, car mode, actor gender and night
    flag (the screens must not shift the family, gender or night mix); gate(drawn) -> {group_id: reason}
    (screen_gate) then screens the drawn groups in rounds and a rejected group is redrawn the same way; after MAX_DRAWS
    rejected draws in all the group id is skipped. A draw that finds no plate (the sampler raises scenario.NoPlan)
    counts as rejected ('no plan') and its families and gender are kept as well (D38). Only accepted draws count
    towards a plate's MAX_USES and the scene cap (t.take / t.unuse); with a.reuse the groups reused from bd count too.
    The night flag is drawn with the car mode at a group's first draw, before any family or outcome: night with
    probability a.night_share in a mode where t.night_feasible (with a.plates, a pilot's list, one night scene is
    enough, D38), else day. Without a.night_share any plate (scenario.py's own CLI, tests)."""
    sd = bd / "scenarios"; sd.mkdir(parents=True, exist_ok=True)
    foot = tuple(f for f in a.families if f != "unload_vehicle_leave")
    jobs, todo = {}, []
    for k in range(a.groups):
        gid = f"g{a.seed:03d}{k:04d}"
        old = sorted(sd.glob(f"{gid}_*.json"))
        if a.reuse and old:  # re-composite: the scenarios the existing renders were made from
            sc = [json.loads(p.read_text()) for p in old]
            g = next(x["group"] for x in sc if x["member"]["y"] == 1)
            jobs[gid] = (g, sorted((x["member"] for x in sc if x["group"] == g), key=lambda m: -m["y"]))
            if t is not None and g["plate_id"] in t.uses:  # MAX_USES and the scene cap see the reused groups
                t.uses[g["plate_id"]] += 1; t.scene_uses[(S.mode_of(g), g["scene_id"])] += 1
            continue
        for p in old:  # an earlier run of this group id must not leak in
            p.unlink()
        todo.append(dict(gid=gid, car=None, fam=None, gender=None, night=None, why=collections.Counter(), n=0))
    night_share = getattr(a, "night_share", None)
    min_sc = 1 if getattr(a, "plates", None) else S.NIGHT_MIN_SCENES  # a pilot's plate list: any night plate (R5)
    feasible = {c: t.night_feasible(c, min_sc) for c in (False, True)} if night_share is not None else {}
    if night_share is not None:
        print(f"night probability per group: on foot {night_share * feasible[False]:.2f}, car mode "
              f"{night_share * feasible[True]:.2f} (night plates in >= {min_sc} scenes"
              f"{'; --plates: one night scene is enough' if min_sc < S.NIGHT_MIN_SCENES else ''})", flush=True)

    def skip(st):
        print(st["gid"], f"SKIPPED: {MAX_DRAWS} draws {st['fam']} {'car' if st['car'] else 'on foot'} failed the "
              f"screen: {dict(st['why'])}", flush=True)
    while todo:
        drawn = []
        for st in todo:
            if st["car"] is None:  # car mode decided at the group's first draw (the random stream of before D32
                share = getattr(a, "car_share", 0.5)  # w/o gate; the CLI default is 0.6, D34)
                st["car"] = "unload_vehicle_leave" in a.families and (not foot or rng.random() < share)
                if night_share is not None:  # group level, before any family or outcome is drawn (D35)
                    st["night"] = bool(rng.random() < night_share) and feasible[st["car"]]
            while st["n"] < MAX_DRAWS:
                try:
                    gr = draw(rng, t, st["gid"], foot, st["car"], st["fam"], st["gender"], st["night"])
                except RuntimeError as e:  # no plate of this kind left (night, MAX_USES) or none planned
                    st["n"] += 1; st["why"].update(["no plan"]); print(st["gid"], "draw failed:", e, flush=True)
                    if isinstance(e, S.NoPlan):  # keep what the failed draw drew, as after a screen rejection (D38)
                        st["fam"] = st["fam"] or e.families; st["gender"] = st["gender"] or e.gender
                    continue
                st["n"] += 1
                st["fam"] = st["fam"] or gr["families"]; st["gender"] = gr["actor"]["gender"]
                ms = S.members(gr)
                w = check(gr, ms, obst, vehicle_masks=True, strict=True)
                if not w:
                    drawn.append((st, gr, ms)); break
                st["why"].update(w)
                t.unuse(gr)
            else:
                skip(st)
        bad = gate([(gr, ms) for _, gr, ms in drawn]) if gate is not None and drawn else {}
        todo = []
        for st, gr, ms in drawn:
            gid = st["gid"]
            if gid in bad:
                st["why"].update([bad[gid]]); t.unuse(gr)
                todo.append(st) if st["n"] < MAX_DRAWS else skip(st)
                continue
            for m in ms:
                (sd / f"{gid}_{m['outcome']}.json").write_text(json.dumps(dict(group=gr, member=m), indent=1))
            print(gid, "planned", gr["plate_id"], gr["families"], f"draws={st['n']}", dict(st["why"]) if st["why"] else "",
                  flush=True)
            jobs[gid] = (gr, ms)
    out = [jobs[g] for g in sorted(jobs)]
    for mode in ("foot", "car"):
        g = [gr for gr, _ in out if S.mode_of(gr) == mode]
        if g:
            sc = collections.Counter(gr["scene_id"] for gr in g)
            print(f"{mode}: {len(g)} groups, night {sum(gr['light']['night'] for gr in g) / len(g):.1%} (night plates "
                  f"{ {True: 'feasible', False: 'not feasible'}.get(feasible.get(mode == 'car'), 'not controlled')}), "
                  f"{len(sc)} scenes, largest "
                  f"{sc.most_common(1)[0][0]} {sc.most_common(1)[0][1] / len(g):.1%}", flush=True)
    if t is not None and t.cap_lifted:
        print(f"scene cap lifted for {t.cap_lifted} plate choices (every allowed scene capped)", flush=True)
    return out


def main(a):
    bd = pathlib.Path(a.data_dir)
    rng = np.random.default_rng(a.seed)
    t = S.Tables(light_fallback=getattr(a, "light_fallback", False))  # plate_light.csv, fail-closed (D40)
    t.scene_cap = a.scene_cap or None
    print(f"plates: {len(t.plates)} usable; not sampled (structure masks, plate_vet.csv, plate_light.csv): "
          + (", ".join(f"{p} ({w})" for p, w in sorted(t.excluded.items())) or "none"), flush=True)
    lit = collections.Counter(t.light[p]["condition"] for p in t.plates.plate_id if p not in t.unlit)
    print(f"light: {dict(lit)} from plate_light.csv"
          + (f"; old random light (--light-fallback) on {sum(p in t.unlit for p in t.plates.plate_id)} plates"
             if t.light_fallback else ""), flush=True)
    if t.plates.empty:
        print("no plate can be sampled: annotate the light (synth/scripts/plate_light.py) or pass --light-fallback",
              flush=True)
        return 1
    n = {p: len(t.lines(p)) for p in t.plates.plate_id}  # approved distractor lines (F3, D36: default deny)
    print(f"distractor lines: {sum(n.values())} approved on {sum(v > 0 for v in n.values())} plates; plates with >= "
          f"{S.MIN_LINES} (distractors possible): {sum(v >= S.MIN_LINES for v in n.values())}", flush=True)
    if a.plates:
        assert set(a.plates) <= set(t.plates.plate_id), {p: t.excluded.get(p, "unknown") for p in set(a.plates) - set(t.plates.plate_id)}
        t.plates = t.plates[t.plates.plate_id.isin(a.plates)].reset_index(drop=True)
    sys.path.insert(0, str(SYN / "scripts"))
    import postqc
    obst = json.loads((D / "obstacles.json").read_text())
    jobs = plan(a, rng, t, obst, bd, postqc.check, lambda c: screen_gate(c, bd))
    print(f"planned {len(jobs)} of {a.groups} groups -> {bd / 'scenarios'}", flush=True)
    if a.plan_only:
        return 0
    return render_all(a, jobs, bd)


def finished(man, bd):
    """Group ids the manifest holds in full: both rows, not failed, and the clips and labels of accepted rows exist
    (resume, D38, review R2)."""
    if not len(man):
        return set()
    out = set()
    for gid, g in man.groupby("pair_id"):
        if len(g) == 2 and not g.qc_why.fillna("").astype(str).str.startswith("failed").any() and all(
                (bd / f"clips/{c}.mp4").exists() and (bd / f"labels/{c}.json").exists() for c in g[g.qc]["clip"]):
            out.add(gid)
    return out


def run_safe(gr, ms, a, bd):
    """run_group with one retry from scratch -> (manifest rows, error or None). A group that fails twice gets rows
    qc False, qc_why 'failed: <error>', and its renders, clean clips, clips and labels are removed (D38, review R3)."""
    gid, err = gr["group_id"], None
    for k in range(2):
        try:
            return run_group(gr, ms, a.samples, a.keep_renders, a.reuse and k == 0, bd, getattr(a, "force", False)), None
        except Exception as e:
            err = f"{type(e).__name__}: {e}".replace("\n", " ").replace(";", ",")[:300]
            print(gid, f"FAILED (attempt {k + 1} of 2)\n" + traceback.format_exc(), flush=True)
    shutil.rmtree(bd / f"renders/{gid}", ignore_errors=True)
    for m in ms:
        for sub, ext in (("clean", "mp4"), ("clips", "mp4"), ("labels", "json")):
            (bd / f"{sub}/{gid}_{m['outcome']}.{ext}").unlink(missing_ok=True)
    return [dict(clip=f"{gid}_{m['outcome']}", y=m["y"], family=m["family"], outcome=m["outcome"], pair_id=gid,
                 plate_id=gr["plate_id"], scene_id=gr["scene_id"], fold=gr["fold"], source=gr["source"],
                 avatar=gr["actor"]["avatar"], object=gr["object"]["asset"], timestamp=m["timestamp"],
                 t_onset_exact=m["t_onset_exact"], t_dec=m["t_dec"], qc=False, qc_why=f"failed: {err}")
            for m in ms], err


def render_all(a, jobs, bd):
    """Render, composite, QC and label the planned groups in a.workers threads; the manifest is rewritten after each
    group. With a.reuse (resume) groups the manifest holds in full are skipped unless a.force. Returns 1 if a group
    failed or the batch stopped after MAX_FAIL_STREAK failures in a row, else 0."""
    man = bd / "synth_manifest.csv"
    done = pd.read_csv(man, dtype={"qc_why": str}) if man.exists() else pd.DataFrame()
    if a.reuse and not getattr(a, "force", False):
        fin = finished(done, bd)
        if fin:
            print(f"resume: {len(fin)} groups already in {man.name}, skipped", flush=True)
        jobs = [(gr, ms) for gr, ms in jobs if gr["group_id"] not in fin]
    nfail = streak = 0
    stop = False
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_safe, gr, ms, a, bd): gr["group_id"] for gr, ms in jobs}
        for fu in cf.as_completed(futs):
            if fu.cancelled():
                continue
            gid = futs[fu]
            rows, err = fu.result()
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("clip", keep="last")
            done.to_csv(man, index=False)
            nfail += err is not None; streak = streak + 1 if err is not None else 0
            if err is not None:
                print(gid, "FAILED twice:", err, flush=True)
            else:
                print(gid, "qc", rows[0]["qc"], rows[0]["qc_why"], f"h={rows[0]['height_ratio']}",
                      f"yolo={rows[0]['yolo_height_ratio']}", flush=True)
            if streak >= MAX_FAIL_STREAK and not stop:
                stop = True
                print(f"STOP: {streak} groups failed in a row; {sum(f.cancel() for f in futs)} queued groups cancelled "
                      f"(resume with --reuse)", flush=True)
    if nfail or stop:
        print(f"{nfail} groups failed{' and the batch stopped early' if stop else ''}: exit code 1", flush=True)
    return int(bool(nfail or stop))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--families", nargs="+", default=["place_leave", "unload_vehicle_leave"],
                    help="default both: the batch default mix, --car-share in car mode (D38)")
    ap.add_argument("--keep-renders", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="resume: keep the scenarios, renders and composites in DIR and skip the groups its manifest "
                         "holds in full")
    ap.add_argument("--force", action="store_true", help="with --reuse: redo finished groups (re-composite renders)")
    ap.add_argument("--data-dir", required=True,
                    help="batch dir: scenarios, renders, clean, clips, labels, manifest (a new one for every batch)")
    ap.add_argument("--plan-only", action="store_true", help="write the screened scenarios, render nothing")
    ap.add_argument("--plates", nargs="+", help="sample only these plate ids (pilots)")
    ap.add_argument("--car-share", type=float, default=0.6, help="share of car-mode groups when both families are given")
    ap.add_argument("--night-share", type=float, default=S.NIGHT_SHARE,
                    help="night groups per mode whose night plates span >= 2 scenes (on foot); 0.25 = 10%% of a 60/40 batch")
    ap.add_argument("--light-fallback", action="store_true",
                    help="also sample plates without a plate_light.csv row, with the old random sun (D40: fail-closed)")
    ap.add_argument("--scene-cap", type=float, default=S.SCENE_CAP,
                    help="a scene stays <= this x the mean count over its mode's scenes in use (0 = off)")
    a = ap.parse_args()
    if (pathlib.Path(a.data_dir) / "synth_manifest.csv").exists() and not a.reuse:  # e.g. synth/data (s15-s17)
        ap.error(f"{a.data_dir} holds a rendered batch (synth_manifest.csv): use a new --data-dir, or --reuse to resume")
    sys.exit(main(a))
