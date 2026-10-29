"""Distractor line library (F3, DECISIONS.md D36): the straight crossings passers-by may walk on each usable plate.
It is an ALLOWLIST: every line is written with status 'pending', and the sampler (scenario.plan_distractors) uses only
lines set to 'approved' after a check on the plate's overlay sheet (default deny).

python synth/scripts/distractor_lines.py [--grid 0.5 0.125] [--plates P ...] [--workers 16] [--out JSON] [--no-sheets]
python synth/scripts/distractor_lines.py set approved|rejected|pending LINE_ID [LINE_ID ...] --by NAME [--note TEXT]

Build (per plate of scenario.Tables, i.e. vetted plates with structure masks): every line of a grid of headings (GRID
degrees) and offsets (GRID m, across the walkable polygon) is a candidate, taken in an order drawn with the random
stream zlib.crc32(plate id), so a rebuild is byte-identical (random points and headings did not converge: narrow
corridors were still being found at 200000 tries, D36); a candidate is kept if
- the walker's body (2.0 m x 1.0 m: the tallest distractor with a carried object) is in view on one stretch of the line
  that ends in both directions (it enters and leaves through the image border, not on the horizon) with 100 m of line
  beyond it out of view (the 60 s walk-in / walk-out keys of tracks.walker_keys at <= 1.63 m/s);
- qc.ground_ok: feet in a black bar anywhere refused; where the feet are in the image they lie inside the walkable
  polygon (margin 0) except one entry and one exit run of <= 2 m in the border band or far; >= 3 m inside;
- not within DEDUP m (symmetric Hausdorff distance of the stretches where the feet are in view) of a line kept
  before (one corridor counts once, so MIN_LINES counts distinct corridors);
- the plate's structure masks: feet (densified to 1 px) and the padded body box of a 1.95 m walker between E and B
  (structures.walk_conflict, plan=True);
- feet off the parked-vehicle mask from 1 m before E to 1 m after B;
- moving traffic (plate_traffic.py boxes that are not a still parked vehicle) covers its feet in <= MAX_TRAFFIC of
  the plate window's traffic samples;
and at most MAX_LINES lines per plate. E and B are stored 1 m beyond the stretch where the body is in view, so the
walk-in / walk-out extensions are never in view; s_in / s_out = where the feet enter / leave the image, ins0 / ins1 =
the stretch inside the walkable polygon (arc lengths from E, m); traffic = the share above.
A rebuild keeps a line's status (and by / date / note) when the plate's camera hash and the line's id and ends are
unchanged; other reviewed lines go back to 'pending' (reported). The plate entry records the camera hash and the
structure-mask hash: the sampler ignores a plate whose entry no longer matches (scenario.approved_lines).

Overlay sheets (unless --no-sheets): synth/data/structures/lines_<plate_id>.jpg (derived, not tracked) with ALL lines
of the plate and their ids, drawn on frames at the start, middle and end of the plate window, then every line alone on
the middle frame: thick = feet inside the walkable polygon, thin = the entry / exit runs, translucent structure masks
(orange S, green F, blue N), parked vehicles (blue), walkable polygon (yellow), camera image box (red, plates with
bars).
Review: set a line 'approved' only if a person could walk it in the real scene over its whole drawn length (not
through piles, fences, hedges, walls, roofs, cars, water; not along a building face); 'rejected' otherwise."""
import argparse, collections, datetime, json, math, pathlib, sys, zlib
import concurrent.futures as cf
import cv2, numpy as np

SYN = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SYN / "gen"), str(SYN / "scripts")]
import qc, structures, scenario as S
from camera import Camera, plate_calib

OUT = S.LINES
SHEETS = SYN / "data/structures"
GRID = (0.5, 0.125)  # candidate lines: heading step (deg), offset step (m); 1.0 / 0.25 missed corridors (D36)
STEP, SPAN = 0.1, 250.0  # m between line samples; m of line each way from the polygon centre's foot point
EXT = 100.0  # m beyond the body-in-view stretch that must stay out of view (60 s at the fastest distractor, 1.63 m/s)
LINE_H, LINE_W = 2.0, 1.0  # m: the body box for E / B (tallest distractor 1.95 m, carried object)
DEDUP, MAX_LINES, MAX_TRAFFIC = 1.5, 20, 0.2
STATUSES = ("pending", "approved", "rejected")


def feet_seg(ln):
    E, B = np.array(ln["E"]), np.array(ln["B"])
    u = (B - E) / np.linalg.norm(B - E)
    return E + ln["s_in"] * u, E + ln["s_out"] * u


def corridor_dist(a, b):
    """Symmetric Hausdorff distance (m) of two lines' stretches where the feet are in view."""
    (a0, a1), (b0, b1) = feet_seg(a), feet_seg(b)
    return max(S.seg_point_dist(a0, b0, b1), S.seg_point_dist(a1, b0, b1), S.seg_point_dist(b0, a0, a1),
               S.seg_point_dist(b1, a0, a1))


def line_geometry(cam, poly, p, u):
    """The line through ground point p with direction u: its stored geometry (dict) or why it is refused (str)."""
    s = np.arange(-SPAN, SPAN + 1e-9, STEP)
    xy = p + s[:, None] * u
    iv = np.flatnonzero(S.in_view(cam, xy, LINE_H, LINE_W))
    if not len(iv):
        return "never in view"
    if len(iv) != iv[-1] - iv[0] + 1:
        return "in view twice"
    b0, b1 = s[iv[0]], s[iv[-1]]
    if b0 - 1.0 - EXT < -SPAN or b1 + 1.0 + EXT > SPAN:
        return "no exit from the view"
    why, ins, rg = qc.ground_ok(cam, poly, xy, STEP)
    if why:
        return why
    i0, i1 = rg
    k = i0 + np.flatnonzero(~qc.off_poly(poly, xy[i0:i1 + 1], qc.GROUND_MARGIN))
    a = b0 - 1.0  # arc length of E
    r = lambda v: round(float(v), 3)
    return dict(E=[r(c) for c in p + a * u], B=[r(c) for c in p + (b1 + 1.0) * u], s_in=r(s[i0] - a),
                s_out=r(s[i1] - a), ins0=r(s[k[0]] - a), ins1=r(s[k[-1]] - a), inside=r(ins))


def traffic_share(cam, pts, traffic):
    """Share of the plate window's traffic samples in which a moving vehicle box (not a still parked vehicle, IoU >=
    0.5; the box's bottom 10% allowed, as qc.traffic_conflict) contains a foot point of pts [N,2]."""
    if traffic is None or not len(traffic[0]["t"]):
        return 0.0
    tr, still = traffic
    tt, bx = tr["t"], tr["boxes"]
    moving = np.array([not any(qc._box_iou(b[1:5], s_) >= 0.5 for s_ in still) for b in bx], bool)
    uv, z = cam.project(np.c_[pts, np.zeros(len(pts))])
    ok = (z > 0) & np.isfinite(uv).all(1)
    u, v = uv[ok, 0] / cam.W, uv[ok, 1] / cam.H
    hit = set()
    for b in bx[moving] if len(bx) else []:
        x0, y0, x1, y1 = b[1:5]
        if ((u > x0) & (u < x1) & (v > y0) & (v < y1 - 0.1 * (y1 - y0))).any():
            hit.add(int(b[0]))
    return len(hit) / len(tt)


def line_screen(cam, ln, st, vlab, traffic):
    """Why a geometric line is refused on this plate (str), or None; sets ln['traffic']."""
    E, B = np.array(ln["E"]), np.array(ln["B"])
    L = float(np.linalg.norm(B - E)); u = (B - E) / L
    if st is not None and any(structures.walk_conflict(cam, st, E + np.arange(0, L + 1e-9, STEP)[:, None] * u,
                                                       S.DIS_H[1], plan=True)):
        return "structure"
    ext = E + np.arange(-1.0, L + 1.0 + 1e-9, STEP)[:, None] * u
    if vlab is not None and qc.feet_on_vehicle_mask(cam, ext, vlab) > 0:
        return "parked vehicle"
    ln["traffic"] = round(traffic_share(cam, ext, traffic), 3)
    return "moving traffic" if ln["traffic"] > MAX_TRAFFIC else None


def plate_lines(cal, st=None, vlab=None, traffic=None, seed=0, grid=GRID):
    """Lines of one plate (render-size camera): (kept lines without ids / statuses, Counter of refusals). Candidates:
    headings every grid[0] deg x offsets every grid[1] m from the walkable polygon's bounding-box centre (covering
    the polygon), in an order drawn with `seed`."""
    cam = Camera(cal, cal["render_size"]); poly = np.asarray(cal["walkable"], float)
    c = (poly.min(0) + poly.max(0)) / 2
    R = float(np.linalg.norm(poly - c, axis=1).max())
    th = np.radians(np.arange(0.0, 180.0, grid[0]))
    rho = np.arange(-R, R + 1e-9, grid[1])
    cand = np.stack(np.meshgrid(th, rho, indexing="ij"), -1).reshape(-1, 2)
    out, why = [], collections.Counter()
    for t_, r_ in cand[np.random.default_rng(seed).permutation(len(cand))]:
        u = np.array([math.cos(t_), math.sin(t_)])
        ln = line_geometry(cam, poly, c + r_ * np.array([-u[1], u[0]]), u)
        if isinstance(ln, str):
            why[ln] += 1; continue
        if any(corridor_dist(ln, o) < DEDUP for o in out):
            why["same corridor"] += 1; continue  # before the costly tests: the result is the same
        w = line_screen(cam, ln, st, vlab, traffic)
        if w:
            why[w] += 1; continue
        out.append(ln)
        if len(out) >= MAX_LINES:
            break
    return out, why


def plate_inputs(pid):
    """(calib, structures, parked-vehicle labels, (traffic, still boxes)) of a plate."""
    import postqc
    cal = plate_calib(pid)
    vf = SYN / f"data/vehicles/{pid}.png"
    vlab = cv2.imread(str(vf), cv2.IMREAD_UNCHANGED) if vf.exists() else None
    return cal, structures.load(pid), vlab, postqc.traffic_data(dict(plate_id=pid, render_size=cal["render_size"]))


def build_one(args):
    pid, grid = args
    cal, st, vlab, traffic = plate_inputs(pid)
    seed = zlib.crc32(pid.encode())
    lines, why = plate_lines(cal, st, vlab, traffic, seed, grid)
    for k, ln in enumerate(lines):
        ln.update(id=f"{pid}#{k:02d}", status="pending", by=None, date=None, note=None)
    return dict(calib=S.calib_hash(cal), structures=None if st is None else st.meta["sha"], seed=seed, grid=list(grid),
                refused=dict(sorted(why.items())), lines=[{k: ln[k] for k in LINE_KEYS} for ln in lines])


LINE_KEYS = ("id", "E", "B", "s_in", "s_out", "ins0", "ins1", "inside", "traffic", "status", "by", "date", "note")


def carry_statuses(new, old):
    """Copy status / by / date / note from the old library to unchanged lines (same plate camera hash, id and ends).
    Returns the reviewed old lines that were not carried over (they go back to 'pending' or are gone)."""
    lost = []
    for pid, e in (old or {}).get("plates", {}).items():
        if pid not in new:
            continue
        same = {ln["id"]: ln for ln in new[pid]["lines"]} if new[pid]["calib"] == e.get("calib") else {}
        for ln in e["lines"]:
            n = same.get(ln["id"])
            if n is not None and n["E"] == ln["E"] and n["B"] == ln["B"]:
                n.update({k: ln.get(k) for k in ("status", "by", "date", "note")})
            elif ln.get("status") != "pending":
                lost.append(ln["id"])
    return lost


def dump(lib):
    """The library as JSON text with one line per distractor line (readable diffs of reviews)."""
    j = lambda v: json.dumps(v, sort_keys=True)
    out = ["{"] + [f" {j(k)}: {j(v)}," for k, v in lib.items() if k != "plates"] + [' "plates": {']
    ps = sorted(lib["plates"].items())
    for i, (pid, e) in enumerate(ps):
        meta = ", ".join(f"{j(k)}: {j(v)}" for k, v in sorted(e.items()) if k != "lines")
        out.append(f"  {j(pid)}: {{{meta}, \"lines\": [")
        out += [f"   {json.dumps(ln)}" + ("," if k < len(e["lines"]) - 1 else "") for k, ln in enumerate(e["lines"])]
        out.append("  ]}" + ("," if i < len(ps) - 1 else ""))
    return "\n".join(out + [" }", "}"]) + "\n"


PALETTE = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75),
           (227, 119, 194), (188, 189, 34), (23, 190, 207), (255, 255, 255), (174, 199, 232), (255, 187, 120),
           (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148), (247, 182, 210), (219, 219, 141),
           (158, 218, 229), (255, 255, 0)]


def sheet(pid, row, e, out_dir=SHEETS):
    """Overlay sheet of every line of a plate (module docstring) -> out_dir/lines_<pid>.jpg."""
    import plate_structures as PS
    cal = plate_calib(pid); W, H = cal["render_size"]; s = 2 if W < 640 else 1
    cam = Camera(cal, (W, H)); st = structures.load(pid)
    ts = [row.start + 0.5, (row.start + row.end) / 2, row.end - 0.5]
    def base(img):
        img = cv2.resize(img, (W * s, H * s), interpolation=cv2.INTER_CUBIC)
        if st is not None:
            L = cv2.resize(st.lab, (W * s, H * s), interpolation=cv2.INTER_NEAREST)
            ov = img.copy()
            for k, c in PS.COL.items():
                ov[L == k] = c
            img = cv2.addWeighted(ov, 0.3, img, 0.7, 0)
        PS.vehicles(img, pid)
        PS.walkable_line(img, cal, cam, s)
        if cam.bars.any():
            x0, y0, x1, y1 = (np.array(cam.box) * s).round().astype(int)
            cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 0), 2)
        return img

    def draw(img, k, ln, th=3):
        E, B = np.array(ln["E"]), np.array(ln["B"]); u = (B - E) / np.linalg.norm(B - E)
        col = PALETTE[k % len(PALETTE)]
        for a, b, w in ((ln["s_in"], ln["s_out"], 1), (ln["ins0"], ln["ins1"], th * s)):
            PS.ground_line(img, cam, E + np.arange(a, b + 1e-9, 0.1)[:, None] * u, s, col, False, w)
        m = E + 0.5 * (ln["ins0"] + ln["ins1"]) * u
        (mu, mv), = cam.project(np.array([[*m, 0.0]]))[0]
        PS.text(img, ln["id"].rsplit("#", 1)[1], (int(mu * s) + 4, int(mv * s) - 4 - 10 * (k % 3)), col, 0.6)
    frames = PS.frames(row, ts, W, H)
    panels = []
    for t_, img in zip(ts, frames):
        img = base(img)
        for k, ln in enumerate(e["lines"]):
            draw(img, k, ln)
        panels.append(PS.vstack([PS.strip(img.shape[1], [f"plate time {t_:.1f} s"]), img]))
    one = []  # every line alone on the middle frame, for the line-by-line review
    for k, ln in enumerate(e["lines"]):
        img = base(frames[1]); draw(img, k, ln, 4)
        img = cv2.resize(img, (img.shape[1] * 2 // 3, img.shape[0] * 2 // 3), interpolation=cv2.INTER_AREA)
        one.append(PS.vstack([PS.strip(img.shape[1], [f"{ln['id'].rsplit('#', 1)[1]}: inside {ln['inside']:.1f} m, "
                                                       f"traffic {ln['traffic']:.0%}, {ln['status']}"]), img]))
    cols = max(3, len(panels) * panels[0].shape[1] // (one[0].shape[1] + 8)) if one else 1
    n = collections.Counter(ln["status"] for ln in e["lines"])
    hd = [f"{pid}  scene {row.scene_id}  {len(e['lines'])} lines (approved {n['approved']}, rejected {n['rejected']}, "
          f"pending {n['pending']}); the sampler uses only approved lines, and a plate needs >= {S.MIN_LINES}",
          "thick: feet inside the walkable area; thin: entry / exit run; yellow: walkable polygon; translucent: "
          "structure masks (orange S, green F, blue N); blue: parked vehicles; red box: camera image (bars outside)",
          "  ".join(f"{ln['id'].rsplit('#', 1)[1]}: {ln['inside']:.0f} m traffic {ln['traffic']:.0%} {ln['status']}"
                    for ln in e["lines"])]
    body = PS.hstack(panels)
    if one:
        grid = [PS.hstack(one[i:i + cols]) for i in range(0, len(one), cols)]
        body = PS.vstack([body, np.full((8, body.shape[1], 3), 255, np.uint8),
                          PS.strip(body.shape[1], ["each line alone (middle frame):"])] + grid)
    out = pathlib.Path(out_dir) / f"lines_{pid}.jpg"
    PS.write(out, PS.vstack([PS.strip(body.shape[1], hd), body]))
    return out


def sheet_one(args):
    return str(sheet(*args))


def cmd_build(a):
    t = S.Tables()
    plates = t.plates if not a.plates else t.plates[t.plates.plate_id.isin(a.plates)]
    assert not a.plates or len(plates) == len(set(a.plates)), set(a.plates) - set(plates.plate_id)
    out = pathlib.Path(a.out)
    old = json.loads(out.read_text()) if out.exists() else None
    with cf.ProcessPoolExecutor(a.workers) as ex:
        new = dict(zip(plates.plate_id, ex.map(build_one, [(p, tuple(a.grid)) for p in plates.plate_id])))
    lost = carry_statuses(new, old)
    keep = {} if old is None or not a.plates else {p: e for p, e in old["plates"].items() if p not in new}
    lib = dict(about="synth/scripts/distractor_lines.py (F3, DECISIONS.md D36): allowlist of distractor lines; only "
                     "status 'approved' is used", rule=dict(
                    step=STEP, ext=EXT, line_h=LINE_H, line_w=LINE_W, dedup=DEDUP, max_lines=MAX_LINES,
                    max_traffic=MAX_TRAFFIC, margin=qc.GROUND_MARGIN, max_run=qc.MAX_RUN, min_inside=qc.MIN_INSIDE,
                    band=[qc.BAND_SIDE, qc.BAND_TOP, qc.BAND_BOTTOM], far=qc.FAR_REL, body_h=S.DIS_H[1]),
               plates=dict(keep, **new))
    out.write_text(dump(lib))
    rows = plates.set_index("plate_id")
    print(f"{'plate':58s} {'scene':>12s} lines approved  refused (top 3)")
    for pid, e in sorted(new.items()):
        n = collections.Counter(ln["status"] for ln in e["lines"])
        top = ", ".join(f"{k} {v}" for k, v in collections.Counter(e["refused"]).most_common(3))
        print(f"{pid:58s} {rows.loc[pid].scene_id:>12s} {len(e['lines']):5d} {n['approved']:8d}  {top}")
    nl = {p: len(e["lines"]) for p, e in new.items()}
    sc = rows.scene_id
    for k in (1, S.MIN_LINES, 5, MAX_LINES):
        ps = [p for p, v in nl.items() if v >= k]
        print(f"plates with >= {k} lines: {len(ps)} of {len(nl)} ({sc[ps].nunique()} of {sc.nunique()} scenes)")
    print(f"{sum(nl.values())} lines; reviewed lines not carried over (back to pending or gone): {lost or 'none'}")
    print(out)
    if not a.no_sheets:
        jobs = [(p, rows.loc[p], new[p]) for p in sorted(new) if new[p]["lines"]]
        with cf.ProcessPoolExecutor(a.workers) as ex:
            for f in ex.map(sheet_one, jobs):
                print(f)


def cmd_set(a):
    out = pathlib.Path(a.out)
    lib = json.loads(out.read_text())
    for lid in a.ids:
        pid = lid.rsplit("#", 1)[0]
        ln = next((x for x in lib["plates"].get(pid, {}).get("lines", []) if x["id"] == lid), None)
        if ln is None:
            sys.exit(f"no line {lid}")
        ln.update(status=a.status, by=a.by, date=datetime.date.today().isoformat(), note=a.note)
        print(lid, a.status)
    out.write_text(dump(lib))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "set":
        ap = argparse.ArgumentParser(prog="distractor_lines.py set")
        ap.add_argument("status", choices=STATUSES)
        ap.add_argument("ids", nargs="+")
        ap.add_argument("--by", required=True)
        ap.add_argument("--note")
        ap.add_argument("--out", default=str(OUT))
        cmd_set(ap.parse_args(sys.argv[2:]))
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--grid", type=float, nargs=2, default=list(GRID), help="heading step (deg), offset step (m)")
        ap.add_argument("--plates", nargs="+")
        ap.add_argument("--workers", type=int, default=16)
        ap.add_argument("--out", default=str(OUT))
        ap.add_argument("--no-sheets", action="store_true")
        cmd_build(ap.parse_args())
