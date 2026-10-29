"""Sun light of the plates from their real cast shadows (step D2, DECISIONS.md D40): reference sheets, annotations ->
world sun azimuth / elevation per clip, check sheets.

python synth/scripts/plate_light.py sheets [CLIP ...] [--list OUT.json] [--workers 8]  -> synth/data/light/refs/<clip>.jpg
python synth/scripts/plate_light.py convert [--ann DIR] [--out CSV]   synth/data/light/ann/*.json -> synth/data/plate_light.csv
python synth/scripts/plate_light.py check [CLIP ...]                  -> synth/data/light/check/<clip>.jpg (sun rows)

Why: until D40 the generator drew the sun per group (on with p 0.6 on day plates, azimuth uniform 0-360 deg,
elevation 25-65 deg), so synthetic shadows ignored the plate's own light. The GPU pilot screen saw two groups on one plate
cast shadows in opposite directions, and a long hard shadow on a plate that looks overcast. Now
scenario.group_light reads each plate's light from synth/data/plate_light.csv, written by this tool from annotations of
the real shadows. A plate without a usable row is not sampled (fail-closed); run_batch --light-fallback keeps the old
law for such plates.

ANNOTATION PROTOCOL (one annotation per source clip)
Clip = a contest video id (vidNNNN: its _p and _x plates) or a MEVA clip = the video stem
(2018-03-11_13-55-04_14-00-04_school_G300_r13: its _p plates). The plates are those scenario.Tables can sample
(structure masks, plate_vet.csv) before the light filter. All plates of a clip share one static camera and one time of
day (convert checks the cameras), so one annotation covers them.
1. Reference sheet synth/data/light/refs/<clip>.jpg (sheets). A header, then for each plate the middle frame of its
   window at 2x render size, then the clip's first and last frames side by side (1x). Every panel has the normalised
   grid (lines every 0.05, labels every 0.1; x = column / width, y = row / height of the full frame, black bars
   included) and the calibrated walkable ground (yellow outline). Coordinates are the same in every panel (static
   camera). clips.json (--list) gives each panel's pixel box ("panels") for zooming in.
2. Condition of the clip:
   - "sun": hard cast shadows with a clear edge on the ground (poles, sign posts, tree trunks, building corners, parked
     cars, people). Annotate shadows (3.).
   - "overcast": diffuse light; no hard cast shadows on the ground (soft dark patches right under cars do not count),
     or shadows too faint to read a direction. With no measurable shadow, choose overcast, not sun.
   - "night": the clip is a night clip (contest label night, MEVA 23:55), whatever its lamps do (the generator draws
     its own lamp). convert refuses "night" on a day plate and any other condition on a night plate.
   If the light changes inside the clip (clouds), give the clip's majority condition and list the other plates in
   "plate_condition": {"<plate_id>": "sun" | "overcast" | "skip"} ("skip": the plate gets no row, it is not sampled).
3. Shadows (condition sun; at least 1, better 2-4 of different objects spread over the image):
   - "base": the ground contact of a VERTICAL object (pole foot, foot of a building's vertical corner, sign post, tree
     trunk, a standing person's feet). "tip": the end of the shadow of that object's top (or of any point straight
     above the base) on the ground. Both points on flat ground at the level of the walkable area (yellow): never on a
     wall, a raised kerb, a car or a roof. The shadow runs from base to tip, away from the sun.
   - "height_m" (optional): the height above the base of the point whose shadow is the tip, when known: a standing
     adult 1.7, a car roof edge 1.5 (base = the ground point under that edge), a door top 2.1. With it the elevation
     is measured too. Without it only the azimuth: the tip may be any point on the shadow line of the vertical edge
     (the farther from the base, the more precise).
   - Prefer long shadows (>= 1 m on the ground, >= 0.03 of the image) away from the image border (wide-angle and
     fisheye views bend lines there). Read the points to 0.005 (zoom into a panel).
   - A building's shadow: only a VERTICAL corner edge gives a direction (base = the corner's ground contact, tip = the
     matching corner of the shadow on the ground); the shadow edge of a roof or a wall top is not usable as a line.
     A person or object standing inside another shadow casts none of its own: use objects in the sun.
   - "frame": the panel the points were read from (a plate id, "start" or "end"); "note": what the object is.
4. Write synth/data/light/ann/<clip>.json (one file per clip):
   {"clip": "<clip>", "condition": "sun",
    "shadows": [{"base": [0.412, 0.733], "tip": [0.470, 0.805], "height_m": 1.7, "frame": "start", "note": "person"},
                {"base": [0.120, 0.640], "tip": [0.180, 0.700], "frame": "<plate id>", "note": "lamp post"}],
    "plate_condition": {}, "author": "<name>", "reviewed_by": null, "note": "<what the light looks like>"}
   Overcast and night: "shadows": [].
5. convert, per clip, with the calibrated camera of its plates (ground plane z = 0): azimuth = the direction from tip
   to base on the ground (towards the sun; degrees counter-clockwise from world +X, build_scene's sun convention);
   elevation = atan(height_m / ground length) over the shadows with a height (mean), else unknown; several shadows ->
   circular mean of the azimuths, spread = the largest deviation from it. Refused (no row, exit code 1): no author,
   unknown clip or condition, shadows on a clip without sun plates or none on one with, a point outside 0..1 or above
   the horizon, a ground length < MIN_LEN 0.2 m, an elevation outside 5-85 deg, a spread > SPREAD_MAX 20 deg, a
   condition that contradicts a plate's night / dim label (scenario.light_conflict), plates with different cameras.
6. check: synth/data/light/check/<clip>.jpg for every sun row, on the first plate's middle frame (2x): the annotated
   shadows (red, dot at the base) and the shadows the generator will cast with the converted sun (yellow, from 1.7 m
   white sticks at the calibration's real feet; drawn at 45 deg when the elevation is unknown). A second check
   compares the yellow shadows with the real ones, corrects the annotation file if needed, sets "reviewed_by" and
   runs convert again.
plate_light.csv (tracked; one row per clip and condition): clip, plates (space separated), condition, sun_az_deg,
sun_el_deg (empty = unknown), n_shadows, spread_deg, author, reviewed_by, note, cam (scenario.cam_sha of the camera it
was converted with: Tables ignores a row whose plate was recalibrated since), shadows (the annotated shadows, JSON).
Generator (scenario.group_light, one light per group, both members): sun iff condition sun; azimuth + U(-8, 8) deg;
elevation + U(-5, 5) deg, or U(25, 65) when unknown; overcast and night: no sun (night lamps as before)."""
import argparse, concurrent.futures as cf, json, math, pathlib, sys
import cv2, numpy as np, pandas as pd

SYN = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SYN / "gen"), str(SYN / "scripts")]
import scenario as S
from camera import Camera, plate_calib
from plate_structures import feet_px, grid, hstack, strip, text, vstack, walkable_line, write

LDIR = SYN / "data/light"
COLS = ["clip", "plates", "condition", "sun_az_deg", "sun_el_deg", "n_shadows", "spread_deg", "author", "reviewed_by",
        "note", "cam", "shadows"]
MIN_LEN, SPREAD_MAX, EL_RANGE = 0.2, 20.0, (5.0, 85.0)
STICK = 1.7  # m: the check sheet's sticks


def clip_plates(t=None):
    """The plates scenario.Tables can sample, before the light filter, with their clip, night and dim flags."""
    t = t or S.Tables(light_fallback=True)
    p = t.plates.copy()
    p["clip"] = p.video
    p["dim"] = [S.plate_dim(t.clip_meta, r) for r in p.itertuples()]
    return p


# ---------------------------------------------------------------- annotation -> sun
def ground(cal, xy):
    """Normalised image points [N,2] -> ground points [N,2] through the calib camera (nan above the horizon)."""
    cam = Camera(cal)
    return cam.ground(np.asarray(xy, float).reshape(-1, 2) * [cam.W, cam.H])


def shadow_sun(cal, s):
    """One shadow -> (azimuth towards the sun in deg, elevation in deg or None, ground length in m)."""
    b, tip = ground(cal, [s["base"], s["tip"]])
    v = b - tip
    L = float(np.hypot(*v))
    h = s.get("height_m")
    return math.degrees(math.atan2(v[1], v[0])) % 360.0, (math.degrees(math.atan2(float(h), L)) if h else None), L


def shadow_why(cal, s):
    """Why an annotated shadow cannot be used ('' = it can)."""
    pts = [s.get("base"), s.get("tip")]
    try:
        ok = all(len(p) == 2 and all(0.0 <= float(x) <= 1.0 for x in p) for p in pts)
    except (TypeError, ValueError):
        ok = False
    if not ok:
        return "base and tip must be [x, y] in 0..1"
    if s.get("height_m") is not None and not float(s["height_m"]) > 0:
        return "height_m must be > 0"
    if not np.isfinite(ground(cal, pts)).all():
        return "a point lies above the horizon"
    _, el, L = shadow_sun(cal, s)
    if L < MIN_LEN:
        return f"ground length {L:.2f} m < {MIN_LEN} m"
    if el is not None and not EL_RANGE[0] <= el <= EL_RANGE[1]:
        return f"elevation {el:.1f} deg outside {EL_RANGE}"
    return ""


def circ_mean(az):
    """Circular mean of azimuths (deg) and the largest deviation from it (deg)."""
    a = np.radians(np.asarray(az, float))
    m = math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean())) % 360.0
    return m, float(np.abs((np.asarray(az, float) - m + 180.0) % 360.0 - 180.0).max())


def convert_one(ann, plates, calib_of=plate_calib):
    """One clip's annotation -> (rows of plate_light.csv, why it was refused or ''). plates: clip_plates() rows (clip,
    plate_id, night, dim); calib_of(plate_id) -> calib dict."""
    clip, cond, sh = ann.get("clip"), ann.get("condition"), ann.get("shadows") or []
    P = plates[plates["clip"] == clip]
    if P.empty:
        return [], f"unknown clip {clip!r} (no plate Tables can sample)"
    if not ann.get("author"):
        return [], "no author"
    if cond not in S.LIGHT_CONDITIONS:
        return [], f"bad condition {cond!r}"
    pc = ann.get("plate_condition") or {}
    if set(pc) - set(P.plate_id) or set(pc.values()) - {"sun", "overcast", "skip"}:
        return [], f"bad plate_condition {pc} (plates of the clip; sun, overcast or skip)"
    eff = {p: pc.get(p, cond) for p in P.plate_id}
    sunny = "sun" in eff.values()
    if sunny != bool(sh):
        return [], "a clip with sun plates needs shadows" if sunny else "shadows given, but no plate is sun"
    cals = [calib_of(p) for p in P.plate_id]
    if len({S.cam_sha(c) for c in cals}) > 1:
        return [], "the clip's plates have different cameras (annotate them separately)"
    cal, az, el, spread = cals[0], None, None, None
    if sh:
        for i, s in enumerate(sh):
            w = shadow_why(cal, s)
            if w:
                return [], f"shadow {i} ({s.get('note', '')}): {w}"
        res = [shadow_sun(cal, s) for s in sh]
        az, spread = circ_mean([r[0] for r in res])
        if spread > SPREAD_MAX:
            return [], f"azimuth spread {spread:.1f} deg > {SPREAD_MAX} ({', '.join(f'{r[0]:.0f}' for r in res)})"
        els = [r[1] for r in res if r[1] is not None]
        el = float(np.mean(els)) if els else None
    for r in P.itertuples():
        w = "" if eff[r.plate_id] == "skip" else S.light_conflict(eff[r.plate_id], az, r.night, r.dim)
        if w:
            return [], f"{r.plate_id}: {w}"
    rows = []
    for c in S.LIGHT_CONDITIONS:
        ps = [p for p in P.plate_id if eff[p] == c]
        if ps:
            s_ = c == "sun"
            rows.append(dict(clip=clip, plates=" ".join(ps), condition=c, sun_az_deg=round(az, 2) if s_ else "",
                             sun_el_deg=round(el, 2) if s_ and el is not None else "", n_shadows=len(sh) if s_ else 0,
                             spread_deg=round(spread, 2) if s_ else "", author=ann["author"],
                             reviewed_by=ann.get("reviewed_by") or "", note=ann.get("note") or "", cam=S.cam_sha(cal),
                             shadows=json.dumps(sh) if s_ else "[]"))
    return rows, ("" if rows else "every plate skipped")


def cmd_convert(a):
    plates = clip_plates()
    rows, bad, seen = [], {}, set()
    for f in sorted(pathlib.Path(a.ann).glob("*.json")):
        try:
            ann = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            bad[f.name] = f"not JSON: {e}"; continue
        if ann.get("clip") != f.stem:
            bad[f.name] = f"clip {ann.get('clip')!r} != file name"; continue
        r, why = convert_one(ann, plates)
        seen.add(f.stem)
        if why:
            bad[f.name] = why
        rows += r
    df = pd.DataFrame(rows, columns=COLS)
    df.to_csv(a.out, index=False)
    for r in df.itertuples():
        print(f"{r.clip:48s} {r.condition:8s} az {r.sun_az_deg!s:>7s} el {r.sun_el_deg!s:>6s} n {r.n_shadows} spread "
              f"{r.spread_deg!s:>5s} plates {len(r.plates.split())} reviewed_by {r.reviewed_by or '-'}")
    todo = sorted(set(plates["clip"]) - seen)
    print(f"{len(df)} rows ({df.plates.str.split().str.len().sum() if len(df) else 0} plates) -> {a.out}; "
          f"refused {len(bad)}; clips without annotation {len(todo)} of {plates["clip"].nunique()}")
    for k, w in bad.items():
        print(f"  REFUSED {k}: {w}")
    return 1 if bad else 0


# ---------------------------------------------------------------- sheets
def frame(r, t):
    from contact_sheet import frames_at
    return frames_at(r.path, [max(0.0, t)], max(r.w, r.h))[0][1]


def panel(img, cal, s, W, H):
    img = cv2.resize(img, (W * s, H * s), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    walkable_line(img, cal, Camera(cal, cal["render_size"]), s)
    return grid(img)


def ref_sheet(job):
    """Reference sheet of one clip -> its clips.json entry (clip, plates, ref_image, panels)."""
    clip, P = job
    from contact_sheet import duration
    r0, cal = P.iloc[0], plate_calib(P.iloc[0].plate_id)
    W, H = cal["render_size"]
    T = duration(r0.path)
    blocks, panels = [strip(2 * W, [
        f"{clip}  {r0.source} {r0.scene_id}  {'night' if P.night.any() else 'day'}{'  dim' if P.dim.any() else ''}  "
        f"render {W}x{H}  clip {T:.1f} s  plates {len(P)}",
        "Light annotation (synth/scripts/plate_light.py): normalised coords (grid every 0.05, labels every 0.1), same in "
        "every panel. base = ground contact of a vertical object, tip = end of its shadow on the ground. Yellow = "
        "calibrated walkable ground."])], []  # panels: (block index, name, x0, scale)
    for r in P.itertuples():
        mid = (r.start + r.end) / 2
        blocks.append(strip(2 * W, [f"{r.plate_id}  window {r.start:.1f}-{r.end:.1f} s  middle frame t = {mid:.1f} s"]))
        blocks.append(panel(frame(r, mid), cal, 2, W, H)); panels.append((len(blocks) - 1, r.plate_id, 0, 2))
    te = max(0.0, T - 0.25)
    blocks.append(strip(2 * W + 8, [f"clip start t = 0.0 s (left) | clip end t = {te:.1f} s (right), 1x"]))
    blocks.append(hstack([panel(frame(r0, 0.0), cal, 1, W, H), panel(frame(r0, te), cal, 1, W, H)]))  # 8 px gap
    panels += [(len(blocks) - 1, "start", 0, 1), (len(blocks) - 1, "end", W + 8, 1)]
    ys = np.cumsum([0] + [b.shape[0] for b in blocks])
    out = LDIR / f"refs/{clip}.jpg"
    write(out, vstack(blocks))
    print(f"{clip}: {len(P)} plates -> {out}", flush=True)
    return dict(clip=clip, plates=list(P.plate_id), ref_image=str(out),
                panels=[dict(name=n, box=[x0, int(ys[i]), x0 + W * s, int(ys[i]) + H * s]) for i, n, x0, s in panels])


def cmd_sheets(a):
    p = clip_plates()
    clips = a.clips or sorted(p["clip"].unique())
    assert set(clips) <= set(p["clip"]), set(clips) - set(p["clip"])
    with cf.ProcessPoolExecutor(a.workers) as ex:
        out = list(ex.map(ref_sheet, [(c, p[p["clip"] == c].reset_index(drop=True)) for c in clips]))
    if a.list:
        pathlib.Path(a.list).write_text(json.dumps(out, indent=1))
        print(f"{len(out)} clips -> {a.list}")
    return 0


def cmd_check(a):
    df = pd.read_csv(a.csv, dtype=str, keep_default_na=False)
    df = df[(df.condition == "sun") & (df["clip"].isin(a.clips) if a.clips else True)]
    plates = clip_plates().set_index("plate_id")
    for r in df.itertuples():
        pid = r.plates.split()[0]
        pr, cal = plates.loc[pid], plate_calib(pid)
        W, H = cal["render_size"]
        img = cv2.resize(frame(pr, (pr.start + pr.end) / 2), (2 * W, 2 * H), interpolation=cv2.INTER_CUBIC)
        cam = Camera(cal, cal["render_size"])
        az, el = math.radians(float(r.sun_az_deg)), math.radians(float(r.sun_el_deg or 45.0))
        uv = feet_px(cal)
        G = cam.ground(uv[np.unique(np.linspace(0, len(uv) - 1, min(12, len(uv))).astype(int))]) if len(uv) else []
        for g in G:  # the generator's shadow of a STICK m pole at a real person's feet
            if not np.isfinite(g).all():
                continue
            tip = g - STICK / math.tan(el) * np.array([math.cos(az), math.sin(az)])
            q, z = cam.project([[*g, 0.0], [*g, STICK], [*tip, 0.0]])
            if (z > 0.3).all():
                q = np.round(q * 2).astype(int)
                cv2.line(img, tuple(q[0]), tuple(q[1]), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(img, tuple(q[0]), tuple(q[2]), (255, 230, 0), 3, cv2.LINE_AA)
        for k, s in enumerate(json.loads(r.shadows)):
            b, t = (np.array(s["base"]) * [2 * W, 2 * H]).astype(int), (np.array(s["tip"]) * [2 * W, 2 * H]).astype(int)
            cv2.line(img, tuple(b), tuple(t), (255, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(img, tuple(b), 5, (255, 0, 0), -1, cv2.LINE_AA)
            text(img, str(k), (int(t[0]) + 4, int(t[1]) - 4), (255, 80, 80), 0.6)
        head = [f"{r.clip}  {pid}  sun az {float(r.sun_az_deg):.1f} deg  el "
                f"{r.sun_el_deg or 'unknown (drawn at 45)'}  shadows {r.n_shadows}  spread {r.spread_deg} deg  "
                f"author {r.author}  reviewed_by {r.reviewed_by or '-'}",
                "red: annotated shadows (dot = base, number = index)   white / yellow: 1.7 m sticks at real feet and the "
                "shadows the generator will cast (+-8 deg per group)"]
        out = LDIR / f"check/{r.clip}.jpg"
        write(out, vstack([strip(2 * W, head), img]))
        print(f"{r.clip} -> {out}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheets"); s.add_argument("clips", nargs="*"); s.add_argument("--list")
    s.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("convert"); s.add_argument("--ann", default=str(LDIR / "ann"))
    s.add_argument("--out", default=str(S.LIGHT_CSV))
    s = sub.add_parser("check"); s.add_argument("clips", nargs="*"); s.add_argument("--csv", default=str(S.LIGHT_CSV))
    a = ap.parse_args()
    sys.exit({"sheets": cmd_sheets, "convert": cmd_convert, "check": cmd_check}[a.cmd](a))
