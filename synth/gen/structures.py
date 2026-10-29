"""Plate structure masks (F4 and F6 part B, DECISIONS.md D34): the one implementation of the structure rules, used by
the sampler (scenario.py), the pre-render screen (postqc.check, strict) and the compositor (no synthetic shadows on
structures); synth/scripts/plate_structures.py draws, rasterises and checks the masks with the same functions.
Masks are drawn per camera view (synth/calib/structures/<view>.json; protocol in plate_structures.py) and rasterised
per plate at its render size (plate_structures.py raster): synth/data/structures/<plate_id>.png labels 0 none, 1 S
standing structure, 2 F front layer, 3 N no-go ground; <plate_id>_contact.png the ground contact row + 1 of S pixels
(vertical-run bottom, 'base' overrides for overhangs; kept under N), 65535 on F.
Rules (render pixels inside the camera image cam.box only; MARGIN x H as composite.occluded for parked vehicles):
- ground point (feet, object footprint): on F or N, or on an S pixel whose contact row is lower than the point by more
  than MARGIN x H (the point is on or behind the structure) -> hidden (Struct.hid; walked lines are tested densified
  to 1 px, so a thin strip cannot fall between two samples)
- a walker's screen box (BODY_W m wide, as tall as the walker, at its ground point; also when the feet are below the
  frame): any F pixel, or an S pixel whose contact row is lower than the walker's foot row by more than MARGIN x H
  (the structure stands in front of the walker and would have to hide it) -> body conflict. The box follows the
  projected upright: BANDS row bands, each centred on the foot-to-head line at its height (on wide down-tilted views
  a person leans outward by up to 2 box half-widths; C5 review G3, D38)
- the carried object (C5 review G1/S4, D38): the box that holds the object over the hand's swing (carry_box: Blender-
  measured envelope in the walker's frame, turned with the walker), projected, by the body rule with the walker's foot
  row as depth; only while the walker holds it
- the crouch at the spot (C5 review G2, D38): the left foot steps CROUCH_TOE forward (feet rule on X -> toe) and the
  crouched body (CROUCH_W x CROUCH_H) stands over the knee point CROUCH_KNEE ahead of X (body rule)
- the resting object: its rest box (scenario.rest_box) footprint by the ground rule, its projected 3D box by the body
  rule with the rest centre's row as depth
Sampling is fail-closed (usable): a plate is drawn only if its view JSON lists it with verdict ok or trim and an
author other than 'prototype', a post-event plate has its plate_adds entry, both rasters exist and equal the JSON
rasterised now, and synth/data/plate_vet.csv (key = plate id or view, verdict ok / trim / drop, reason, source, date)
has no drop row for the plate or its view."""
import hashlib, json, pathlib
import numpy as np

SYN = pathlib.Path(__file__).resolve().parents[1]
ROOT = SYN.parent
SDIR, ODIR, VET = SYN / "calib/structures", SYN / "data/structures", SYN / "data/plate_vet.csv"
LAYERS = {"S": 1, "F": 2, "N": 3}
FRONT = 65535
MARGIN = 0.01  # fraction of H, as composite.occluded
BODY_W = 0.6  # m: a walker's screen box (body, swinging arms; a carried object is tested on its own: carry_box)
BANDS = 6  # row bands of the screen box along the projected upright (G3)
OBJ_GRID = 5  # footprint points per rest-box side
# carried object (build_scene.bake_object: bottom centre under the right palm + 0.06 m outward, top at the palm):
# where its ground point lies in the walker's frame (forward, left) and where its top is, per unit actor scale;
# Blender-measured envelope (leg_overlap.py 'carry') over 160 members of both genders and all outcomes (crouch frames
# excluded: X is tested by the rest box and the crouch), rounded outwards. Walking, and the first CARRY_SETTLE s after
# the root stops (the arm swings, then settles): forward -0.45 .. 0.51, left -0.40 .. -0.14 (the right side), top
# 0.735 .. 1.014. Standing still longer (idle, in-place turns): forward -0.02 .. 0.10, left -0.35 .. -0.29, top 0.79 ..
# 0.82 (tracks.moving tells the two apart)
CARRY_F, CARRY_L, CARRY_TOP = (-0.46, 0.52), (-0.41, -0.13), (0.73, 1.02)
CARRY_STAND = ((-0.03, 0.12), (-0.36, -0.28), (0.78, 0.83))
CARRY_SETTLE = 0.35  # s
# the crouch at the spot (Blender probe of D35, male crouch clip on both genders, scale 1): left toe at the bottom
# 0.518-0.549 m forward, 0.138 m left; knees 0.511-0.546 m forward, 0.43-0.45 m apart; pelvis 0.42-0.44 m up
CROUCH_TOE, CROUCH_KNEE, CROUCH_W, CROUCH_H = (0.55, 0.14), 0.55, 0.6, 1.1
_CACHE, _AVATARS = {}, None


# ---------------------------------------------------------------- rasterising (shared with plate_structures.py)
def poly_mask(pts, W, H):
    import cv2
    p = np.asarray(pts, float)
    if p.ndim != 2 or p.shape[0] < 3 or p.shape[1] != 2:
        raise ValueError(f"polygon needs >= 3 [x, y] points: {pts}")
    if (p < -0.1).any() or (p > 1.1).any():
        raise ValueError(f"polygon not in normalised coordinates: {pts}")
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(p * [W, H]).astype(np.int32)], 1)
    return m.astype(bool)


def run_bottom(mask):
    """Per True pixel: the row where its vertical run of True pixels ends (its ground contact); -1 elsewhere."""
    H, W = mask.shape
    rb = np.full((H, W), -1, np.int32)
    rb[H - 1] = np.where(mask[H - 1], H - 1, -1)
    for y in range(H - 2, -1, -1):
        rb[y] = np.where(mask[y], np.where(mask[y + 1], rb[y + 1], y), -1)
    return rb


def plate_polys(js, pid):
    return list(js.get("polygons", [])) + list((js.get("plate_adds") or {}).get(pid, []))


def rasterize(js, pid, W, H):
    """View JSON -> (labels uint8 [H,W]: 0 none, 1 S, 2 F, 3 N; contact uint16 [H,W], see the module docstring)."""
    polys = plate_polys(js, pid)
    m = {k: np.zeros((H, W), bool) for k in LAYERS}
    for p in polys:
        if p.get("layer") not in LAYERS:
            raise ValueError(f"unknown layer {p.get('layer')!r} ({p.get('note')})")
        m[p["layer"]] |= poly_mask(p["pts"], W, H)
    lab = np.zeros((H, W), np.uint8)
    lab[m["S"]] = 1; lab[m["N"]] = 3; lab[m["F"]] = 2
    rb = run_bottom(m["S"])
    every = list(js.get("polygons", [])) + [p for v in (js.get("plate_adds") or {}).values() for p in v]
    for note, line in (js.get("base") or {}).items():
        named = [p for p in every if p.get("note") == note]
        if not named or any(p["layer"] != "S" for p in named):
            raise ValueError(f"base {note!r} must name S polygons of the view")
        hit = [p for p in polys if p.get("note") == note]
        if not hit:
            continue  # a polygon of another plate's plate_adds
        b = np.asarray(line, float)
        if b.ndim != 2 or b.shape[1] != 2 or len(b) < 2:
            raise ValueError(f"base {note!r} needs >= 2 [x, y] points")
        b = b[np.argsort(b[:, 0])] * [W, H]
        by = np.clip(np.round(np.interp(np.arange(W), b[:, 0], b[:, 1])), 0, H - 1).astype(np.int32)
        pm = np.zeros((H, W), bool)
        for p in hit:
            pm |= poly_mask(p["pts"], W, H)
        rb = np.where(pm, np.maximum(rb, by[None, :]), rb)
    contact = np.where(rb >= 0, rb + 1, 0).astype(np.uint16)
    contact[m["F"]] = FRONT
    return lab, contact


def hidden(lab, contact, uv):
    """Ground pixels [N,2] on F or N, or on/behind an S structure (contact row below the point by > MARGIN x H);
    out of the frame -> False."""
    H, W = lab.shape
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    inf = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    out = np.zeros(len(uv), bool)
    l, c = lab[v[inf], u[inf]], contact[v[inf], u[inf]].astype(np.int64)
    out[inf] = (l == 2) | (l == 3) | ((c > 0) & (c - 1 > v[inf] + MARGIN * H))
    return out


def sha(lab, contact):
    return hashlib.sha1(np.ascontiguousarray(lab).tobytes() + np.ascontiguousarray(contact).tobytes()).hexdigest()[:12]


# ---------------------------------------------------------------- views, vetting, loading
def view_of(source, scene_id, video):
    """A plate's view: the MEVA camera (scene id) or the contest video (its _p and _x plates share one view)."""
    return scene_id if source == "meva" else video


def load_view(view, sdir=SDIR):
    f = pathlib.Path(sdir) / f"{view}.json"
    return json.loads(f.read_text()) if f.exists() else None


def onset(video):
    """Dumping onset g (s) of a positive contest clip from its label JSON; None for negatives and MEVA."""
    for split in ("train", "public_test"):
        f = ROOT / f"data/{split}/labels/{video}.json"
        if f.exists():
            lab = json.loads(f.read_text())
            return float(lab["timestamp"]) if lab["label"] == "true" else None
    return None


def post_event(source, video, start):
    """(is post-event, g): the plate window starts after its source clip's onset (x-plates included)."""
    g = onset(video) if source == "contest" else None
    return (g is not None and start > g), g


def vet_drops(vet=VET):
    """Keys (plate ids or views) with a drop row in plate_vet.csv; None if the table is missing (fail-closed)."""
    vet = pathlib.Path(vet)
    if not vet.exists():
        return None
    import pandas as pd
    t = pd.read_csv(vet, dtype=str)
    return set(t.key[t.verdict.str.strip() == "drop"].str.strip())


def read_raster(pid, odir=ODIR):
    import cv2
    f, fc = pathlib.Path(odir) / f"{pid}.png", pathlib.Path(odir) / f"{pid}_contact.png"
    if not (f.exists() and fc.exists()):
        return None
    return cv2.imread(str(f), cv2.IMREAD_UNCHANGED), cv2.imread(str(fc), cv2.IMREAD_UNCHANGED)


def usable(plates, render_size, sdir=SDIR, odir=ODIR, vet=VET):
    """Fail-closed sampling filter (module docstring). plates: rows with plate_id, video, scene_id, source, start;
    render_size(plate_id) -> (W, H). Returns one reason per row ('' = may be sampled)."""
    drops = vet_drops(vet)
    views, out = {}, []
    for r in plates.itertuples():
        v = view_of(r.source, r.scene_id, r.video)
        if v not in views:
            views[v] = load_view(v, sdir)
        js = views[v]
        pe, _ = post_event(r.source, r.video, r.start)
        if drops is None:
            why = "no plate_vet.csv"
        elif r.plate_id in drops or v in drops:
            why = "plate_vet drop"
        elif js is None:
            why = "no view JSON"
        elif r.plate_id not in js.get("plates", []):
            why = "not in its view JSON"
        elif js.get("verdict") not in ("ok", "trim"):
            why = f"view verdict {js.get('verdict')}"
        elif js.get("author") in (None, "", "prototype"):
            why = "view not verified (author prototype)"
        elif pe and r.plate_id not in (js.get("plate_adds") or {}):
            why = "post-event plate without plate_adds"
        else:
            ras = read_raster(r.plate_id, odir)
            if ras is None or ras[0] is None or ras[1] is None:
                why = "no raster"
            else:
                W, H = render_size(r.plate_id)
                try:
                    lab, contact = rasterize(js, r.plate_id, W, H)
                    why = "" if (ras[0].shape == lab.shape and (ras[0] == lab).all() and (ras[1] == contact).all()) \
                        else "stale raster (plate_structures.py raster)"
                except ValueError as e:
                    why = f"bad view JSON: {e}"
        out.append(why)
    return np.array(out, dtype=object)


def integral(m):
    return np.pad(np.asarray(m, np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))


class Struct:
    """A plate's structure rasters (render size) with integral images for screen-box tests."""

    def __init__(self, lab, contact):
        self.lab = np.asarray(lab, np.uint8)
        self.H, self.W = self.lab.shape
        c = np.asarray(contact, np.int32)
        self.meta = dict(sha=sha(self.lab, np.asarray(contact, np.uint16)))
        self.cs = np.where(c == FRONT, 0, c).astype(np.uint16)  # S contact row + 1 (also under N), 0 elsewhere
        self.iF, self.iS = integral(self.lab == 2), integral(self.cs > 0)
        rows = np.arange(self.H)[:, None]
        self.hid = (self.lab == 2) | (self.lab == 3) | ((self.cs > 0) & (self.cs.astype(np.int32) - 1 > rows + MARGIN * self.H))

    @property
    def solid(self):
        """Pixels of a standing structure or the front layer (S, also under N, or F): no synthetic shadows there."""
        return (self.cs > 0) | (self.lab == 2)


def load(plate_id, odir=ODIR):
    """The plate's Struct (cached), None if it has no raster."""
    key = (plate_id, str(odir))
    if key not in _CACHE:
        ras = read_raster(plate_id, odir)
        _CACHE[key] = None if ras is None or ras[0] is None or ras[1] is None else Struct(*ras)
    return _CACHE[key]


# ---------------------------------------------------------------- tests on ground points, walkers and the object
def _content(cam):
    """The camera image in whole render pixels (as composite.content_px): x0, y0, x1, y1 (exclusive ends)."""
    x0, y0, x1, y1 = cam.box
    return (int(np.ceil(x0 - 1e-6)), int(np.ceil(y0 - 1e-6)), int(np.floor(x1 + 1e-6)), int(np.floor(y1 + 1e-6)))


def _box_sum(I, y0, x0, y1, x1):
    return I[y1 + 1, x1 + 1] - I[y0, x1 + 1] - I[y1 + 1, x0] + I[y0, x0]


def _boxes(st, x0, y0, x1, y1, vref, ok):
    """Screen boxes [x0, x1] x [y0, y1] (inclusive, inside the image) where ok: any F pixel, or an S pixel whose
    contact - 1 > vref + MARGIN x H."""
    out = np.zeros(len(ok), bool)
    i = np.flatnonzero(ok)
    if not len(i):
        return out
    out[i] = _box_sum(st.iF, y0[i], x0[i], y1[i], x1[i]) > 0
    lim = np.asarray(vref, float) + MARGIN * st.H + 1.0
    for k in i[(_box_sum(st.iS, y0[i], x0[i], y1[i], x1[i]) > 0) & ~out[i]]:
        out[k] = st.cs[y0[k]:y1[k] + 1, x0[k]:x1[k] + 1].max() > lim[k]
    return out


def _clip(a, lo, hi):
    return np.clip(np.nan_to_num(a, nan=-1e6, posinf=1e6, neginf=-1e6), lo, hi)


def feet_conflict(cam, st, xy):
    """Ground points xy [N,2] inside the camera image that are hidden (on F or N, on or behind an S structure)."""
    xy = np.atleast_2d(np.asarray(xy, float))
    uv, z = cam.project(np.c_[xy, np.zeros(len(xy))])
    bx0, by0, bx1, by1 = _content(cam)
    ok = (z > 0.1) & np.isfinite(uv).all(1)
    uu, vv = np.round(_clip(uv[:, 0], -1, 1e6)), np.round(_clip(uv[:, 1], -1, 1e6))
    ok &= (uu >= bx0) & (uu < bx1) & (vv >= by0) & (vv < by1)
    out = np.zeros(len(xy), bool)
    out[ok] = st.hid[vv[ok].astype(int), uu[ok].astype(int)]  # = hidden(st.lab, st.cs, uv[ok]), per pixel
    return out


def densify(cam, xy, px=1.0):
    """Ordered ground points [N,2] (a walked line) with points inserted so that consecutive pixels are <= px apart
    (pairs that are not both in front of the camera are left as they are): the feet rule is per pixel, so a thin
    masked strip cannot fall between two samples."""
    xy = np.atleast_2d(np.asarray(xy, float))
    if len(xy) < 2:
        return xy
    uv, z = cam.project(np.c_[xy, np.zeros(len(xy))])
    ok = (z[:-1] > 0.1) & (z[1:] > 0.1) & np.isfinite(uv).all(1)[:-1] & np.isfinite(uv).all(1)[1:]
    n = np.where(ok, np.ceil(np.linalg.norm(np.diff(_clip(uv, -1e5, 1e5), axis=0), axis=1) / px), 1).astype(int)
    n = np.clip(n, 1, 4000)
    k = np.repeat(np.arange(len(xy) - 1), n)
    f = (np.arange(len(k)) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n, n)
    return np.r_[xy[k] + f[:, None] * (xy[k + 1] - xy[k]), xy[-1:]]


def body_conflict(cam, st, xy, height, width=BODY_W, pad=0):
    """Walkers of `height` m at ground points xy [N,2] whose screen box (width m wide, grown by pad px; also with the
    feet below the frame) meets an F pixel or stands behind an S pixel (its contact row lower than the foot row by >
    MARGIN x H). The box is BANDS row bands along the projected upright (each centred on the foot-to-head line at
    its height, G3); without lean their union is the upright box."""
    xy = np.atleast_2d(np.asarray(xy, float))
    foot, head, z = cam.person_px(xy, height)
    bx0, by0, bx1, by1 = _content(cam)
    ok = (z > 0.1) & np.isfinite(foot).all(1) & np.isfinite(head).all(1)
    fu, fv, hu, hv = (_clip(a, -1e6, 1e6) for a in (foot[:, 0], foot[:, 1], head[:, 0], head[:, 1]))
    half = 0.5 * width * np.abs(fv - hv) / height + pad
    top, n = np.floor(np.minimum(hv, fv) - pad), np.floor(fv + pad) - np.floor(np.minimum(hv, fv) - pad) + 1
    out = np.zeros(len(xy), bool)
    for b in range(BANDS):  # rows [r0, r1] of band b; the bands tile the box's rows
        r0, r1 = top + np.floor(n * b / BANDS), top + np.floor(n * (b + 1) / BANDS) - 1
        cu = fu + (hu - fu) * np.clip((fv - 0.5 * (r0 + r1)) / np.maximum(fv - hv, 1e-6), 0.0, 1.0)
        x0 = np.maximum(np.floor(cu - half), bx0).astype(np.int64); x1 = np.minimum(np.floor(cu + half), bx1 - 1).astype(np.int64)
        y0, y1 = np.maximum(r0, by0).astype(np.int64), np.minimum(r1, by1 - 1).astype(np.int64)
        k = ok & ~out & (x0 <= x1) & (y0 <= y1)
        out |= _boxes(st, np.where(k, x0, 0), np.where(k, y0, 0), np.where(k, x1, 0), np.where(k, y1, 0), fv, k)
    return out


def ground_conflict(cam, st, xy, height, width=BODY_W):
    """Walkers at ground points xy [N,2]: feet hidden or body conflict (only what is inside the camera image)."""
    return feet_conflict(cam, st, xy) | body_conflict(cam, st, xy, height, width)


PLAN_W, PLAN_PAD = BODY_W + 0.1, 2  # the planners' body box (m, px): their samples are not the screen's 0.1 s steps


def walk_conflict(cam, st, xy, height, plan=False):
    """(feet, body) of a walker along ordered ground points xy [N,2] (a track or a planned line): feet on the line
    densified to 1 px, body boxes at the given points (plan: grown by PLAN_W / PLAN_PAD, so a planned line that
    passes is not failed by the screen's own samples of it)."""
    feet = bool(feet_conflict(cam, st, densify(cam, xy)).any())
    w, pad = (PLAN_W, PLAN_PAD) if plan else (BODY_W, 0)
    return feet, bool(body_conflict(cam, st, xy, height, w, pad).any())


def _rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def object_conflict(cam, st, box, height):
    """The resting object (rest box (centre, angle, half extents), `height` m tall): a footprint point (OBJ_GRID x
    OBJ_GRID over the box) hidden, or its projected 3D box meets F or stands behind S (depth: the centre's row)."""
    c, th, half = (np.asarray(box[0], float), float(box[1]), np.asarray(box[2], float))
    g = np.linspace(-1.0, 1.0, OBJ_GRID)
    P = c + (np.stack(np.meshgrid(g, g), -1).reshape(-1, 2) * half) @ _rot(th).T
    if feet_conflict(cam, st, P).any():
        return True
    K = c + (np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) * half) @ _rot(th).T
    uv, z = cam.project(np.r_[np.c_[K, np.zeros(4)], np.c_[K, np.full(4, height)]])
    if (z <= 0.1).any() or not np.isfinite(uv).all():
        return False
    vc = cam.project(np.array([[*c, 0.0]]))[0][0, 1]
    bx0, by0, bx1, by1 = _content(cam)
    x0, x1 = max(int(np.floor(uv[:, 0].min())), bx0), min(int(np.floor(uv[:, 0].max())), bx1 - 1)
    y0, y1 = max(int(np.floor(uv[:, 1].min())), by0), min(int(np.floor(uv[:, 1].max())), by1 - 1)
    if x0 > x1 or y0 > y1:
        return False
    return bool(_boxes(st, np.array([x0]), np.array([y0]), np.array([x1]), np.array([y1]), [vc], np.array([True]))[0])


def _proj_boxes(cam, st, P, vref, pad=0):
    """Screen bounding boxes of 3D point sets P [N, K, 3] (grown by pad px, clipped to the camera image) by the body
    rule with foot rows vref [N] as depth; sets not wholly in front of the camera are skipped."""
    N, K = P.shape[:2]
    uv, z = cam.project(P.reshape(-1, 3))
    uv, z = uv.reshape(N, K, 2), z.reshape(N, K)
    ok = (z > 0.1).all(1) & np.isfinite(uv).all((1, 2))
    uv = _clip(uv, -1e6, 1e6)
    bx0, by0, bx1, by1 = _content(cam)
    x0 = np.maximum(np.floor(uv[..., 0].min(1) - pad), bx0).astype(np.int64)
    x1 = np.minimum(np.floor(uv[..., 0].max(1) + pad), bx1 - 1).astype(np.int64)
    y0 = np.maximum(np.floor(uv[..., 1].min(1) - pad), by0).astype(np.int64)
    y1 = np.minimum(np.floor(uv[..., 1].max(1) + pad), by1 - 1).astype(np.int64)
    ok &= (x0 <= x1) & (y0 <= y1)
    return _boxes(st, np.where(ok, x0, 0), np.where(ok, y0, 0), np.where(ok, x1, 0), np.where(ok, y1, 0),
                  _clip(np.asarray(vref, float), -1e6, 1e6), ok)


def carry_box(xy, heading, scale, obj, walking=True):
    """World corners [N, 8, 3] of the box that holds a carried object over the hand's swing for walkers at ground
    points xy [N,2] facing heading [N] rad (0 = +x), actor scale; obj = (half extents [2] m, height m, yaw deg relative
    to the walker: build_scene turns the carried object with the walker + the object's yaw); walking [N] (or one
    bool): the walking envelope (CARRY_F, CARRY_L, CARRY_TOP), else the standing one (CARRY_STAND)."""
    xy = np.atleast_2d(np.asarray(xy, float)); a = np.broadcast_to(np.asarray(heading, float), (len(xy),))
    w = np.broadcast_to(np.asarray(walking, bool), (len(xy),))
    (hx, hy), dz, yaw = np.asarray(obj[0], float), float(obj[1]), np.radians(obj[2])
    ex, ey = abs(np.cos(yaw)) * hx + abs(np.sin(yaw)) * hy, abs(np.sin(yaw)) * hx + abs(np.cos(yaw)) * hy
    env = np.where(w[:, None, None], np.array([CARRY_F, CARRY_L, CARRY_TOP]), np.array(CARRY_STAND)) * scale  # [N,3,2]
    F, L = env[:, 0] + [-ex, ex], env[:, 1] + [-ey, ey]
    Z = np.c_[np.maximum(0.0, env[:, 2, 0] - dz), env[:, 2, 1]]
    i, j, k = (g.ravel() for g in np.meshgrid([0, 1], [0, 1], [0, 1], indexing="ij"))
    u, n = np.c_[np.cos(a), np.sin(a)], np.c_[-np.sin(a), np.cos(a)]
    G = xy[:, None] + F[:, i][..., None] * u[:, None] + L[:, j][..., None] * n[:, None]
    return np.concatenate([G, Z[:, k][..., None]], 2)


def carry_conflict(cam, st, xy, heading, scale, obj, pad=0, walking=True):
    """Walkers at ground points xy [N,2] facing heading [N] that carry obj (carry_box; walking [N] or one bool picks
    the envelope): its projected box meets F or stands behind S, with the walker's foot row as depth (the body rule;
    C5 review G1/S4, D38)."""
    xy = np.atleast_2d(np.asarray(xy, float))
    if not len(xy):
        return np.zeros(0, bool)
    vref = cam.project(np.c_[xy, np.zeros(len(xy))])[0][:, 1]
    return _proj_boxes(cam, st, carry_box(xy, heading, scale, obj, walking), vref, pad)


def crouch_conflict(cam, st, X, heading, scale, plan=False):
    """The crouch at the spot X facing heading (C5 review G2, D38): the left foot's step to CROUCH_TOE (feet rule on the
    densified X -> toe line) or the crouched body (CROUCH_W x CROUCH_H, x scale; plan: PLAN_W-style padding) over the
    knee point CROUCH_KNEE ahead of X."""
    X = np.asarray(X, float); u = np.array([np.cos(heading), np.sin(heading)]); n = np.array([-u[1], u[0]])
    T = X + scale * (CROUCH_TOE[0] * u + CROUCH_TOE[1] * n)
    if feet_conflict(cam, st, densify(cam, np.array([X, T]))).any():
        return True
    w, pad = (CROUCH_W + 0.1, PLAN_PAD) if plan else (CROUCH_W, 0)
    return bool(body_conflict(cam, st, (X + scale * CROUCH_KNEE * u)[None], CROUCH_H * scale, w, pad)[0])


def turn_arc(a, b, step=np.radians(15)):
    """Headings of an in-place turn from a to b as scenario.plan_member keys it (the shorter way; exactly 180 deg
    clockwise)."""
    d = (b - a + np.pi) % (2 * np.pi) - np.pi
    d = -np.pi if abs(abs(d) - np.pi) < 1e-6 else d
    return a + d * np.linspace(0.0, 1.0, int(np.ceil(abs(d) / step)) + 1)


def distractor_height(D):
    """A distractor's height (m): scale x its avatar's height (avatar_table.csv; plan_distractors draws the height)."""
    global _AVATARS
    if _AVATARS is None:
        import pandas as pd
        _AVATARS = pd.read_csv(SYN / "data/avatar_table.csv").set_index("avatar").height_m
    return float(D["scale"] * _AVATARS[D["avatar"]])


def group_conflict(cam, st, tracks, heights, obj=None, carried=(), crouch=None):
    """Which parts of a group are on or behind a plate structure: tracks = [(who, ground track [T,2])] of every walker
    of every member (0.1 s steps), heights = their heights (m); obj = (rest box, height m) or None; carried = [(who,
    track [T,2], heading [T], holding [T] bool, scale, obj spec of carry_box, walking [T] bool)]: carried objects while
    held (walking: tracks.moving, the envelope); crouch =
    (X, heading, scale) of the crouch at the spot (crouch_conflict) or None. Returns the sorted reasons ('<who> feet',
    '<who> body', '<who> carried object', 'crouch', 'object'); [] = clear."""
    why = set()
    for (who, w), h in zip(tracks, heights):
        feet, body = walk_conflict(cam, st, w, h)
        if feet:
            why.add(f"{who} feet")
        if body:
            why.add(f"{who} body")
    for who, w, a, hold, s, o, mv in carried:
        hold = np.asarray(hold, bool)
        if hold.any() and carry_conflict(cam, st, np.asarray(w)[hold], np.asarray(a)[hold], s, o,
                                         walking=np.broadcast_to(np.asarray(mv, bool), hold.shape)[hold]).any():
            why.add(f"{who} carried object")
    if crouch is not None and crouch_conflict(cam, st, *crouch):
        why.add("crouch")
    if obj is not None and object_conflict(cam, st, *obj):
        why.add("object")
    return sorted(why)
