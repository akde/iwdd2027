"""Automatic rejects (group level: a group with a failing member is dropped whole, so pairs stay intact).
object_area(masks, member, scale)  object pixels at rest (90th percentile over rest frames: the actor may stand in
                                   front of it), after degradation (scale = degraded/render height)
actor_in_view(masks, member, fps)  actor visible through the disposal motion [t_event, t_release] (or t_event + 1 s)
mask_height_ratio(masks, k, cam, X, h)  actor mask height at frame k (standing at the spot) / the camera model's
                                   pixel height of an h m person at X. The calibration itself was validated
                                   against real people per plate (stage 2, within 15%); this guards the render.
height_ratio(frame, cal, foot, box) YOLO11x box overlapping the actor / real people's box height at the foot point
                                   (information only: small, dark or face-blurred avatars are often missed)
person_free(video, plate, t0, t1, cal)  no real person in the plate window (det_x; on extended plates only people in
                                   the walkable area + 2 m, det_xe: D28)
rest_in_image(cam, xy)             the object's possible rest area around the spot stays out of black bars (sampling)
motion_regions / motion_conflict   moving vehicles from the plate's motion foreground (plate_motion.py, D32)
object_apart_s(masks, fps, scale)  seconds the released object is seen apart from the actor (masks.npz obj_apart_px)
leg_phase / leg_phases / leg_verdict  legs in the resting object by motion phase (leg_overlap.py records, F1, D33;
                                   crouch phases at crouch_th of the object, D41)
poly_dist / ground_ok              the distractor ground rule of the line library (distractor_lines.py, F3, D36)
Thresholds: object >= 20 px; actor visible in >= 90% of event frames; |height ratio - 1| <= 0.25; object apart from
the actor >= 0.5 s in the positive (new batches, D32); legs in the object < 0.05 m outside the crouch (D33), while
crouching < 0.08 m in rigid objects and < 0.15 m in soft bags (D41)."""
import os, json, pathlib, sys
import numpy as np

from camera import inside

SYN = pathlib.Path(__file__).resolve().parents[1]
MIN_OBJ_PX, MAX_HEIGHT_ERR = 20.0, 0.25
_yolo = None


def object_area(masks, member, scale):
    rest = masks["obj_at_rest"] & (masks["obj_px"] > 0)
    if not rest.any():
        return 0.0 if member["outcome"] in ("leave", "pickup", "preexisting") else None
    return float(np.percentile(masks["obj_px"][rest], 90) * scale * scale)  # size, not occlusion by the actor


def actor_in_view(masks, member, fps):
    ev = member["events"]
    t0, t1 = ev["t_event"], ev.get("t_release", ev["t_event"] + 1.0)
    k = np.arange(int(np.floor(t0 * fps)), min(len(masks["act_px"]), int(np.ceil(t1 * fps)) + 1))
    return float((masks["act_px"][k] > 50).mean()) if len(k) else 0.0


def mask_height_ratio(masks, k, cam, X, height):
    b = masks["act_box"][k]
    if b[0] < 0:
        return None
    foot, head, _ = cam.person_px(np.array([X]), height)
    return float((b[3] - b[1] + 1) / max(foot[0, 1] - head[0, 1], 1.0))


def height_ratio(frame_rgb, cal, foot_uv, render_size, act_box=None):
    global _yolo
    if _yolo is None:
        os.environ.setdefault("YOLO_AUTOINSTALL", "false"); os.environ.setdefault("YOLO_OFFLINE", "true")
        from ultralytics import YOLO
        _yolo = YOLO(str(SYN / "assets/yolo11x.pt"))
    W, H = render_size; k = W / cal["image_size"][0]
    fb = cal["fit_boxes"]
    u, vb, vt = (np.array(fb[x]) * k for x in ("u", "vb", "vt"))
    A = np.c_[np.ones_like(vb), vb, u]; y = vb - vt
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(3):
        res = y - A @ coef
        keep = np.abs(res) < 2.5 * 1.4826 * np.median(np.abs(res)) + 1e-6
        coef = np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]
    fu, fv = foot_uv
    r = _yolo.predict(frame_rgb[..., ::-1].copy(), imgsz=1280, conf=0.25, classes=[0], verbose=False)[0]
    bx = r.boxes.xyxy.cpu().numpy()
    if act_box is not None and len(bx):  # only a box that overlaps the actor's mask box
        ix = np.minimum(bx[:, 2], act_box[2]) - np.maximum(bx[:, 0], act_box[0])
        iy = np.minimum(bx[:, 3], act_box[3]) - np.maximum(bx[:, 1], act_box[1])
        bx = bx[(ix > 0) & (iy > 0.5 * (act_box[3] - act_box[1]))]
    if not len(bx):
        return None
    j = np.argmin(np.abs((bx[:, 0] + bx[:, 2]) / 2 - fu) + np.abs(bx[:, 3] - fv))
    ref = coef @ [1, fv, fu]
    return float((bx[j, 3] - bx[j, 1]) / ref) if ref > 4 else None


def extended(plate_id):
    """Extended contest plate (extend_plates.py: <video>_x<k><letter>)."""
    return plate_id.split("_")[-1].startswith("x")


def person_free(video, plate_id, t0, t1, cal=None):
    """No real person in the plate window [t0 - 1, t1 + 1] s (plate time): no YOLO11x person box (conf >= 0.10) of
    synth/data/det_x (3 fps). Extended plates allow people outside the walkable area + 2 m (D28): there only the
    det_xe boxes that extend_plates.occupied counts (with the group's camera cal) break the window, as they broke the
    plate when it was made (before D32 det_x rejected 8 of the 10 s17 groups on extended plates)."""
    if cal is not None and extended(plate_id):
        if str(SYN / "scripts") not in sys.path:
            sys.path.insert(0, str(SYN / "scripts"))
        from extend_plates import occupied, PAD
        from camera import Camera
        d = np.load(SYN / f"data/det_xe/{plate_id[:-1]}.npz")
        t = d["t"][occupied(d["boxes"], Camera(cal), cal, len(d["t"]))]
        return not bool(((t >= t0 - PAD) & (t <= t1 + PAD)).any())
    d = np.load(SYN / f"data/det_x/{video}.npz")
    b = d["boxes"]; t = b[:, 0] / 3.0  # verify_plates samples at 3 fps
    return not bool(((t >= t0 - 1.0) & (t <= t1 + 1.0)).any())


def feet_on_vehicle(cam, ground_xy, vehicles, in_view_only=True):
    """Fraction of ground points (actor feet over time) whose pixel lies inside a parked vehicle box (normalised
    boxes; the bottom 10% of a box is allowed: someone standing just in front of the bumper)."""
    if not len(vehicles) or not len(ground_xy):
        return 0.0
    uv, z = cam.project(np.c_[ground_xy, np.zeros(len(ground_xy))])
    W, H = cam.W, cam.H
    vis = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    if in_view_only:
        uv = uv[vis]
    if not len(uv):
        return 0.0
    hit = np.zeros(len(uv), bool)
    for x0, y0, x1, y1 in vehicles:
        bx0, by0, bx1, by1 = x0 * W, y0 * H, x1 * W, y1 * H
        hit |= (uv[:, 0] > bx0) & (uv[:, 0] < bx1) & (uv[:, 1] > by0) & (uv[:, 1] < by1 - 0.1 * (by1 - by0))
    return float(hit.mean())


def spot_luma(cam, X, luma_grid):
    """8-bit luma of the plate around the event spot (16x16 grid cell under the actor's body centre)."""
    g = np.asarray(luma_grid)
    foot, head, _ = cam.person_px(np.array([X]), 1.0)
    u, v = foot[0, 0], (foot[0, 1] + head[0, 1]) / 2
    i = int(np.clip(v / cam.H * g.shape[0], 0, g.shape[0] - 1)); j = int(np.clip(u / cam.W * g.shape[1], 0, g.shape[1] - 1))
    return float(g[i, j])


MIN_SPOT_LUMA = 20.0


# 8-bit luma; actors below 10 were invisible on inspection (D25); 12 for new batches: the D29 screen found 2 invisible
# actors at 10.5 and 11.0, the lowest kept pair scores 12.2 (D32). postqc.py DIR keeps an old batch's 10.
MIN_VISIBILITY, MIN_VISIBILITY_OLD = 12.0, 10.0


def visibility(clean_frames, plate_frames_at_ks, act_box, ks):
    """Median over event frames ks of mean |composite - plate| luma inside the actor's box (how much the actor
    changes the image). clean_frames: decoded clean composite; plate_frames_at_ks: plate frames at those ks."""
    import cv2
    out = []
    for k, p in zip(ks, plate_frames_at_ks):
        x0, y0, x1, y1 = act_box[k]
        a = cv2.cvtColor(clean_frames[k][y0:y1 + 1, x0:x1 + 1], cv2.COLOR_RGB2GRAY).astype(float)
        b = cv2.cvtColor(p[y0:y1 + 1, x0:x1 + 1], cv2.COLOR_RGB2GRAY).astype(float)
        out.append(np.abs(a - b).mean())
    return float(np.median(out)) if out else 0.0


def event_frames(masks, member, fps):
    ev = member["events"]
    return [k for k in range(int(np.floor(ev["t_event"] * fps)), int(np.ceil(ev.get("t_release", ev["t_event"] + 1) * fps)) + 1)
            if k < len(masks["act_box"]) and masks["act_box"][k][0] >= 0]


# where the released object can lie around the spot X: rest point up to 0.55 m from the root (Blender probe of 70
# avatar/walk pairs) plus a half-diagonal up to 0.42 m, both x1.1: <= 1.055 m; the 16-gon of radius 1.10 m has
# inradius 1.079 m (the F2 review's 1.05 m left 2.5 cm uncovered)
REST_R, REST_N = 1.10, 16


def rest_in_image(cam, xy, r=REST_R, n=REST_N, margin=2.0):
    """True for ground points xy [N,2] whose r m disk does not reach into a black bar of the plate (sides of
    cam.box that are bars, with a margin in px; real frame edges are not tested: a real edge cuts an object as it
    cuts a real one) and lies in front of the camera. All True on plates without bars."""
    xy = np.atleast_2d(np.asarray(xy, float))
    if not cam.bars.any():
        return np.ones(len(xy), bool)
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    P = (xy[:, None, :] + r * np.stack([np.cos(a), np.sin(a)], 1)[None]).reshape(-1, 2)
    uv, z = cam.project(np.c_[P, np.zeros(len(P))])
    (x0, y0, x1, y1), (bl, bt, br, bb) = cam.box, cam.bars
    bad = ((z <= 0.1) | (bl & (uv[:, 0] < x0 + margin)) | (bt & (uv[:, 1] < y0 + margin))
           | (br & (uv[:, 0] > x1 - margin)) | (bb & (uv[:, 1] > y1 - margin)))
    return ~bad.reshape(len(xy), n).any(1)


def feet_on_vehicle_mask(cam, ground_xy, vlab, exempt=()):
    """Fraction of in-view ground points whose pixel lies inside a parked-vehicle mask (plate_vehicles.py label
    image at the render size); vehicles in `exempt` are ignored (e.g. the actor's own car, near side)."""
    if vlab is None or not len(ground_xy):
        return 0.0
    uv, z = cam.project(np.c_[ground_xy, np.zeros(len(ground_xy))])
    H, W = vlab.shape
    vis = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    if not vis.any():
        return 0.0
    lab = vlab[uv[vis, 1].astype(int), uv[vis, 0].astype(int)]
    return float(((lab > 0) & ~np.isin(lab, list(exempt))).mean())


def _box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def traffic_conflict(gr, walkers, ts, cam, traffic, still_norm, obj_xy=None, t_obj=None):
    """True if moving traffic (plate_traffic.py boxes not matching a still parked vehicle, IoU >= 0.5) covers a
    walker's feet (bottom 10% of the box allowed) or the resting object's ground point, at the nearest 2-fps sample
    of the plate time plate_start + t. walkers: list of [T,2] ground tracks at times ts (clip seconds)."""
    if traffic is None or not len(traffic["boxes"]):
        return False
    tt, bx = traffic["t"], traffic["boxes"]
    moving = np.array([not any(_box_iou(b[1:5], s) >= 0.5 for s in still_norm) for b in bx])
    if not moving.any():
        return False
    bx = bx[moving]
    ts = np.asarray(ts, float)
    pt = gr["plate_start"] + ts
    j = np.abs(tt[None, :] - pt[:, None]).argmin(1)
    near = np.abs(tt[j] - pt) <= 0.6
    tracks = [(w, None) for w in walkers]
    if obj_xy is not None:
        tracks.append((np.repeat(np.asarray(obj_xy, float)[None], len(ts), 0), t_obj))
    for track, t_from in tracks:
        uv, z = cam.project(np.c_[track, np.zeros(len(track))])
        ok = near & (z > 0)
        if t_from is not None:
            ok &= ts >= t_from
        if not ok.any():
            continue
        u, v, jj = uv[ok, 0] / cam.W, uv[ok, 1] / cam.H, j[ok]
        for s in np.unique(jj):
            bs = bx[bx[:, 0] == s]
            if not len(bs):
                continue
            m = jj == s
            x0, y0, x1, y1 = bs[:, 1][None], bs[:, 2][None], bs[:, 3][None], bs[:, 4][None]
            uu, vv = u[m][:, None], v[m][:, None]
            if ((uu > x0) & (uu < x1) & (vv > y0) & (vv < y1 - 0.1 * (y1 - y0))).any():
                return True
    return False


def person_overlap(gr, walkers, ts, cam, persons, height=1.8, width=0.7):
    """True if a walker's screen box (a width x height m upright box at its ground position) overlaps a real
    person's YOLO11x box (plates that allow people outside the walkable area: extend_plates.py, det_xe/) by more
    than 10% of the walker box, at the nearest 3-fps sample of plate time. persons: npz with t (plate time), boxes
    (sample index, x0, y0, x1, y1 normalised, conf)."""
    if persons is None or not len(persons["boxes"]):
        return False
    tt, bx = persons["t"], persons["boxes"]
    W, H = cam.W, cam.H
    for w in walkers:
        foot, head, z = cam.person_px(w, height)
        half = 0.5 * width * np.abs(foot[:, 1] - head[:, 1]) / height
        for k, t in enumerate(ts):
            if z[k] <= 0:
                continue
            j = int(np.argmin(np.abs(tt - (gr["plate_start"] + t))))
            if abs(tt[j] - (gr["plate_start"] + t)) > 0.5:
                continue
            a = (foot[k, 0] - half[k], head[k, 1], foot[k, 0] + half[k], foot[k, 1])
            area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
            for b in bx[bx[:, 0] == j]:
                x0, y0, x1, y1 = b[1] * W, b[2] * H, b[3] * W, b[4] * H
                ix = max(0.0, min(a[2], x1) - max(a[0], x0)); iy = max(0.0, min(a[3], y1) - max(a[1], y0))
                if ix * iy > 0.1 * area:
                    return True
    return False


MOTION_LA, MOTION_NEAR, MOTION_MIN_AREA = 1.0, 0.5, 0.002


def motion_regions(t_fg, traffic, size, bottom=None):
    """Vehicle-attributed moving regions per motion frame (plate_motion.py: t, fg [T, h, w]): connected components of
    the foreground (closed 5x5) with area >= MOTION_MIN_AREA of the frame whose box overlaps a vehicle box of
    plate_traffic.py (any vehicle, still or moving: a person at a departing car's tailgate counts) within +-1 s (IoU
    >= 0.2 or inside it). Glass reflections, headlight sweeps and vegetation have no vehicle box (raw motion flagged
    22 of the 292 v2 pairs, attributed motion 10). Returns t and per frame a list of regions (x0, x1, top, bot,
    trunc) in pixels of size (W, H): top/bot = callables column(s) -> highest/lowest region row there; trunc = it
    reaches the bottom of the camera image (bottom px, default H)."""
    import cv2
    t, fg = t_fg
    W, H = size
    h, w = fg.shape[1:]
    sx, sy = W / w, H / h
    yb = (bottom if bottom is not None else H) / sy  # camera image bottom in motion px
    tt, bx = (traffic["t"], traffic["boxes"]) if traffic is not None else (np.zeros(0), np.zeros((0, 6)))
    out = []
    for M, ts in zip(fg, t):
        M = cv2.morphologyEx(M.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(M)
        bb = bx[np.isin(bx[:, 0], np.flatnonzero(np.abs(tt - ts) <= 1.0))] if len(bx) else bx
        R = []
        for c in range(1, n):
            x, y, ww, hh, a = st[c]
            if a < MOTION_MIN_AREA * h * w:
                continue
            bn = [x / w, y / h, (x + ww) / w, (y + hh) / h]
            if not any(_box_iou(bn, b[1:5]) >= 0.2 or (b[1] <= bn[0] and b[2] <= bn[1] and b[3] >= bn[2] and b[4] >= bn[3])
                       for b in bb):
                continue
            sub = lab[y:y + hh, x:x + ww] == c
            r = np.arange(hh)[:, None]
            bot = np.where(sub, r, -1).max(0) + y; top = np.where(sub, r, hh + 1).min(0) + y

            def col(u, arr, x=x, ww=ww):
                return arr[np.clip((np.atleast_1d(u) / sx - x).astype(int), 0, ww - 1)] * sy
            R.append((x * sx, (x + ww) * sx, lambda u, a_=top, f=col: f(u, a_).min(),
                      lambda u, a_=bot, f=col: f(u, a_).max(), y + hh >= yb - 1))
        out.append(R)
    return np.asarray(t, float), out


def motion_conflict(gr, walkers, ts, cam, regions, obj=None, height=1.8, width=0.7):
    """True if a moving vehicle region (motion_regions) meets a walker or the resting object at plate time
    plate_start + t (t in ts, clip seconds) or up to MOTION_LA s later (0.5 s earlier): in the columns of the walker's
    screen box (width x height m at its ground point; the object's footprint) the foot point is inside the region
    (behind or under the vehicle), the region's lowest row is within MOTION_NEAR m on the ground of the foot row
    (contact or approach; no bottom-10% exemption), or the region is cut by the image bottom and the feet are at or
    below its top (a car leaving through the bottom edge as the walker enters there). walkers: [T, 2] ground tracks
    at ts; obj = (xy, half-size m, active [T] bool) or None."""
    tt, R = regions
    ts = np.asarray(ts, float)
    items = []
    for wk in walkers:
        foot, head, z = cam.person_px(wk, height)
        items.append((wk, foot, 0.5 * width * np.abs(foot[:, 1] - head[:, 1]) / height, z, np.ones(len(ts), bool)))
    if obj is not None:
        xy, r, act = obj
        wk = np.repeat(np.asarray(xy, float)[None], len(ts), 0)
        foot, _, z = cam.person_px(wk, 0.0)
        half = abs(cam.project(np.array([[xy[0] + r, xy[1], 0.0]]))[0][0, 0] - foot[0, 0])
        items.append((wk, foot, np.full(len(ts), max(half, 2.0)), z, np.asarray(act, bool)))
    C = cam.C[:2]
    for wk, foot, half, z, act in items:
        for k, t in enumerate(ts):
            if not act[k] or z[k] <= 0:
                continue
            pt = gr["plate_start"] + t
            js = [j for j in np.flatnonzero((tt >= pt - 0.5) & (tt <= pt + MOTION_LA)) if R[j]]
            if not js:
                continue
            u0, u1, vf = foot[k, 0] - half[k], foot[k, 0] + half[k], foot[k, 1]
            g = wk[k]; dirc = (C - g) / max(np.linalg.norm(C - g), 1e-6)
            dpx = abs(cam.project(np.array([[*(g + MOTION_NEAR * dirc), 0.0]]))[0][0, 1] - vf)
            for j in js:
                for x0, x1, top_f, bot_f, trunc in R[j]:
                    if x1 < u0 or x0 > u1:
                        continue
                    uu = np.linspace(max(u0, x0), min(u1, x1), 5)
                    bot, top = float(bot_f(uu)), float(top_f(uu))
                    if top < vf < bot or abs(vf - bot) <= dpx or (trunc and vf >= top):
                        return True
    return False


# object seen apart from the actor (F5, D32): visible object pixels farther than max(APART_MIN_R, APART_DILATE x actor
# box height) px from the actor's pixels (composite.py, masks.npz obj_apart_px); the positive needs >= MIN_APART_S s of
# rest frames with >= MIN_OBJ_PX such pixels at the degraded scale, else the whole group is dropped
APART_DILATE, APART_MIN_R, MIN_APART_S = 0.03, 2, 0.5


def disk(r):
    """Structuring element: offsets dx^2 + dy^2 <= r^2 (uint8 [2r+1, 2r+1])."""
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r).astype(np.uint8)


def apart_radius(act_h):
    return max(APART_MIN_R, int(round(APART_DILATE * act_h)))


def object_apart_s(masks, fps, scale):
    """Seconds of rest frames in which >= MIN_OBJ_PX object pixels (degraded scale) are seen apart from the actor."""
    if "obj_apart_px" not in masks:
        return None
    n = int((masks["obj_at_rest"] & (masks["obj_apart_px"] * scale * scale >= MIN_OBJ_PX)).sum())
    return n / fps


# legs in the resting object (F1, D33), from leg_overlap.py records (horizontal penetration of the leg joints into
# the object's box, per rest frame). Phases of the actor (member events): stride = walk-in until the arrival's
# cross-fade ends (t_arrive + SETTLE), stand = standing at X before the bend, crouch_in / crouch_out = the bend and the
# stand-up, settle = SETTLE s after a stand-up (cross-fade into the idle), turn = leave's idle and in-place turn to the
# exit, pause = pick-up's standing pause before the second bend, crouch2 = pick-up's second bend, walk_out. Standing
# and walking contact is gated at LEG_TH in every outcome (turn: leave only, pause: pick-up only, stand: pre-existing
# only, as the object lies there). The crouch phases (CROUCH_PHASES) are gated in every crouching member (leave,
# both bends of the pick-up, pre-existing, the car-mode families) at a depth by the object's rigidity (D41, step D3;
# D38 S3): the GPU pilot showed a thigh on a rigid box with the shin inside it at 0.10-0.12 m and no visible rigid
# contact at <= 0.095 m, while knees 0.10-0.146 m deep in soft bags read as contact. Carry negatives never crouch, so
# visible crouch contact was a positive-heavy cue. Soft: SOFT_OBJECTS; every other asset is rigid (fail-closed).
LEG_TH, SETTLE = 0.05, 0.3
LEG_GATED = {"stride": "walk-in through object", "stand": "standing in object", "turn": "standing in object",
             "pause": "standing in object", "walk_out": "walk-out through object"}
CROUCH_PHASES = ("crouch_in", "crouch_out", "settle", "crouch2")
SOFT_OBJECTS = ("trashbag", "compost_bag_02", "cement_bag")
CROUCH_TH_RIGID, CROUCH_TH_SOFT = 0.08, 0.15
CROUCH_WHY = "leg in object while crouching"


def crouch_th(asset):
    """Crouch-phase leg depth (m) that rejects a group with this object (D41): CROUCH_TH_SOFT for the soft bags,
    CROUCH_TH_RIGID for every other asset."""
    return CROUCH_TH_SOFT if asset in SOFT_OBJECTS else CROUCH_TH_RIGID


def leg_phase(ev, outcome, t):
    """Motion phase of member time t (see LEG_GATED)."""
    if t < ev["t_event"]:
        return "stride" if t < ev["t_arrive"] + SETTLE else "stand"
    if "t_release" not in ev:  # carry: stands at X, then walks out (its object never rests)
        return "walk_out" if t >= ev["t_walk_out"] else "stand"
    if t < ev["t_release"]:
        return "crouch_in"
    if t < ev["t_stand"]:
        return "crouch_out"
    if t < ev["t_stand"] + SETTLE:
        return "settle"
    if outcome == "pickup" and t < ev["t_walk_out"]:
        return "pause" if t < ev["t_grasp"] - (ev["t_release"] - ev["t_event"]) else "crouch2"
    return "turn" if t < ev["t_walk_out"] else "walk_out"


def leg_phases(ev, outcome, hits):
    """{phase: max depth} of the actor's hits [[t, depth], ...]."""
    out = {}
    for t, d in hits:
        ph = leg_phase(ev, outcome, t)
        out[ph] = max(out.get(ph, 0.0), float(d))
    return out


def leg_verdict(rec, th=LEG_TH, obj=None):
    """Reasons (sorted) to reject a member's group from its leg_overlap.py record: actor depth >= th in a gated phase,
    actor depth >= crouch_th(obj) in a crouch phase (obj: the object asset, default the record's 'object'; records
    without one, written before D41, skip this rule), or any distractor depth >= th."""
    out = rec.get("outcome") or rec["clip"].rsplit("_", 1)[1]
    obj = obj or rec.get("object")
    ph = leg_phases(rec["events"], out, rec["hits"].get("actor", []))
    why = {LEG_GATED[p] for p, d in ph.items() if p in LEG_GATED and d >= th}
    if obj is not None and any(ph.get(p, 0.0) >= crouch_th(obj) for p in CROUCH_PHASES):
        why.add(CROUCH_WHY)
    if any(d >= th for w, h in rec["hits"].items() if w != "actor" for _, d in h):
        why.add("distractor through object")
    return sorted(why)


def poly_dist(poly, P):
    """Signed distance (m) of ground points P [N,2] to a polygon's boundary: > 0 inside, < 0 outside."""
    A = np.asarray(poly, float); B = np.roll(A, -1, 0); AB = B - A
    P = np.atleast_2d(np.asarray(P, float))
    s = np.clip(((P[:, None] - A[None]) * AB[None]).sum(2) / np.maximum((AB ** 2).sum(1), 1e-12)[None], 0.0, 1.0)
    d = np.linalg.norm(P[:, None] - (A[None] + s[..., None] * AB[None]), axis=2).min(1)
    return np.where(inside(A, P), d, -d)


def off_poly(poly, P, margin=0.0):
    """Ground points P [N,2] not inside the polygon by `margin` m (margin 0: the point-in-polygon test alone)."""
    return ~inside(poly, P) if margin == 0.0 else poly_dist(poly, P) < margin


# the distractor ground rule (F3, D36): feet inside the walkable polygon wherever they are in the camera image, except
# one entry and one exit run of <= MAX_RUN m in the border band or far; >= MIN_INSIDE m inside; feet in a bar refused
GROUND_MARGIN, MAX_RUN, MIN_INSIDE = 0.0, 2.0, 3.0
BAND_SIDE, BAND_TOP, BAND_BOTTOM, FAR_REL = 0.35, 1.0, 0.02, 0.04


def ground_ok(cam, poly, xy, step, margin=GROUND_MARGIN):
    """The ground rule of a distractor line (synth/scripts/distractor_lines.py, F3 D36) on ordered ground points xy
    [N,2] of a straight line, `step` m apart. Returns (reason or None, inside length m, (i0, i1) = first and last index
    whose feet are in the camera image).
    - feet (the ground point's pixel) in the frame but outside the camera image cam.box (a black bar), anywhere:
      'feet in a bar' (the F3 review: until a GPU pilot has checked the compositor's bar masking);
    - where the feet are in the camera image, every point lies inside the walkable polygon (by `margin` m), except one
      leading and one trailing run of at most MAX_RUN m where the walker enters or leaves: feet within BAND_SIDE person
      heights of a side, BAND_TOP of the top or BAND_BOTTOM x H of the bottom of the image, or far (a 1.7 m person
      < FAR_REL x H); exemptions anywhere else drew lines along the image edge and on the far horizon (F3 design);
    - the inside stretch is >= MIN_INSIDE m."""
    xy = np.atleast_2d(np.asarray(xy, float))
    foot, head, z = cam.person_px(xy, 1.7)
    ok = (z > 0.5) & np.isfinite(foot).all(1) & np.isfinite(head).all(1)
    fu, fv = np.where(ok, foot[:, 0], -1e9), np.where(ok, foot[:, 1], -1e9)
    x0, y0, x1, y1 = cam.box
    fin = ok & (fu >= x0) & (fu < x1) & (fv >= y0) & (fv < y1)
    if (ok & (fu >= 0) & (fu < cam.W) & (fv >= 0) & (fv < cam.H) & ~fin).any():
        return "feet in a bar", 0.0, None
    idx = np.flatnonzero(fin)
    if not len(idx):
        return "never in view", 0.0, None
    i0, i1 = int(idx[0]), int(idx[-1])
    if len(idx) != i1 - i0 + 1:
        return "view not one interval", 0.0, None
    hpx = np.abs(fv - np.where(ok, head[:, 1], 0.0))[i0:i1 + 1]
    u, v = fu[i0:i1 + 1], fv[i0:i1 + 1]
    band = ((u - x0 < BAND_SIDE * hpx) | (x1 - u < BAND_SIDE * hpx) | (v - y0 < BAND_TOP * hpx)
            | (y1 - v < BAND_BOTTOM * cam.H))
    free = band | (hpx < FAR_REL * cam.H)
    c = np.arange(0, len(free), 10)  # every 10th point first: most lines fail there (the same verdict, faster)
    if (off_poly(poly, xy[i0:i1 + 1][c], margin) & ~free[c]).any():
        return "off the walkable area", 0.0, (i0, i1)
    out = off_poly(poly, xy[i0:i1 + 1], margin)
    if (out & ~free).any():
        return "off the walkable area", 0.0, (i0, i1)
    k = np.flatnonzero(~out)
    if not len(k):
        return "off the walkable area", 0.0, (i0, i1)
    if out[k[0]:k[-1] + 1].any():
        return "gap in the walkable area", 0.0, (i0, i1)
    if max(k[0], len(out) - 1 - k[-1]) * step > MAX_RUN:
        return "long entry or exit off the walkable area", 0.0, (i0, i1)
    ins = float((k[-1] - k[0]) * step)
    return (None if ins >= MIN_INSIDE else "short"), ins, (i0, i1)
