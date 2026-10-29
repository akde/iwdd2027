"""Stage 2: automatic camera calibration from pedestrian boxes of the same camera, walkable polygon from their feet.
python synth/scripts/calibrate.py   -> synth/calib/<plate_id>.json for every plate that passes, synth/data/calib_summary.csv

Camera model: pinhole, principal point at the image centre, square pixels, roll 0, pitch theta (down), height h (m),
focal f (px). People are vertical segments of PERSON_H m standing on the ground plane z = 0.
Fit: for each upright, untruncated person box (YOLO11s, synth/data/det, conf >= 0.5), the foot pixel (box bottom
centre) is cast to the ground, a PERSON_H m segment is raised and projected; the residual is the log ratio of the
predicted and observed pixel heights. Robust least squares (soft L1) with weak log-normal priors on f (HFOV 70 deg),
h (5 m) and theta (20 deg); one refit on inliers (|res| < 0.25). Boxes pool over the source video (contest) or over
all clips of the camera (MEVA).
Contest clips also pool the boxes of same-view clips of their scene (same_view). Walkable polygon: ground foot points of inliers, 1 m disks on a 0.25 m grid, closed, largest component, simplified.
Static check: phase correlation between 1 fps plate frames (640 px); a plate with a shift > 1.5 px is dropped.
Content box (black pillarbox/letterbox bars outside it): plate_content.content_boxes, the same rule as plate_content.py.
Fail rules (plate dropped, not hand-fixed): < MIN_BOX inliers, foot rows spanning < 8% of the image height,
calibration error (median |relative height error| of inliers) > 12%, moving camera."""
import json, pathlib, sys
import cv2, numpy as np, pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from contact_sheet import frames_at
from plate_content import content_boxes

ROOT = pathlib.Path(__file__).resolve().parents[2]
SYN = ROOT / "synth"
PERSON_H, MIN_BOX, MAX_ERR, MIN_SPAN, MAX_SHIFT = 1.70, 8, 0.12, 0.04, 1.5
HFOV0, H0, TH0 = 70.0, 5.0, 20.0


def basis(theta):
    s, c = np.sin(theta), np.cos(theta)
    return np.array([1.0, 0, 0]), np.array([0, -s, -c]), np.array([0, c, -s])  # x_c, y_c (image down), z_c (forward)


def ground(u, v, f, theta, h, W, H):
    """Image points -> ground (x, y); nan above the horizon."""
    x, y = (u - W / 2) / f, (v - H / 2) / f
    s, c = np.sin(theta), np.cos(theta)
    den = s + y * c
    t = np.where(den > 1e-6, h / np.where(den > 1e-6, den, 1), np.nan)
    return np.stack([x * t, (c - y * s) * t], -1)


def project(P, f, theta, h, W, H):
    """World points [N,3] -> pixels [N,2] (and depth)."""
    xc, yc, zc = basis(theta)
    d = P - np.array([0, 0, h])
    z = d @ zc
    return np.stack([W / 2 + f * (d @ xc) / z, H / 2 + f * (d @ yc) / z], -1), z


def head_v(ub, vb, f, theta, h, W, H):
    g = ground(ub, vb, f, theta, h, W, H)
    P = np.concatenate([g, np.full((len(g), 1), PERSON_H)], 1)
    return project(P, f, theta, h, W, H)[0][:, 1]


def residuals(p, ub, vb, vt, W, H):
    lf, theta, lh = p
    f, h = np.exp(lf), np.exp(lh)
    hp = vb - head_v(ub, vb, f, theta, h, W, H)
    r = np.log(np.clip(hp, 1e-3, None) / (vb - vt))
    r = np.where(np.isfinite(r), r, 2.0)
    prior = [(lf - np.log(W / 2 / np.tan(np.radians(HFOV0 / 2)))) / 0.35, (theta - np.radians(TH0)) / np.radians(25),
             (lh - np.log(H0)) / 0.8]
    return np.concatenate([r, np.array(prior) * 0.3])


def boxes_for(vids, W, H):
    out = []
    for v in vids:
        b = np.load(SYN / f"data/det/{v}.npz")["boxes"]
        out.append(b)
    b = np.concatenate(out) if out else np.zeros((0, 6))
    x1, y1, x2, y2, conf = b[:, 1] * W, b[:, 2] * H, b[:, 3] * W, b[:, 4] * H, b[:, 5]
    ok = (conf >= 0.5) & (b[:, 2] > 0.01) & (b[:, 4] < 0.99) & (b[:, 1] > 0.005) & (b[:, 3] < 0.995)
    asp = (y2 - y1) / np.maximum(x2 - x1, 1)
    ok &= (asp > 1.8) & (asp < 5.0) & (y2 - y1 > 12)
    return (x1[ok] + x2[ok]) / 2, y2[ok], y1[ok]


def fit(ub, vb, vt, W, H):
    p0 = [np.log(W / 2 / np.tan(np.radians(HFOV0 / 2))), np.radians(TH0), np.log(H0)]
    bounds = ([np.log(W * 0.2), np.radians(-5), np.log(1.0)], [np.log(W * 4), np.radians(85), np.log(60)])
    sol = least_squares(residuals, p0, args=(ub, vb, vt, W, H), loss="soft_l1", f_scale=0.1, bounds=bounds)
    r = residuals(sol.x, ub, vb, vt, W, H)[:len(ub)]
    inl = np.abs(r) < 0.25
    if inl.sum() >= MIN_BOX:
        sol = least_squares(residuals, sol.x, args=(ub[inl], vb[inl], vt[inl], W, H), loss="soft_l1", f_scale=0.1,
                            bounds=bounds)
        r = residuals(sol.x, ub, vb, vt, W, H)[:len(ub)]
        inl = np.abs(r) < 0.25
    return sol.x, r, inl


def walkable(gxy, cell=0.25, rad=1.0):
    gxy = gxy[np.isfinite(gxy).all(1)]
    lo = gxy.min(0) - 2 * rad
    n = np.ceil((gxy.max(0) + 2 * rad - lo) / cell).astype(int) + 1
    if n.prod() > 4e6:
        return None
    g = np.zeros((n[1], n[0]), np.uint8)
    for x, y in ((gxy - lo) / cell).astype(int):
        cv2.circle(g, (int(x), int(y)), int(rad / cell), 255, -1)
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cs, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 0.2 / cell, True)[:, 0].astype(float)
    return (c * cell + lo).round(3).tolist(), float(cv2.contourArea(c.astype(np.float32)) * cell * cell)


def static_shift(path, start, end):
    ts = list(np.arange(start + 0.1, end - 0.05, 1.0))
    fr = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for _, f in frames_at(path, ts, 640)]
    if len(fr) < 2:
        return 0.0
    win = cv2.createHanningWindow(fr[0].shape[::-1], cv2.CV_32F)
    return float(max(np.hypot(*cv2.phaseCorrelate(fr[0], f, win)[0]) for f in fr[1:]))


def blender_camera(f, theta, h, W, H):
    """4x4 world matrix in Blender's camera convention (looks along local -Z, local +Y is image up)."""
    xc, yc, zc = basis(theta)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = xc, -yc, -zc, [0, 0, h]
    return M


def same_view(a, b):
    """Two contest clips show the same camera view: same frame shape, median frames (cache/median_frames, from
    tools/clip_index.py) aligned within 3 px at 320 px wide and edge maps correlated > 0.3."""
    A, B = (cv2.imread(str(ROOT / f"cache/median_frames/{v}.png"), cv2.IMREAD_GRAYSCALE) for v in (a, b))
    if A is None or B is None or abs(A.shape[1] / A.shape[0] - B.shape[1] / B.shape[0]) > 0.01:
        return False
    size = (320, round(320 * A.shape[0] / A.shape[1]))
    A, B = (cv2.resize(x, size, interpolation=cv2.INTER_AREA).astype(np.float32) for x in (A, B))
    (dx, dy), _ = cv2.phaseCorrelate(A, B, cv2.createHanningWindow(size, cv2.CV_32F))
    ea, eb = (cv2.Canny(x.astype(np.uint8), 50, 150).astype(np.float32).ravel() for x in (A, B))
    return bool(np.hypot(dx, dy) < 3 and np.corrcoef(ea, eb)[0, 1] > 0.3)


def sources(cam, P, folds):
    """Videos whose person boxes calibrate this camera: all clips of a MEVA camera; for a contest clip, itself and
    the clips of its scene with the same view (boxes are stored normalised, so they pool across resolutions)."""
    if P.source.iloc[0] == "meva":
        return sorted(p.stem for p in (SYN / "data/det").glob("2018-*.npz") if p.stem.split("_")[-2] == cam.split("_")[1])
    sib = folds[(folds.scene == P.scene_id.iloc[0]) & (folds.video != cam)].video
    return [cam] + sorted(v for v in sib if same_view(cam, v))


def main():
    plates = pd.read_csv(SYN / "data/plates.csv", dtype={"label": str})
    folds = pd.read_csv(ROOT / "data/folds.csv")
    (SYN / "calib").mkdir(exist_ok=True)
    for f in (SYN / "calib").glob("*.json"):
        f.unlink()  # derived; plates that fail must not keep an old calibration
    plates["cam"] = np.where(plates.source == "meva", plates.scene_id, plates.video)
    rows = []
    for cam, P in plates.groupby("cam"):
        W, H = int(P.w.iloc[0]), int(P.h.iloc[0])
        src = sources(cam, P, folds)
        ub, vb, vt = boxes_for(src, W, H)
        rec = dict(cam=cam, n_boxes=len(ub), n_src=len(src))
        if len(ub) < MIN_BOX:
            rows += [dict(rec, plate_id=p, ok=False, why="few boxes") for p in P.plate_id]; continue
        (lf, theta, lh), r, inl = fit(ub, vb, vt, W, H)
        f, h = float(np.exp(lf)), float(np.exp(lh))
        err = float(np.median(np.abs(np.expm1(r[inl])))) if inl.any() else 1.0
        span = float((np.percentile(vb[inl], 95) - np.percentile(vb[inl], 5)) / H) if inl.sum() > 2 else 0.0
        rec.update(n_inl=int(inl.sum()), f=round(f, 1), pitch_deg=round(float(np.degrees(theta)), 2), height_m=round(h, 2),
                   err=round(err, 4), span=round(span, 3))
        why = ("few inliers" if inl.sum() < MIN_BOX else "narrow foot rows" if span < MIN_SPAN else
               "height error" if err > MAX_ERR else "")
        wk = walkable(ground(ub[inl], vb[inl], f, theta, h, W, H)) if not why else None
        if not why and wk is None:
            why = "walkable too large"
        content = content_boxes(P)[1] if not why else {}
        for p in P.itertuples():
            shift = static_shift(p.path, p.start, p.end)
            w_why = why or ("moving camera" if shift > MAX_SHIFT else "")
            rows.append(dict(rec, plate_id=p.plate_id, ok=not w_why, why=w_why, shift=round(shift, 2)))
            if w_why:
                continue
            k = min(1.0, 960 / W, 540 / H)  # render size cap (DECISIONS.md D17)
            js = dict(plate_id=p.plate_id, video=p.video, scene_id=p.scene_id, fold=int(p.fold), source=p.source,
                      image_size=[W, H], render_size=[round(W * k) // 2 * 2, round(H * k) // 2 * 2],
                      content=content[p.plate_id], focal_px=round(f, 2),
                      camera_world=blender_camera(f, theta, h, W, H).round(6).tolist(), camera_convention=
                      "Blender: looks along local -Z, local +Y up; world Z up, metres, ground z = 0",
                      pitch_deg=rec["pitch_deg"], roll_deg=0.0, height_m=rec["height_m"],
                      horizon_v=round(H / 2 - f * np.tan(theta), 1), person_h_m=PERSON_H,
                      walkable=wk[0], walkable_area_m2=round(wk[1], 1), occluders=[],
                      calib_error_pct=round(100 * err, 2), n_boxes=len(ub), n_inliers=int(inl.sum()),
                      foot_row_span=rec["span"], static_shift_px=round(shift, 2),
                      fit_boxes=dict(u=ub[inl].round(1).tolist(), vb=vb[inl].round(1).tolist(), vt=vt[inl].round(1).tolist()))
            (SYN / f"calib/{p.plate_id}.json").write_text(json.dumps(js))
    df = pd.DataFrame(rows)
    df.to_csv(SYN / "data/calib_summary.csv", index=False)
    print(df.why.replace("", "ok").value_counts().to_string())
    ok = df[df.ok]
    print(f"{len(ok)}/{len(df)} plates calibrated; scenes {plates[plates.plate_id.isin(ok.plate_id)].scene_id.nunique()}")


if __name__ == "__main__":
    main()
