"""Longer contest plates (limit 2): only people in the walkable area break a plate (spec invariant 3), not people
anywhere in the frame (find_plates.py's stricter first rule).
python synth/scripts/extend_plates.py runs     -> synth/data/plates_ext_candidates.csv (CPU, from YOLO11s det/)
python synth/scripts/extend_plates.py verify   -> YOLO11x on the candidate windows (GPU), synth/data/det_xe/<plate>.npz
                                                  (person boxes with plate-relative sample times), plates_ext.csv,
                                                  calib JSONs and calib_check rows for the new plate ids
A person occupies a sample if the ground point under the box bottom is inside the calibrated walkable polygon or
within NEAR m of it (or cannot be projected). Free runs: PAD s padding, >= MIN_LEN s, contest positives never use
[g - 2, g + 10]; runs that do not add >= 2 s to the video's existing plates are dropped. New plate ids: <video>_x<k>.
Videos: contest clips with at least one plate that passed the stage-2 check (their camera is validated)."""
import json, os, pathlib, sys
os.environ.setdefault("YOLO_AUTOINSTALL", "false"); os.environ.setdefault("YOLO_OFFLINE", "true")
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
SYN = HERE.parent
sys.path[:0] = [str(HERE), str(SYN / "gen")]
from find_plates import free_runs, FPS
from camera import Camera, inside
from scenario import boundary_points

NEAR, PAD, MIN_LEN = 2.0, 1.0, 5.0


def occupied(boxes, cam, cal, n):
    W, H = cal["image_size"]
    poly = np.asarray(cal["walkable"], float); bp = boundary_points(poly, 0.25)
    occ = np.zeros(n, bool)
    if not len(boxes):
        return occ
    gp = cam.ground(np.c_[(boxes[:, 1] + boxes[:, 3]) / 2 * W, boxes[:, 4] * H])
    fin = np.isfinite(gp).all(1)
    near = ~fin
    if fin.any():
        dmin = np.min(np.linalg.norm(gp[fin][:, None, :] - bp[None], axis=2), axis=1)
        near[fin] = inside(poly, gp[fin]) | (dmin < NEAR)
    occ[boxes[near, 0].astype(int)] = True
    return occ


def runs():
    P = pd.read_csv(SYN / "data/plates.csv"); chk = pd.read_csv(SYN / "data/calib_check.csv")
    ok = set(chk[chk.ok].plate_id)
    idx = pd.read_csv(SYN.parent / "cache/index.csv").set_index("video")
    rows = []
    for v, g in P[P.plate_id.isin(ok) & (P.source == "contest")].groupby("video"):
        base = g.iloc[0]
        cal = json.loads((SYN / f"calib/{base.plate_id}.json").read_text()); cam = Camera(cal)
        d = np.load(SYN / f"data/det/{v}.npz")
        occ = occupied(d["boxes"], cam, cal, int(d["n"]))
        y, gg = idx.loc[v, "y"], idx.loc[v, "g"]
        banned = [(gg - 2, gg + 10)] if y == 1 else []
        old = sum(g.dur)
        rr = free_runs(occ, float(d["T"]), FPS, PAD, MIN_LEN, banned)
        if sum(e - s for s, e in rr) < old + 2.0:
            continue
        for k, (s, e) in enumerate(rr):
            rows.append(dict(base._asdict() if hasattr(base, "_asdict") else base.to_dict(), start=round(s, 3),
                             end=round(e, 3), dur=round(e - s, 3), plate_id=f"{v}_x{k}", calib_from=base.plate_id))
    df = pd.DataFrame(rows)
    df.to_csv(SYN / "data/plates_ext_candidates.csv", index=False)
    print(f"{len(df)} candidate plates from {df.video.nunique()} videos, {df.dur.sum():.0f} s")


def verify():
    from ultralytics import YOLO
    from contact_sheet import frames_at
    model = YOLO(str(SYN / "assets/yolo11x.pt"))
    C = pd.read_csv(SYN / "data/plates_ext_candidates.csv", dtype={"label": str})
    out = SYN / "data/det_xe"; out.mkdir(parents=True, exist_ok=True)
    keep, chk_rows = [], []
    for r in C.itertuples():
        cal = json.loads((SYN / f"calib/{r.calib_from}.json").read_text()); cam = Camera(cal)
        ts = list(np.arange(r.start, r.end, 1 / FPS))
        fr = frames_at(r.path, ts, 1280)
        boxes = []
        for i in range(0, len(fr), 16):
            res = model.predict([x[1][..., ::-1].copy() for x in fr[i:i + 16]], imgsz=1280, conf=0.10, classes=[0],
                                verbose=False)
            for j, rr in enumerate(res):
                for b, c in zip(rr.boxes.xyxyn.cpu().numpy(), rr.boxes.conf.cpu().numpy()):
                    boxes.append([i + j, *b, c])
        boxes = np.array(boxes, np.float32).reshape(-1, 6)
        np.savez_compressed(out / f"{r.plate_id}.npz", boxes=boxes, t=np.array([t for t, _ in fr]))
        occ = occupied(boxes, cam, cal, len(fr))
        k = 0
        for s, e in free_runs(occ, len(fr) / FPS, FPS, PAD, MIN_LEN):
            pid = f"{r.plate_id}{'abcdefghijklmnop'[k]}"; k += 1
            keep.append(dict(r._asdict(), start=round(r.start + s, 3), end=round(r.start + e, 3), dur=round(e - s, 3),
                             plate_id=pid, det_xe=r.plate_id))
            js = dict(cal, plate_id=pid); (SYN / f"calib/{pid}.json").write_text(json.dumps(js))
            chk_rows.append(dict(plate_id=pid, ok=True, why="", inherited_from=r.calib_from))
        print(r.plate_id, len(fr), "frames", len(boxes), "boxes ->", k, "plates", flush=True)
    K = pd.DataFrame(keep).drop(columns=["Index"])
    K.to_csv(SYN / "data/plates_ext.csv", index=False)
    P = pd.read_csv(SYN / "data/plates.csv", dtype={"label": str})  # new plates join the plate table
    P = pd.concat([P[~P.plate_id.isin(K.plate_id)], K[P.columns]], ignore_index=True)
    P.to_csv(SYN / "data/plates.csv", index=False)
    chk = pd.read_csv(SYN / "data/calib_check.csv")
    chk = pd.concat([chk[~chk.plate_id.isin(K.plate_id)], pd.DataFrame(chk_rows)], ignore_index=True)
    chk.to_csv(SYN / "data/calib_check.csv", index=False)
    print(f"{len(K)} extended plates, {K.dur.sum():.0f} s, {K.video.nunique()} videos")


if __name__ == "__main__":
    runs() if sys.argv[1] == "runs" else verify()
