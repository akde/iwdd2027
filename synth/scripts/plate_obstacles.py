"""Parked vehicles and local darkness per plate (actors must not walk over real cars; events must be visible).
python synth/scripts/plate_obstacles.py -> synth/data/obstacles.json {plate_id: {"vehicles": [[x0,y0,x1,y1] normalised],
                                                                               "luma": [[g x g] grid, 0-255]}}
Vehicles: YOLO11x (imgsz 1280, conf >= 0.3; car, motorcycle, bus, truck, bicycle) on 5 frames over the plate;
a box is kept if a box of the same class overlaps it (IoU >= 0.5) in at least 3 of the 5 frames (parked, not passing).
luma: mean 8-bit luma of a 16x16 grid of the middle frame (darkness check of the event spot)."""
import json, os, pathlib, sys
os.environ.setdefault("YOLO_AUTOINSTALL", "false"); os.environ.setdefault("YOLO_OFFLINE", "true")
import cv2, numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from contact_sheet import frames_at

SYN = pathlib.Path(__file__).resolve().parents[1]
CLASSES = [1, 2, 3, 5, 7]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def main():
    from ultralytics import YOLO
    model = YOLO(str(SYN / "assets/yolo11x.pt"))
    plates = pd.read_csv(SYN / "data/plates.csv")
    chk = pd.read_csv(SYN / "data/calib_check.csv")
    plates = plates[plates.plate_id.isin(chk[chk.ok].plate_id)]
    js = SYN / "data/obstacles.json"
    out = json.loads(js.read_text()) if js.exists() else {}  # incremental: plates already done are kept
    for r in plates.itertuples():
        if r.plate_id in out:
            continue
        ts = list(np.linspace(r.start + 0.1, r.end - 0.1, 5))
        fr = frames_at(r.path, ts, 1280)
        dets = []
        for _, f in fr:
            res = model.predict(f[..., ::-1].copy(), imgsz=1280, conf=0.3, classes=CLASSES, verbose=False)[0]
            dets.append([(int(c), b.tolist()) for b, c in zip(res.boxes.xyxyn.cpu().numpy(), res.boxes.cls.cpu().numpy())])
        keep = []
        for c, b in dets[len(dets) // 2] + [d for ds in dets for d in ds]:
            n = sum(any(cc == c and iou(b, bb) >= 0.5 for cc, bb in ds) for ds in dets)
            if n >= 3 and not any(iou(b, k) >= 0.5 for k in keep):
                keep.append(b)
        g = cv2.cvtColor(fr[len(fr) // 2][1], cv2.COLOR_RGB2GRAY)
        luma = cv2.resize(g, (16, 16), interpolation=cv2.INTER_AREA).astype(float)
        out[r.plate_id] = dict(vehicles=[[round(x, 4) for x in b] for b in keep], luma=luma.round(1).tolist())
        print(r.plate_id, len(keep), "parked vehicles", flush=True)
    (SYN / "data/obstacles.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
