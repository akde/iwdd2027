"""All vehicle boxes over time per usable plate (moving traffic must not run over the actor or the object).
python synth/scripts/plate_traffic.py -> synth/data/traffic/<plate_id>.npz: t (s, plate PTS), boxes [N,6] =
(sample index, x0, y0, x1, y1 normalised, cls). YOLO11x, imgsz 1280, conf >= 0.3, classes as plate_obstacles.py,
2 frames per second over the plate window (bounded decode: one plate at a time)."""
import os, pathlib, sys
os.environ.setdefault("YOLO_AUTOINSTALL", "false"); os.environ.setdefault("YOLO_OFFLINE", "true")
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from contact_sheet import frames_at
from plate_obstacles import CLASSES

SYN = pathlib.Path(__file__).resolve().parents[1]
FPS = 2.0


def main():
    from ultralytics import YOLO
    model = YOLO(str(SYN / "assets/yolo11x.pt"))
    plates = pd.read_csv(SYN / "data/plates.csv")
    chk = pd.read_csv(SYN / "data/calib_check.csv")
    plates = plates[plates.plate_id.isin(chk[chk.ok].plate_id)]
    out = SYN / "data/traffic"; out.mkdir(parents=True, exist_ok=True)
    for r in plates.itertuples():
        f = out / f"{r.plate_id}.npz"
        if f.exists():
            continue
        ts = list(np.arange(r.start, r.end, 1 / FPS))
        fr = frames_at(r.path, ts, 1280)
        boxes = []
        for i in range(0, len(fr), 16):
            res = model.predict([x[1][..., ::-1].copy() for x in fr[i:i + 16]], imgsz=1280, conf=0.3, classes=CLASSES,
                                verbose=False)
            for j, rr in enumerate(res):
                for b, c in zip(rr.boxes.xyxyn.cpu().numpy(), rr.boxes.cls.cpu().numpy()):
                    boxes.append([i + j, *b, c])
        np.savez_compressed(f, t=np.array([t for t, _ in fr]), boxes=np.array(boxes, np.float32).reshape(-1, 6))
        print(r.plate_id, len(fr), "frames", len(boxes), "boxes", flush=True)


if __name__ == "__main__":
    main()
