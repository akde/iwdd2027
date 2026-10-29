"""Parked-vehicle instance masks per usable plate (occlusion of synthetic people/objects, vehicle families).
python synth/scripts/plate_vehicles.py -> synth/data/vehicles/<plate_id>.png (uint8 label image at the render size,
0 = none, k = vehicle k) + synth/data/vehicles.json {plate_id: [{id, cls, box (render px), bottom_v, contact_uv}]}
YOLO11x-seg (imgsz 1280, conf >= 0.3; bicycle, car, motorcycle, bus, truck) on 9 frames over the plate window; a
vehicle is parked if a detection of the same class overlaps it (box IoU >= 0.7) in at least 6 of the 9 frames
(n_frames stored; car mode needs all 9: present and still for the whole window); its mask is the middle frame's. bottom_v = lowest mask row (ground contact, used for depth ordering)."""
import json, os, pathlib, sys
os.environ.setdefault("YOLO_AUTOINSTALL", "false"); os.environ.setdefault("YOLO_OFFLINE", "true")
import cv2, numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from contact_sheet import frames_at
from plate_obstacles import iou, CLASSES

SYN = pathlib.Path(__file__).resolve().parents[1]


def main():
    from ultralytics import YOLO
    model = YOLO(str(SYN / "assets/yolo11x-seg.pt"))
    plates = pd.read_csv(SYN / "data/plates.csv")
    chk = pd.read_csv(SYN / "data/calib_check.csv")
    plates = plates[plates.plate_id.isin(chk[chk.ok].plate_id)]
    (SYN / "data/vehicles").mkdir(parents=True, exist_ok=True)
    js = SYN / "data/vehicles.json"
    meta = json.loads(js.read_text()) if js.exists() else {}  # incremental: plates already done are kept
    for r in plates.itertuples():
        if r.plate_id in meta:
            continue
        cal = json.loads((SYN / f"calib/{r.plate_id}.json").read_text())
        W, H = cal["render_size"]
        NF = 9  # frames over the plate window
        fr = [f for _, f in frames_at(r.path, list(np.linspace(r.start + 0.1, r.end - 0.1, NF)), max(r.w, r.h))]
        fr = [cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in fr]
        dets = []
        for f in fr:
            res = model.predict(f[..., ::-1].copy(), imgsz=1280, conf=0.3, classes=CLASSES, verbose=False, retina_masks=True)[0]
            ms = res.masks.data.cpu().numpy() > 0.5 if res.masks is not None else np.zeros((0, H, W), bool)
            dets.append([(int(c), b.tolist(), m) for b, c, m in
                         zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), ms)])
        lab = np.zeros((H, W), np.uint8); vs = []
        for c, b, m in dets[NF // 2]:
            n = sum(any(cc == c and iou(b, bb) >= 0.7 for cc, bb, _ in ds) for ds in dets)
            if n < (NF + 1) // 2 + 1 or m.sum() < 30:
                continue
            k = len(vs) + 1
            lab[m & (lab == 0)] = k
            ys, xs = np.nonzero(m)
            bottom = int(ys.max())
            vs.append(dict(id=k, cls=c, box=[round(x, 1) for x in b], bottom_v=bottom, n_frames=int(n), n_of=NF,
                           contact_uv=[float(xs[ys >= bottom - 2].mean()), float(bottom)]))
        cv2.imwrite(str(SYN / f"data/vehicles/{r.plate_id}.png"), lab)
        meta[r.plate_id] = vs
        print(r.plate_id, len(vs), "parked vehicles", flush=True)
    (SYN / "data/vehicles.json").write_text(json.dumps(meta))


if __name__ == "__main__":
    main()
