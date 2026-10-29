"""Stage 1, second pass: re-check every plate with a stronger detector and drop dark plates.
python synth/scripts/verify_plates.py   (after find_plates.py; keeps its output as synth/data/plates_s.csv)
YOLO11x (imgsz 1280, conf >= CONF) on the 3 fps samples inside each plate; any person box occupies +-PAD s.
The stage-1 inspection found people missed by YOLO11s at conf 0.25: half hidden by a van door, an arm out of a
car window, a pixelated person beside a car (DECISIONS.md D09). Plates are re-split into free runs >= MIN_LEN s and
renumbered; plates with mean luma < MIN_LUMA (nothing visible) are dropped."""
import os, pathlib, sys, concurrent.futures as cf
os.environ.setdefault("YOLO_AUTOINSTALL", "false"); os.environ.setdefault("YOLO_OFFLINE", "true")
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from find_plates import sample, free_runs, FPS, LONG

SYN = pathlib.Path(__file__).resolve().parents[1]
CONF, PAD, MIN_LEN, MIN_LUMA, MAX_LEN = 0.10, 1.0, 5.0, 20.0, 120.0
WEIGHTS = SYN / "assets/yolo11x.pt"
DETX = SYN / "data/det_x"


def main():
    from ultralytics import YOLO
    src = SYN / "data/plates_s.csv"
    if not src.exists():
        (SYN / "data/plates.csv").rename(src)
    P = pd.read_csv(src, dtype={"label": str})
    model = YOLO(str(WEIGHTS))
    DETX.mkdir(parents=True, exist_ok=True)
    vids = [v for v in P.video.unique() if not (DETX / f"{v}.npz").exists()]
    path = P.groupby("video").path.first()

    def load(v):
        return v, sample(path[v])[0]

    def bounded(ex, items, ahead=2):  # 5-min MEVA clips are ~2.5 GB decoded
        pending = set()
        for v in items:
            pending.add(ex.submit(load, v))
            if len(pending) >= ahead:
                done, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                yield from done
        yield from cf.as_completed(pending)

    with cf.ThreadPoolExecutor(2) as ex:
        for n, fut in enumerate(bounded(ex, vids)):
            v, fr = fut.result()
            pl = P[P.video == v]
            keep = np.zeros(len(fr), bool)
            for r in pl.itertuples():
                keep[int(np.floor(r.start * FPS)):int(np.ceil(r.end * FPS)) + 1] = True
            idx = np.flatnonzero(keep[:len(fr)])
            boxes = []
            for i in range(0, len(idx), 16):
                res = model.predict([fr[k] for k in idx[i:i + 16]], imgsz=LONG, conf=CONF, classes=[0], verbose=False)
                for k, r in zip(idx[i:i + 16], res):
                    for b, c in zip(r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                        boxes.append([k, *b, c])
            np.savez_compressed(DETX / f"{v}.npz", boxes=np.array(boxes, np.float32).reshape(-1, 6), n=len(fr),
                                checked=idx)
            print(n, v, len(idx), "checked", len(boxes), "boxes", flush=True)

    rows = []
    for v, pl in P.groupby("video", sort=False):
        d = np.load(DETX / f"{v}.npz")
        occ = np.zeros(int(d["n"]), bool); occ[d["boxes"][:, 0].astype(int)] = True
        det = np.load(SYN / f"data/det/{v}.npz")
        k = 0
        for r in pl.itertuples():
            for s, e in free_runs(occ, float(det["T"]), FPS, PAD, MIN_LEN, [(0, r.start), (r.end, 1e9)]):
                sel = slice(int(np.ceil(s * FPS)), int(np.floor(e * FPS)))
                luma = float(det["luma"][sel].mean())
                if luma < MIN_LUMA:
                    continue
                rows.append(dict(r._asdict(), start=round(s, 3), end=round(e, 3), dur=round(e - s, 3),
                                 luma=round(luma, 1), chroma=round(float(det["chroma"][sel].mean()), 2),
                                 plate_id=f"{v}_p{k}"))
                k += 1
    out = pd.DataFrame(rows).drop(columns=["Index"])
    out.to_csv(SYN / "data/plates.csv", index=False)
    print(f"verified: {len(out)}/{len(P)} plates kept (after re-splitting), {out.video.nunique()} videos, "
          f"{out.scene_id.nunique()} scenes, {out.dur.sum() / 3600:.2f} h; by source "
          f"{out.groupby('source').size().to_dict()}")


if __name__ == "__main__":
    main()
