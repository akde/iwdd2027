"""Stage 1: person-free plate segments.
python synth/scripts/find_plates.py [--min-len 5] [--src contest meva]
Per video: frames sampled at FPS on the PTS grid, YOLO11s person boxes (conf >= CONF, imgsz 1280), cached in
synth/data/det/<video>.npz (t, frame index in the sample list, x1 y1 x2 y2 normalised, conf; also reused for calibration).
A sample is occupied if any person box exists; occupied samples are dilated by PAD s; free runs >= min-len become
plates, split into chunks of at most MAX_LEN s. Contest positives never use [g - 2, g + 10].
Output synth/data/plates.csv: video, start, end, dur, luma, chroma, label, plate_id, scene_id, fold, source
(+ path, w, h, fps). Contest scene/fold from data/folds.csv; MEVA scene = meva_<camera>, fold = -1 (train in all folds)."""
import argparse, os, pathlib, sys, concurrent.futures as cf
os.environ.setdefault("YOLO_AUTOINSTALL", "false"); os.environ.setdefault("YOLO_OFFLINE", "true")
import av, numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
SYN = ROOT / "synth"
DET = SYN / "data/det"
FPS, CONF, PAD, MAX_LEN, LONG = 3.0, 0.25, 0.7, 120.0, 1280
WEIGHTS = SYN / "assets/yolo11s.pt"
# outdoor, visible-light MEVA cameras (screening sheet synth/data/meva/screen{1,2}.jpg; DECISIONS.md D08)
MEVA_CAMS = {"G300", "G301", "G328", "G336", "G340", "G341", "G424", "G436", "G505", "G506", "G638"}
# clips whose view changes mid-clip (phase correlation across the clip) or show nothing but flares (D16)
MEVA_EXCLUDE = {"2018-03-07_17-20-06_17-25-03_school_G300_r13", "2018-03-11_23-55-09_00-00-09_hospital_G301_r13"}


def sample(path, fps=FPS, long_side=LONG):
    """Frames on the grid t = k / fps (last frame with pts <= t), BGR, long side <= long_side; T as src/data.py."""
    out, ts, prev, p0 = [], [], None, None
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        W, H = s.codec_context.width, s.codec_context.height
        k = min(1.0, long_side / max(W, H))
        for f in c.decode(s):
            if f.pts is None or (prev is not None and f.pts <= prev[0].pts):
                continue
            p0 = f.pts if p0 is None else p0
            t = float((f.pts - p0) * s.time_base)
            while prev is not None and len(out) / fps < t:
                out.append(prev[0].reformat(width=round(W * k) // 2 * 2, height=round(H * k) // 2 * 2,
                                            format="bgr24", interpolation="AREA").to_ndarray())
            prev = (f, t); ts.append(t)
        T = ts[-1] + (float(np.median(np.diff(ts))) if len(ts) > 1 else 0.0)
        fr = float(1 / np.median(np.diff(ts))) if len(ts) > 1 else 0.0
        while len(out) / fps < T:
            out.append(prev[0].reformat(width=round(W * k) // 2 * 2, height=round(H * k) // 2 * 2,
                                        format="bgr24", interpolation="AREA").to_ndarray())
    return out, T, W, H, fr


def look(frames):
    """Mean luma and mean chroma (|Cb - 128| + |Cr - 128|) over a list of BGR frames."""
    import cv2
    y = c = 0.0
    for f in frames:
        ycc = cv2.cvtColor(f[::4, ::4], cv2.COLOR_BGR2YCrCb).astype(np.float32)
        y += ycc[..., 0].mean(); c += (np.abs(ycc[..., 1] - 128) + np.abs(ycc[..., 2] - 128)).mean()
    return y / len(frames), c / len(frames)


def videos(srcs):
    rows = []
    if "contest" in srcs:
        idx = pd.read_csv(ROOT / "cache/index.csv").merge(pd.read_csv(ROOT / "data/folds.csv"), on="video")
        for r in idx.itertuples():
            rows.append(dict(video=r.video, path=str(ROOT / f"data/{r.split}/videos/{r.video}.mp4"), source="contest",
                             label="true" if r.y == 1 else "false", g=r.g, scene_id=r.scene, fold=r.fold))
    if "meva" in srcs:
        for p in sorted((SYN / "data/meva/videos").glob("*.avi")):
            cam = p.name.split(".")[-3]
            if cam not in MEVA_CAMS or p.stem.replace(".", "_") in MEVA_EXCLUDE:
                continue
            rows.append(dict(video=p.stem.replace(".", "_"), path=str(p), source="meva", label="none", g=-1,
                             scene_id=f"meva_{cam}", fold=-1))
    return rows


def free_runs(occ, T, fps, pad, min_len, banned=()):
    """occ: bool per sample (t = k / fps). Returns [(start, end)] of free time >= min_len."""
    t = np.arange(len(occ)) / fps
    bad = [(ti - pad, ti + pad) for ti in t[occ]] + list(banned)
    bad.sort()
    runs, cur = [], 0.0
    for a, b in bad:
        if a > cur:
            runs.append((cur, min(a, T)))
        cur = max(cur, b)
    if cur < T:
        runs.append((cur, T))
    return [(a, b) for a, b in runs if b - a >= min_len]


def main(a):
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    DET.mkdir(parents=True, exist_ok=True)
    vids = videos(a.src)
    todo = [v for v in vids if not (DET / f"{v['video']}.npz").exists()]
    print(len(vids), "videos,", len(todo), "to detect", flush=True)

    def load(v):
        fr, T, W, H, fps = sample(v["path"])
        return v, fr, T, W, H, fps

    def bounded(ex, items, ahead=4):  # at most `ahead` decoded videos in RAM (5-min MEVA clips are ~2.5 GB each)
        pending, it = set(), iter(items)
        for v in it:
            pending.add(ex.submit(load, v))
            if len(pending) >= ahead:
                done, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                yield from done
        yield from cf.as_completed(pending)

    with cf.ThreadPoolExecutor(4) as ex:
        for n, fut in enumerate(bounded(ex, todo)):
            v, fr, T, W, H, fps = fut.result()
            boxes = []
            for i in range(0, len(fr), 32):
                for j, r in enumerate(model.predict(fr[i:i + 32], imgsz=LONG, conf=CONF, classes=[0], verbose=False)):
                    for b, cf_ in zip(r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                        boxes.append([i + j, *b, cf_])
            boxes = np.array(boxes, np.float32).reshape(-1, 6)
            ly, ch = zip(*[look(fr[k:k + 1]) for k in range(len(fr))])
            np.savez_compressed(DET / f"{v['video']}.npz", boxes=boxes, T=T, W=W, H=H, fps=fps, n=len(fr),
                                sample_fps=FPS, luma=np.array(ly, np.float32), chroma=np.array(ch, np.float32))
            if n % 25 == 0:
                print(n, v["video"], len(fr), "samples", len(boxes), "boxes", flush=True)

    rows = []
    for v in vids:
        d = np.load(DET / f"{v['video']}.npz")
        occ = np.zeros(int(d["n"]), bool); occ[d["boxes"][:, 0].astype(int)] = True
        banned = [(v["g"] - 2, v["g"] + 10)] if v["label"] == "true" else []
        k = 0
        for s, e in free_runs(occ, float(d["T"]), FPS, PAD, a.min_len, banned):
            n_chunks = int(np.ceil((e - s) / MAX_LEN))
            for c in range(n_chunks):
                cs, ce = s + (e - s) * c / n_chunks, s + (e - s) * (c + 1) / n_chunks
                sel = slice(int(np.ceil(cs * FPS)), int(np.floor(ce * FPS)))
                rows.append(dict(video=v["video"], start=round(cs, 3), end=round(ce, 3), dur=round(ce - cs, 3),
                                 luma=round(float(d["luma"][sel].mean()), 1), chroma=round(float(d["chroma"][sel].mean()), 2),
                                 label=v["label"], plate_id=f"{v['video']}_p{k}", scene_id=v["scene_id"], fold=v["fold"],
                                 source=v["source"], path=v["path"], w=int(d["W"]), h=int(d["H"]), fps=round(float(d["fps"]), 3)))
                k += 1
    df = pd.DataFrame(rows)
    df.to_csv(SYN / "data/plates.csv", index=False)
    print(f"{len(df)} plates from {df.video.nunique()} videos, {df.scene_id.nunique()} scenes; "
          f"by source {df.groupby('source').size().to_dict()}; hours {df.dur.sum() / 3600:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min-len", type=float, default=5.0)
    p.add_argument("--src", nargs="+", default=["contest", "meva"])
    main(p.parse_args())
