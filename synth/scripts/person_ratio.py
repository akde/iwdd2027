"""Luminance of real people relative to the plate around them (linear light), for the compositor's gain.
python synth/scripts/person_ratio.py -> synth/data/person_ratio.npy (all clips), person_ratio_night.npy (clips labelled
night or low lighting): one ratio per sampled detection
Up to 3 confident (>= 0.6), untruncated YOLO11s person boxes per contest clip; inner box (shrunk) vs a ring of the
same size around it."""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from contact_sheet import frames_at

SYN = pathlib.Path(__file__).resolve().parents[1]


def lin(x):
    x = x.astype(np.float32) / 255
    y = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return 0.2126 * y[..., 0] + 0.7152 * y[..., 1] + 0.0722 * y[..., 2]


rng = np.random.default_rng(0)
idx = pd.read_csv(SYN.parent / "cache/index.csv")
meta = pd.read_csv(SYN / "data/taxonomy/clips.csv").set_index("video")
out, night = [], []
for r in idx.itertuples():
    d = np.load(SYN / f"data/det/{r.video}.npz")
    b = d["boxes"]
    b = b[(b[:, 5] >= 0.6) & (b[:, 2] > 0.02) & (b[:, 4] < 0.98) & ((b[:, 4] - b[:, 2]) > 0.08)]
    if not len(b):
        continue
    b = b[rng.choice(len(b), min(3, len(b)), replace=False)]
    path = SYN.parent / f"data/{r.split}/videos/{r.video}.mp4"
    fr = frames_at(path, list(b[:, 0] / float(d["sample_fps"]) + 1e-3), 640)
    for (t, img), bb in zip(fr, sorted(b, key=lambda x: x[0])):
        H, W = img.shape[:2]
        x0, y0, x1, y1 = bb[1] * W, bb[2] * H, bb[3] * W, bb[4] * H
        w, h = x1 - x0, y1 - y0
        L = lin(img)
        inner = L[int(y0 + 0.1 * h):int(y1 - 0.1 * h), int(x0 + 0.25 * w):int(x1 - 0.25 * w)]
        m = np.zeros_like(L, bool)
        m[max(0, int(y0 - h / 2)):int(y1 + h / 2), max(0, int(x0 - w)):int(x1 + w)] = True
        m[int(y0):int(y1), int(x0):int(x1)] = False
        if inner.size > 20 and m.sum() > 50:
            out.append(inner.mean() / max(L[m].mean(), 1e-4))
            if meta.loc[r.video, "time_of_day"] == "night" or meta.loc[r.video, "lighting"] == "low":
                night.append(out[-1])
for name, a in (("person_ratio", out), ("person_ratio_night", night)):
    a = np.sort(np.array(a)); np.save(SYN / f"data/{name}.npy", a)  # sorted: the compositor draws from the pool
    print(name, len(a), "ratios; quantiles 10/25/50/75/90:", np.percentile(a, [10, 25, 50, 75, 90]).round(2))
