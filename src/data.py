"""Stage 1 data layer (wiki/baseline.md): PyAV decoding on a fixed time grid, and the clip index.

decode(path) -> frames uint8 [N,H,W,3] RGB, times [N] (s from the first frame's PTS), T (s).
index()      -> DataFrame video (key of data/folds.csv), split, y, g, T, g_ge_T for the 500 clips (cache/index.csv).
Build:  python src/data.py
"""
import json
from multiprocessing import Pool
from pathlib import Path

import av, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FPS, LONG = 8, 448
INDEX = ROOT / "cache" / "index.csv"


def _rgb(f, long_side):  # long side -> long_side, keep aspect, no crop
    k = long_side / max(f.width, f.height)
    return f.reformat(width=round(f.width * k), height=round(f.height * k), format="rgb24",
                      interpolation="AREA").to_ndarray()


def decode(path, fps=FPS, long_side=LONG):
    """Grid t = i/fps for t < T; each t takes the last frame with pts <= t (times from PTS).
    T = last pts + median PTS step. Not frame.duration: in B-frame MP4s it is the decode-order
    packet duration (vid0360: +0.12 s vs ffprobe)."""
    out, ts, prev, p0 = [], [], None, None

    def fill(until):  # grid points before `until` get prev, converted once
        img = None
        while len(out) / fps < until:
            img = _rgb(prev, long_side) if img is None else img
            out.append(img)

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for f in c.decode(s):
            if f.pts is None or (prev is not None and f.pts <= prev.pts):
                continue
            p0 = f.pts if p0 is None else p0
            t = float((f.pts - p0) * s.time_base)
            if prev is not None:
                fill(t)
            prev = f
            ts.append(t)
        if prev is None:
            raise ValueError(f"no decodable frames: {path}")
        T = ts[-1] + (float(np.median(np.diff(ts))) if len(ts) > 1 else float(prev.duration * s.time_base))
    fill(T)
    return np.stack(out), np.arange(len(out)) / fps, T


def _T(path):
    return decode(path)[2]


def index(rebuild=False):
    if rebuild or not INDEX.exists():
        rows = []
        for p in sorted((ROOT / "data").glob("*/labels/*.json")):
            d = json.loads(p.read_text())
            if d["label"] not in ("true", "false"):
                raise ValueError(f"{p}: label {d['label']!r}")
            rows.append(dict(video=p.stem, split=p.parent.parent.name, y=int(d["label"] == "true"),
                             g=int(d["timestamp"])))
        with Pool(16) as pool:
            T = pool.map(_T, [ROOT / "data" / r["split"] / "videos" / f"{r['video']}.mp4" for r in rows], chunksize=1)
        df = pd.DataFrame(rows).assign(T=T)
        INDEX.parent.mkdir(exist_ok=True)
        df.assign(g_ge_T=(df.y == 1) & (df.g >= df["T"])).to_csv(INDEX, index=False)  # label noise: onset >= end
    return pd.read_csv(INDEX, float_precision="round_trip")  # T bit-identical to decode()


if __name__ == "__main__":
    d = index(rebuild=True)
    print(f"{INDEX}: {len(d)} clips, splits {d.split.value_counts().to_dict()}, y {d.y.value_counts().to_dict()}, "
          f"T {d['T'].min():.2f}-{d['T'].max():.2f} s")
    print("positives with g >= T:\n" + d[d.g_ge_T][["video", "split", "g", "T"]].to_string(index=False))
