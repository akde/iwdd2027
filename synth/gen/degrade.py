"""Shared degradation (invariant 2): the same function and parameter law for real and synthetic clips.
Synthetic clips: gen/run_batch.py. Real clips (the detector's training loader): degrade_file(real.mp4, out.mp4,
sample_params(rng, w, h, fps)), or degrade_frames() on decoded frames + PTS. Never apply it to synthetic clips only.

sample_params(rng, w, h, fps) -> dict(height, fps, crf, noise, drop)
  height: a real clip's height (MIVIA-IWDD-500 law), never above the source (no upscaling); width keeps the aspect.
  fps:    a real clip's frame rate, never above the source.
  crf:    H.264 CRF uniform in [28, 42].   noise: 50% Gaussian sigma U(1, 5) (8-bit units), else 0.
  drop:   30% drop a fraction U(0.01, 0.05) of frames at random (timestamps kept: variable frame rate), else 0.
degrade_frames(frames, times, p, rng) -> frames, times (RGB uint8 lists; times = PTS seconds from 0)
encode(frames, times, path, crf, fps)  H.264 yuv420p, PTS from times (time base 1/90000).
Times are PTS throughout (invariant 5), never frame_index / fps."""
import json, pathlib
from fractions import Fraction
import av, cv2, numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
_LAW = None


def law():
    """(height, fps) of the 500 contest clips (label metadata)."""
    global _LAW
    if _LAW is None:
        hs, fs = [], []
        for p in sorted((ROOT / "data").glob("*/labels/*.json")):
            m = json.loads(p.read_text())["metadata"]
            hs.append(int(m["resolution"].split("x")[1])); fs.append(float(m["frame_rate"]))
        _LAW = np.array(hs), np.array(fs)
    return _LAW


def sample_params(rng, w, h, fps):
    hs, fs = law()
    th = int(min(h, rng.choice(hs))) // 2 * 2
    tw = int(round(w * th / h)) // 2 * 2
    return dict(width=tw, height=th, fps=float(min(fps, rng.choice(fs))), crf=int(rng.integers(28, 43)),
                noise=float(rng.uniform(1, 5)) if rng.random() < 0.5 else 0.0,
                drop=float(rng.uniform(0.01, 0.05)) if rng.random() < 0.3 else 0.0)


def degrade_frames(frames, times, p, rng):
    times = np.asarray(times, float)
    # frame rate: for each target grid time keep the last source frame at or before it (each used once)
    keep, last = [], -1
    for t in np.arange(0, times[-1] + 1e-9, 1.0 / p["fps"]):
        i = int(np.searchsorted(times, t + 1e-9) - 1)
        if i > last:
            keep.append(i); last = i
    if p["drop"] > 0 and len(keep) > 10:
        n = int(round(p["drop"] * len(keep)))
        gone = set(rng.choice(np.arange(1, len(keep) - 1), size=n, replace=False).tolist())
        keep = [k for j, k in enumerate(keep) if j not in gone]
    out = []
    for i in keep:
        f = frames[i]
        if (f.shape[1], f.shape[0]) != (p["width"], p["height"]):
            f = cv2.resize(f, (p["width"], p["height"]), interpolation=cv2.INTER_AREA)
        if p["noise"] > 0:
            f = np.clip(f.astype(np.float32) + rng.normal(0, p["noise"], f.shape), 0, 255).astype(np.uint8)
        out.append(f)
    return out, times[keep] - times[keep[0]]


def encode(frames, times, path, crf, fps):
    tb = Fraction(1, 90000)
    with av.open(str(path), "w") as c:
        st = c.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
        st.width, st.height, st.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
        st.codec_context.time_base = tb
        st.options = {"crf": str(crf), "preset": "medium"}
        last = -1
        for f, t in zip(frames, times):
            fr = av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")
            fr.pts = max(last + 1, int(round(t * 90000))); fr.time_base = tb; last = fr.pts
            for pkt in st.encode(fr):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)


def decode(path):
    """All frames (RGB) and PTS seconds from the first frame."""
    frames, times, p0 = [], [], None
    with av.open(str(path)) as c:
        s = c.streams.video[0]; s.thread_type = "AUTO"
        for f in c.decode(s):
            if f.pts is None:
                continue
            p0 = f.pts if p0 is None else p0
            frames.append(f.to_ndarray(format="rgb24")); times.append(float((f.pts - p0) * s.time_base))
    o = np.argsort(times, kind="stable")
    return [frames[i] for i in o], np.array(times)[o]


def degrade_file(src, dst, p, seed):
    """Decode src, degrade with params p, encode to dst. Returns label metadata of the result."""
    rng = np.random.default_rng(seed)
    frames, times = decode(src)
    out, t = degrade_frames(frames, times, p, rng)
    encode(out, t, dst, p["crf"], p["fps"])
    step = float(np.median(np.diff(t))) if len(t) > 1 else 1.0 / p["fps"]
    return dict(resolution=f"{p['width']}x{p['height']}", frame_rate=round(p["fps"], 3),
                duration=round(float(t[-1] + step), 3), n_frames=len(out))
