"""Frame grabbing by presentation timestamp (PyAV), read-only on data/. Shared by the failure sheets and figure."""
from pathlib import Path
import av, numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"


def video_path(v):
    for s in ("train", "public_test"):
        for p in (DATA / s / "videos").glob(f"{v}.*"):
            return p
    raise FileNotFoundError(v)


def grab(v, times):  # -> list of RGB arrays, the last frame with PTS <= t for each t (times in s)
    with av.open(str(video_path(v))) as c:
        s = c.streams.video[0]; tb = float(s.time_base); fr = []
        for f in c.decode(s):
            if f.pts is None:
                continue
            fr.append((f.pts * tb, f.to_ndarray(format="rgb24")))
            if fr[-1][0] > max(times) + 0.5:
                break
    ts = np.array([t for t, _ in fr])
    return [fr[max(int(np.searchsorted(ts, t, side="right")) - 1, 0)][1] for t in times]
