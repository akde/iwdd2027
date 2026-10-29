"""Synthetic pixels drawn inside a plate's black pillarbox/letterbox bars (DECISIONS.md D29).
python synth/scripts/bar_check.py CLEAN.mp4 [CLEAN.mp4 ...] -> CSV lines clip,bar_l,bar_r,bar_t,bar_b,frames,max_px
Bars = edge columns/rows whose temporal-median luma (every 2nd frame) is < 12 (bars >= 4 px; a 2 px margin is kept);
content = luma > 50 inside a bar; frames = frames with >= 30 such pixels, max_px = the most in one frame.
Rule (D29): max_px >= 200 (an avatar or object drawn on black, outside the camera image) -> exclude the pair."""
import sys
import av, numpy as np


def run(v):
    k = 0
    while k < len(v) and v[k] < 12:
        k += 1
    return k


for p in sys.argv[1:]:
    with av.open(p) as c:
        s = c.streams.video[0]; s.thread_type = "AUTO"
        A = np.stack([f.to_ndarray(format="gray") for i, f in enumerate(c.decode(s)) if i % 2 == 0])
    med = np.median(A, 0); col, row = med.mean(0), med.mean(1)
    L, R, T, B = run(col), run(col[::-1]), run(row), run(row[::-1])
    H, W = med.shape; m = np.zeros((H, W), bool)
    if L >= 4: m[:, :L - 2] = True
    if R >= 4: m[:, W - R + 2:] = True
    if T >= 4: m[:T - 2] = True
    if B >= 4: m[H - B + 2:] = True
    cnt = (A[:, m] > 50).sum(1) if m.any() else np.zeros(len(A), int)
    print(f"{p.rsplit('/', 1)[-1][:-4]},{L},{R},{T},{B},{int((cnt >= 30).sum())},{int(cnt.max())}")
