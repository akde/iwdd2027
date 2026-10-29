"""Failure-case figure: one row per clip with four frames and the out-of-fold frame-logit curve of the submitted
recipe (run mixed_g0). The clip score q and the threshold tau are printed as text: they are on another scale than the
curve, so tau is not drawn on the logit axis. Blue: onset g and the accepted window [g-5, g+10]; red dashed: peak t*;
purple dotted: output p.
Usage (repository root, after failures_select.py): python camera_ready/failures_figure.py VIDEO:TITLE [VIDEO:TITLE ...]
e.g. vidXXXX:"(a) Detected". Writes camera_ready/output/failure_cases.pdf and .png."""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import OUT, ROOT
from frames import grab

plt.rcParams.update({"font.size": 7, "font.family": "serif"})
LG = dict(np.load(OUT / "frame_logits_mixed_g0.npz"))
C = pd.read_csv(OUT / "failures_categories_g0.csv").set_index("video")
ROWS = [tuple(x.split(":", 1)) for x in sys.argv[1:]]
assert ROWS, __doc__
fig = plt.figure(figsize=(7.0, 1.15 * len(ROWS)))
gs = fig.add_gridspec(len(ROWS), 6, width_ratios=[1, 1, 1, 1, 0.42, 2.3], wspace=0.05, hspace=0.45)
for i, (v, title) in enumerate(ROWS):
    r = C.loc[v]; T = r.dur
    c = r.t if r.y == 0 else r.g
    last = r.t if r.y == 1 and abs(r.t - c) > 2.5 else c + 4  # a positive's 4th frame: its peak when that lies apart
    times = [min(max(x, 0.0), T - 0.05) for x in ((c - 2, c, c + 2, last) if r.y == 1 else (c - 2, c - 1, c, c + 2))]
    for j, (t, im) in enumerate(zip(times, grab(v, times))):
        ax = fig.add_subplot(gs[i, j]); ax.imshow(im, aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"{t:.1f} s", labelpad=1)
        if j == 0:
            ax.set_ylabel(title, rotation=0, ha="left", va="bottom", labelpad=0)
            ax.yaxis.set_label_coords(0, 1.05)
    ax = fig.add_subplot(gs[i, 5])
    ax.plot(np.load(ROOT / "features_movinet" / f"{v}.npz")["t"], LG[v], "k-", lw=1)
    if r.y == 1:
        ax.axvspan(r.g - 5, min(r.g + 10, T), color="tab:blue", alpha=0.12, lw=0)
        ax.axvline(r.g, color="tab:blue", lw=1)
    ax.axvline(r.t, color="tab:red", ls="--", lw=1)
    if r.q >= r.tau:
        ax.axvline(r.p_learned, color="tab:purple", ls=":", lw=1.2)
    ax.set_xlim(0, T); ax.tick_params(pad=1, length=2); ax.set_ylabel("frame logit", labelpad=1)
    if i == len(ROWS) - 1:
        ax.set_xlabel("time in the clip (s)", labelpad=1)
    dec = "emit" if r.q >= r.tau else "no output"
    gtxt = f"g = {int(r.g)} s, " if r.y == 1 else "negative, "
    ax.set_title(f"{v}: {gtxt}q = {r.q:.2f} {'>=' if r.q >= r.tau else '<'} tau = {r.tau:.2f}: {dec}"
                 + (f", p = {int(r.p_learned)} s" if r.q >= r.tau else ""), fontsize=6.5, pad=2)
fig.savefig(OUT / "failure_cases.pdf", bbox_inches="tight")
fig.savefig(OUT / "failure_cases.png", bbox_inches="tight", dpi=150)
print("figure written")
