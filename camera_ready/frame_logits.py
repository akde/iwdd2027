"""Out-of-fold frame logits of one cross-validation run, rebuilt from its fold models (GPU): for every clip, the
frame logits of the clip's held-out fold models (one per seed) are computed with the training code's own inference
(src/finetune.py: full clip, one window per forward) and averaged over seeds. The result must reproduce the run's
oof.csv q and t (checked). Writes camera_ready/output/frame_logits_<run>.npz: {video: logits on the clip's 5 Hz grid};
the grid is features_movinet/<video>.npz["t"].
Usage (repository root): python camera_ready/frame_logits.py [--run mixed_g0]"""
import argparse, json
import numpy as np, pandas as pd, torch
from common import OUT, ROOT
import finetune as ft
from head import score

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="mixed_g0")
a = ap.parse_args()
run = ROOT / "runs" / a.run
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
e = ft.E3("m")
seeds = json.loads((run / "config.json").read_text())["seeds"]
L = {}
for k in sorted({r.fold for r in e.R.values()}):
    te = [v for v in e.R if e.R[v].fold == k]
    per = {v: [] for v in te}
    for s in seeds:
        tail, head = ft.load_ckpt(e, run / f"f{k}_s{s}.pt")
        for v, l in e.logits(tail, head, te).items():
            per[v].append(l)
    L.update({v: np.mean(ls, 0) for v, ls in per.items()})
o = pd.read_csv(run / "oof.csv").set_index("video")
dq = max(abs(score(L[v], e.t[v])[0] - o.q[v]) for v in L)
dt = max(abs(score(L[v], e.t[v])[1] - o.t[v]) for v in L)
print(f"{a.run}: {len(L)} clips; max |q - oof q| {dq:.1e}, max |t - oof t| {dt:.1e}")
np.savez_compressed(OUT / f"frame_logits_{a.run}.npz", **L)
