"""Run-to-run check (GPU): retrain fold 0, seed 0 of a finished cross-validation run with identical settings and compare
its fold-0 out-of-fold q and t with the run's oof_seed0.csv. cuDNN deterministic mode is on, but other GPU kernels are
not bit-reproducible; in our runs an identical retraining changed clip scores by 0.033 on average (max 0.14) and moved
the peak in 24 of 98 clips. Set --robust/--extra/--pad-real as the run was trained.
Usage (repository root): python camera_ready/parity.py --run realorig_g0 [--extra <index.csv>] [--pad-real]"""
import argparse, json
import pandas as pd, torch
from common import OUT, ROOT
import finetune as ft
from train import rows

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="realorig_g0")
ap.add_argument("--extra", nargs="*", default=[])
ap.add_argument("--pad-real", action="store_true")
a = ap.parse_args()
ft.P_LOWQ, ft.P_ASPECT = 0.5, 0.3  # --robust, as every run here
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
e = ft.E3("m", a.extra)
R = e.R
tr, te = [v for v in R if R[v].fold != 0], [v for v in R if R[v].fold == 0]
tail, head, _ = e.fit(tr, 0, stop=5, tag="parity fold 0 seed 0", extra=e.syn(0), pad=a.pad_real)
lg = e.logits(tail, head, te)
new = pd.DataFrame(rows(R, {v: [lg[v]] for v in te}, {v: (None, e.t[v]) for v in R})).set_index("video")
old = pd.read_csv(ROOT / "runs" / a.run / "oof_seed0.csv").set_index("video").loc[new.index]
out = dict(run=a.run, clips=len(new), max_abs_dq=float((new.q - old.q).abs().max()),
           mean_abs_dq=float((new.q - old.q).abs().mean()), t_equal=int(((new.t - old.t).abs() < 1e-9).sum()))
print(json.dumps(out))
(OUT / f"parity_{a.run}.json").write_text(json.dumps(out, indent=1))
