"""Per-clip BUDGET/SEEDS records: for every arm, seed group and unit (the 3-seed
ensemble and each single seed), one row per clip with its cross-fitted (tau, delta), emission, final timestamp and event
outcome under the contest rule -> camera_ready/output/budget_predictions.csv.gz.
Run from the repository root: python camera_ready/budget_predictions.py"""
import numpy as np, pandas as pd
from common import OUT, load_oof, outcomes, xfit_points, decide, public_mask
from budget import RUNS, SEEDS

parts = []
for arm, gs in RUNS.items():
    for g, path in gs.items():
        for unit, f in [("ensemble", "oof.csv")] + [(f"seed {s}", f"oof_seed{s}.csv") for s in SEEDS[g]]:
            d = load_oof(path / f)
            pts = xfit_points(d)
            emit, p = decide(d, pts)
            o = outcomes(d, emit, p)
            out = np.where(d.y == 1, np.where(~emit, "FN", np.where(o[:, 0] == 1, "TP", "FP+FN")), np.where(emit, "FP", "TN"))
            parts.append(pd.DataFrame(dict(video=d.video, scene=d.scene, fold=d.fold, split=np.where(public_mask(d), "public_test", "train"),
                                           condition=arm, group=g, unit=unit, run=path.name, y=d.y, g=d.g, dur=d.dur, q=d.q, t=d.t,
                                           tau=[pts[k][0] for k in d.fold], delta=[pts[k][1] for k in d.fold], emit=emit,
                                           p=np.where(emit, p, np.nan), outcome=out)))
P = pd.concat(parts)
P.to_csv(OUT / "budget_predictions.csv.gz", index=False)
print(len(P), "rows;", P.groupby(["condition", "unit"]).size().shape[0], "condition x unit")
