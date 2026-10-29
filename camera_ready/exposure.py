"""BUDGET exposure table: per arm, seed group, fold and seed: optimiser updates, real and synthetic crop presentations,
unique real clips, unique synthetic pairs drawn over the 5 epochs (re-drawn exactly with the training code's pair
generator, numpy default_rng([seed, 1])), and training wall time (sum of the epoch times in the run's train.log).
Equal updates do not mean equal real exposure: REAL_EXTENDED shows the real clips about twice per epoch.
Run from the repository root: python camera_ready/exposure.py"""
import json, re
import numpy as np, pandas as pd
pd.set_option("display.width", 200)
from common import OUT, ROOT, FOLDS, SYNTH_INDEX
from budget import RUNS, SEEDS

idx = pd.read_csv(SYNTH_INDEX)
rows = []
for arm, gs in RUNS.items():
    for g, path in gs.items():
        log = path / "train.log"
        secs = {}
        if log.exists():
            for m in re.finditer(r"fold (\d) seed (\d+) ep \d+ .* (\d+) s$", log.read_text(), re.M):
                secs[int(m[1]), int(m[2])] = secs.get((int(m[1]), int(m[2])), 0) + int(m[3])
        for k in range(5):
            n = int((FOLDS.fold != k).sum())
            ids = sorted(idx[idx.fold != k].pair.unique())
            npair = min(len(ids), n // 2) if arm == "MIXED_MATCHED" else 0
            npad = 2 * (n // 2) if arm == "REAL_EXTENDED" else 0
            pool = n + 2 * npair + npad
            for s in SEEDS[g]:
                uniq = None
                if npair:
                    srng = np.random.default_rng([s, 1])
                    uniq = len(set().union(*[set(srng.choice(ids, npair, replace=False)) for _ in range(5)]))
                rows.append(dict(arm=arm, group=g, fold=k, seed=s, run=path.name, updates=5 * -(-pool // 16),
                                 pool_per_epoch=pool, real_presentations=5 * (n + npad), synthetic_presentations=5 * 2 * npair,
                                 synthetic_share=round(2 * npair / pool, 3), unique_real_clips=n, unique_pairs_drawn=uniq,
                                 pairs_available=len(ids) if arm == "MIXED_MATCHED" else 0,
                                 lr_schedule=f"AdamW, cosine T_max = {5 * -(-pool // 16)} updates, stepped per update",
                                 checkpoint="final update (end of epoch 5)", train_seconds=secs.get((k, s)),
                                 status="done" if (path / "oof.csv").exists() else "missing"))
X = pd.DataFrame(rows)
X.to_csv(OUT / "exposure.csv", index=False)
print(X.groupby("arm")[["updates", "real_presentations", "synthetic_presentations", "unique_pairs_drawn", "train_seconds"]].mean().round(1))
