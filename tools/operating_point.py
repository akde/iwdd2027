#!/usr/bin/env python3
"""Stage 6 operating point (wiki/baseline.md) from a run's OOF predictions, scored as test.py writes them (--int).
DELTA = the delta tuned on all clips; TAU = the middle of the flat F1 region at DELTA (F1 within 0.01 of its max,
contiguous around the max; the selection protocol). Also prints each fold's cross-fitted (tau, delta).
Usage: python tools/operating_point.py runs/s5_cls_inner/oof.csv [--variant e5_fpfn]  (default official; the contest
rule is e5_fpfn, wiki/decisions.md 2026-10-08)
"""
import argparse
import numpy as np, pandas as pd
import iwdd_eval as E

ap = argparse.ArgumentParser()
ap.add_argument("csv"); ap.add_argument("--variant", choices=list(E.VARIANTS), default="official")
args = ap.parse_args()
early, fpfn = E.VARIANTS[args.variant]
kw = dict(early=early, fpfn=fpfn, integer=True)
d = pd.read_csv(args.csv).sort_values("video").reset_index(drop=True)
for k in sorted(d.fold.unique()):
    f, a, b = E.tune(d[d.fold != k], **kw)
    print(f"fold {k} held out: tau {a:.3f}, delta {b}, F1 on the other folds {f:.3f}")
F, tau_max, DELTA = E.tune(d, **kw)
taus = np.unique(np.quantile(d.q.values, np.linspace(0, 1, 101)))  # the scorer's tau grid
f1 = np.array([E.f1(E.counts(d, a, DELTA, **kw)) for a in taus])
i = j = int(np.argmax(f1))
while i > 0 and f1[i - 1] >= f1.max() - 0.01:
    i -= 1
while j < len(taus) - 1 and f1[j + 1] >= f1.max() - 0.01:
    j += 1
TAU = (taus[i] + taus[j]) / 2
print(f"all {len(d)} clips: max F1 {F:.3f} at tau {tau_max:.3f}, delta {DELTA}")
print(f"flat region at delta {DELTA}: tau {taus[i]:.3f}-{taus[j]:.3f} (width {taus[j] - taus[i]:.3f}; "
      f"q quantiles {i}-{j} of 100)")
print(f"TAU = {TAU:.3f}, DELTA = {DELTA}: F1 {E.f1(E.counts(d, TAU, DELTA, **kw)):.3f}, "
      f"emits on {np.mean(d.q >= TAU):.2f} of clips")
