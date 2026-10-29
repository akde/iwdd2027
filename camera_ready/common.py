"""Shared helpers for the camera-ready analyses. Scoring goes through tools/iwdd_eval.py (counts, tune) and
src/decision.py (to_output). Scene bootstrap: scene draws from numpy default_rng(SEED + 1), 5000 resamples, percentile
2.5/97.5, per-scene TP/FP/FN sums. F1 = 2TP / (2TP + FP + FN); a zero denominator (no positives and no emissions in a
resample) gives F1 = 0, as in the scorer, and such resamples are counted (n_zero_denominator).
Outputs go to camera_ready/output/; they hold per-clip labels and stay out of version control."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "camera_ready" / "output"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "src"))
import iwdd_eval as E
from decision import to_output

KW = dict(early=5, fpfn=True, integer=True)  # contest rule e5_fpfn, integer output as test.py writes it
TAU_SUB, DELTA_SUB = 0.363, -1.0
NBOOT, SEED = 5000, 0
FOLDS = pd.read_csv(ROOT / "data/folds.csv").set_index("video")
SYNTH_INDEX = ROOT / "cache/frames5_synth/release_v3/index.csv"  # `python src/finetune.py cache --release <release_v3 dir>`

# Out-of-fold runs (runs/<id>/), trained with the commands in camera_ready/README.md. Seed group gK = seeds 3K..3K+2.
SEEDS = {f"g{k}": tuple(range(3 * k, 3 * k + 3)) for k in range(5)}
RUN = {"MIXED_MATCHED": {f"g{k}": f"mixed_g{k}" for k in range(5)},      # the submitted recipe (real + synthetic pairs)
       "REAL_EXTENDED": {f"g{k}": f"realext_g{k}" for k in range(3)},    # real only, --pad-real
       "REAL_ORIGINAL": {f"g{k}": f"realorig_g{k}" for k in range(3)}}   # real only
run_dir = lambda arm, g: ROOT / "runs" / RUN[arm][g]


def load_oof(path):
    d = pd.read_csv(path).sort_values("video").reset_index(drop=True)
    assert len(d) == 500 and (d.scene.values == FOLDS.loc[d.video, "scene"].values).all(), path
    assert (d.fold.values == FOLDS.loc[d.video, "fold"].values).all(), path
    return d


def outcomes(d, emit, p):  # per clip [TP, FP, FN]; a mistimed positive is FP + FN
    pos = d.y.values == 1
    hit = (p >= d.g.values - 5) & (p <= d.g.values + 10)
    tp = emit & pos & hit
    mis = emit & pos & ~hit
    return np.stack([tp, (emit & ~pos) | mis, (pos & ~emit) | mis], 1).astype(float)


def xfit_points(d):  # fold -> (tau, delta) tuned on the other four folds (iwdd_eval.tune, contest rule, --int)
    return {int(k): E.tune(d[d.fold != k], **KW)[1:] for k in sorted(d.fold.unique())}


def decide(d, pts):  # per clip emit mask and integer output at per-fold (tau, delta)
    emit, p = np.zeros(len(d), bool), np.zeros(len(d))
    for k, (tau, delta) in pts.items():
        m = d.fold.values == k
        emit[m] = d.q.values[m] >= tau
        p[m] = to_output(d.t.values[m], delta, d.dur.values[m])
    return emit, p


def f1(c):
    tp, fp, fn = c
    return 2 * tp / max(2 * tp + fp + fn, 1)


class SceneBoot:  # paired scene bootstrap over the clips of a set
    def __init__(self, d, mask=None):
        mask = np.ones(len(d), bool) if mask is None else mask
        idx = np.flatnonzero(mask)
        self.scenes = sorted(set(d.scene.values[idx]))
        self.members = [idx[d.scene.values[idx] == s] for s in self.scenes]
        self.draw = np.random.default_rng(SEED + 1).integers(0, len(self.scenes), (NBOOT, len(self.scenes)))

    def ci(self, a, b=None):  # a, b: per clip [TP, FP, FN] (all 500 rows) -> point, [lo, hi], zero-denominator count
        sa = np.array([a[m].sum(0) for m in self.members])
        sb = np.array([b[m].sum(0) for m in self.members]) if b is not None else None
        v, zero = [], 0
        for i in self.draw:
            ca = sa[i].sum(0)
            zero += (2 * ca[0] + ca[1] + ca[2]) == 0
            v.append(f1(ca) - (f1(sb[i].sum(0)) if sb is not None else 0.0))
        pt = f1(sa.sum(0)) - (f1(sb.sum(0)) if sb is not None else 0.0)
        return round(float(pt), 4), [round(float(x), 4) for x in np.percentile(v, [2.5, 97.5])], int(zero)


def public_mask(d):  # the 100 official public-test clips (cache/index.csv split, written by src/data.py)
    lab = pd.read_csv(ROOT / "cache/index.csv").set_index("video")
    m = lab.loc[d.video, "split"].values == "public_test"
    assert m.sum() == 100
    return m
