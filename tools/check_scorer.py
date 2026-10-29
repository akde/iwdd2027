#!/usr/bin/env python3
"""Synthetic checks for tools/iwdd_eval.py. Rerun: python tools/check_scorer.py  (exit 1 on any failure).
Checks the rule as written; it never changes it."""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import iwdd_eval as E

fails = []
def check(name, got, want):
    ok = np.allclose(got, want)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got}, want {want}")
    if not ok:
        fails.append(name)

def make(n_pos=50, n_neg=50, seed=0):
    r = np.random.default_rng(seed)
    g = r.integers(0, 20, n_pos).astype(float)
    return pd.DataFrame({"video": [f"v{i:03d}" for i in range(n_pos + n_neg)],
                         "fold": np.arange(n_pos + n_neg) % 5,
                         "y": [1] * n_pos + [0] * n_neg,
                         "g": np.r_[g, -np.ones(n_neg)],
                         "dur": np.r_[g + 30, np.full(n_neg, 40.0)],
                         "q": np.r_[np.ones(n_pos), np.zeros(n_neg)],
                         "t": np.r_[g, np.zeros(n_neg)]})

d = make()
for integer in (False, True):
    s = " int" if integer else ""
    # 1. perfect predictions -> F1 = 1 (counts at tau=0.5, delta=0; tuned; cross-fitted; all variants)
    check(f"perfect counts{s}", E.counts(d, 0.5, 0.0, integer=integer), [50, 0, 0])
    check(f"perfect tuned F1{s}", E.tune(d, integer=integer)[0], 1.0)
    for k, (e, f) in E.VARIANTS.items():
        check(f"perfect xfit {k}{s}", E.xfit(d, early=e, fpfn=f, integer=integer), 1.0)
    # 2. never emitting -> F1 = 0 (TP 0, FP 0, FN = all positives)
    check(f"never emit counts{s}", E.counts(d, np.inf, 0.0, integer=integer), [0, 0, 50])
    check(f"never emit F1{s}", E.f1(E.counts(d, np.inf, 0.0, integer=integer)), 0.0)

# 3. window edges (float mode = exact rule). One positive, g = 10, dur = 40.
one = lambda t, g=10.0, dur=40.0: pd.DataFrame({"video": ["a"], "fold": [0], "y": [1], "g": [g], "dur": [dur],
                                                "q": [1.0], "t": [t]})
for off, hit in [(-3, 1), (10, 1), (-3.5, 0), (10.5, 0), (0, 1), (-3.01, 0), (10.01, 0)]:
    check(f"edge p=g{off:+} float", E.counts(one(10 + off), 0.5, 0.0)[0], hit)

# 4. mistimed positive: official = FP only; *_fpfn = FP + FN
mis = one(30.0)
check("mistimed official [TP,FP,FN]", E.counts(mis, 0.5, 0.0, 3, False), [0, 1, 0])
check("mistimed fpfn [TP,FP,FN]", E.counts(mis, 0.5, 0.0, 3, True), [0, 1, 1])
check("early 5: p=g-5 hit", E.counts(one(5.0), 0.5, 0.0, 5, False)[0], 1)
check("early 3: p=g-5 miss", E.counts(one(5.0), 0.5, 0.0, 3, False)[0], 0)
# 5. clamp p <= dur: t + delta beyond the clip end is pulled back into the window
check("clamp: t=50, dur=12, g=10 -> p=12 hit", E.counts(one(50.0, 10, 12.0), 0.5, 0.0)[0], 1)
# 6. negatives: emit = FP
neg = pd.DataFrame({"video": ["n"], "fold": [0], "y": [0], "g": [-1.0], "dur": [20.0], "q": [1.0], "t": [5.0]})
check("negative emitted [TP,FP,FN]", E.counts(neg, 0.5, 0.0), [0, 1, 0])

# 7. --int scores what test.py writes: p = floor(min(t + delta, dur)), never past the clip end
for off, hit in [(-3.5, 0), (-2.5, 1), (10.5, 1), (10.99, 1), (11.0, 0)]:
    check(f"int p=g{off:+} floored", E.counts(one(10 + off), 0.5, 0.0, integer=True)[0], hit)
check("int clamp: t=50, dur=9.6 -> p=9", E.pred(one(50.0, 13.0, 9.6), 0.0, True), [9.0])
check("int clamp: g=13, p=9 -> miss", E.counts(one(50.0, 13.0, 9.6), 0.5, 0.0, integer=True)[0], 0)

# 8. --int = src/decision.py to_output = max(0, min(floor(t + delta), floor(dur))) (stage 6)
from decision import to_output
check("int clamp at 0: t=0, delta=-2, g=3 -> p=0 hit", E.counts(one(0.0, 3.0), 0.5, -2.0, integer=True)[0], 1)
check("to_output scalar: t=5.7, delta=2.5, T=7.9 -> 7", to_output(5.7, 2.5, 7.9), 7)
check("to_output scalar: t=0.2, delta=-1, T=10 -> 0", to_output(0.2, -1.0, 10.0), 0)
r = np.random.default_rng(1)
tt, TT = r.uniform(-5, 120, 10000), r.uniform(2, 110, 10000)
check("to_output = max(0, floor(min(t + delta, T))) on 10000 random", np.array_equal(
    to_output(tt, -1.5, TT), np.maximum(0, np.floor(np.minimum(tt - 1.5, TT)))), True)

# 9. within-scene AUROC uses same-scene (pos, neg) pairs only
ws = lambda q, y, sc: E.within_scene(pd.DataFrame({"q": q, "y": y, "scene": sc}), n=200)
check("within-scene perfect", ws([.9, .8, .1, .2], [1, 1, 0, 0], ["a", "b", "a", "b"])["auroc"], 1.0)
check("within-scene reversed", ws([.1, .2, .9, .8], [1, 1, 0, 0], ["a", "b", "a", "b"])["auroc"], 0.0)
w = ws([.9, .1, .2, .8], [1, 0, 1, 0], ["a", "a", "b", "b"])  # pooled AUROC would be 0.75
check("within-scene ignores cross-scene pairs", [w["auroc"], w["pairs"]], [0.5, 2])
check("within-scene: single-class scenes give None", ws([.9, .1], [1, 0], ["a", "b"]) is None, True)

# 10. G1b removes ln T: a score that knows only the clip length gets exactly 0.5; a T-free perfect score stays 1
r = np.random.default_rng(1)
sc = np.repeat([f"s{i}" for i in range(10)], 6)
yy, TT = np.tile([1, 1, 1, 0, 0, 0], 10), r.uniform(5, 60, 60)
L = pd.DataFrame({"y": yy, "dur": TT, "scene": sc, "q": 1 / (1 + TT / 20)})
check("G1b: duration-only score -> 0.5", E.within_scene_len(L, n=10)["auroc"], 0.5)
check("G1b: T-free perfect score (negatives constant) -> 1", E.within_scene_len(L.assign(q=np.where(yy == 1, .9, .2)), n=10)["auroc"], 1.0)
check("perm p: perfect within-scene score is significant", E.perm_p(L.assign(q=np.where(yy == 1, .9, .2))) < 0.01, True)
check("perm p: duration-only score after G1b is not", E.perm_p(E.within_len_q(L)) > 0.5, True)
# 11. G3: peak in [g - 1, g + 3] on positives with g + 0.5 < dur, against the best constant time
P = pd.DataFrame({"y": [1] * 4 + [0], "g": [5.0, 5, 5, 20, -1], "dur": [30.0, 30, 30, 20.2, 30], "t": [5.0, 8, 9, 0, 0],
                  "scene": list("aabbc")})
h = E.timing_hit(P, n=10)
check("G3: hits [g-1, g+3], g + 0.5 >= dur excluded", [h["hit"], h["n"]], [0.667, 3])  # reported to 3 decimals
check("G3: best constant (first argmax) and its hit", [h["best_constant_t"], h["best_constant_hit"]], [4.0, 1.0])

# 12. --variant pairing (2026-10-08): each variant tunes and bootstraps with its own rule; a run against itself is 0
m = make(seed=3)
m = m.assign(q=np.clip(m.q + np.random.default_rng(4).normal(0, .4, len(m)), 0, 1),
             t=m.t + np.random.default_rng(5).normal(0, 6, len(m)))
for k, (e, f) in E.VARIANTS.items():
    _, a, b = E.tune(m, early=e, fpfn=f, integer=True)
    check(f"paired self {k}", E.boot([(m, a, b), (m, a, b)], n=50, early=e, fpfn=f, integer=True), [0, 0, 0])
check("e5_fpfn F1 <= e5_fp F1 at the same settings (mistimed costs twice)",
      E.f1(E.counts(m, .5, 0, 5, True, True)) <= E.f1(E.counts(m, .5, 0, 5, False, True)), True)

print(f"\n{len(fails)} failure(s)" + (": " + ", ".join(fails) if fails else ""))
sys.exit(1 if fails else 0)
