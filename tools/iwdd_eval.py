#!/usr/bin/env python3
"""IWDD-2027 scorer. The single source of truth for every reported metric.

Input CSV, one row per video (out-of-fold predictions):
  video, fold, y (0/1), g (onset s, ignored if y=0), dur (true duration s, from PTS),
  q (video score, higher = more likely positive), t (raw event time s, before offset)
  optional: night, dim, event_type, scene  -> subgroup report
Contest rule: early 5 s, mistimed = FP + FN (variant e5_fpfn). The key 'official' below is the website's wording
(3 s, FP only); it stays the default key (--variant selects).
Official rule (mivia.unisa.it/iwddcontest2027, read 2026-10-03):
  TP: positive video with g-3 <= p <= g+10.  FP: detection in a negative video, or a
  mistimed detection in a positive video.  FN: positive video with no detection.
--int scores the integer test.py writes: p = src/decision.py to_output(t, delta, dur)
  = max(0, min(floor(t + delta), floor(dur))), so 0 <= p <= dur (decided 2026-10-03, max(0) since stage 6).
With a scene column the report adds within-scene AUROC (same-scene positive/negative pairs, scene-bootstrap CI).
Usage: python tools/iwdd_eval.py runs/X/oof.csv [--vs runs/Y/oof.csv [--variant e5_fpfn]] [--int]
--variant (2026-10-08): the --vs pairing tunes both runs and bootstraps under that variant (default official).
"""
import argparse, json, pathlib, sys
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from decision import to_output

VARIANTS = {"official": (3, False), "e3_fpfn": (3, True), "e5_fp": (5, False), "e5_fpfn": (5, True)}
DELTAS = np.arange(-2.0, 6.01, 0.5)


def pred(d, delta, integer=False):  # p = min(t + delta, dur); --int: what test.py writes (src/decision.py)
    return to_output(d.t.values, delta, d.dur.values) if integer else np.minimum(d.t.values + delta, d.dur.values)


def counts(d, tau, delta, early=3, fpfn=False, integer=False):
    emit, pos = d.q.values >= tau, d.y.values == 1
    p = pred(d, delta, integer)
    hit = (p >= d.g.values - early) & (p <= d.g.values + 10)
    mis = np.sum(emit & pos & ~hit)
    return np.array([np.sum(emit & pos & hit), np.sum(emit & ~pos) + mis, np.sum(pos & ~emit) + mis * fpfn])


def f1(c):
    return 2 * c[0] / max(2 * c[0] + c[1] + c[2], 1)


def tune(d, **kw):  # -> (f1, tau, delta); optimistic, use only to pick settings
    taus = np.append(np.unique(np.quantile(d.q.values, np.linspace(0, 1, 101))), np.inf)
    return max(((f1(counts(d, a, b, **kw)), a, b) for a in taus for b in DELTAS), key=lambda r: r[0])


def xfit(d, **kw):  # tune on the other folds, apply to the held-out fold: honest estimate
    c = np.zeros(3)
    for k in d.fold.unique():
        _, a, b = tune(d[d.fold != k], **kw)
        c += counts(d[d.fold == k], a, b, **kw)
    return f1(c)


def blind(d, early=3, fpfn=False, integer=False):  # always emit at one constant time
    b = d.assign(q=1.0, t=0.0)
    return max((f1(counts(b, 0.5, c, early, fpfn, integer)), c) for c in np.arange(0, 30.01, 0.5))


def auroc(y, s):
    r, n1 = pd.Series(s).rank().values, y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def within_scene(d, n=10000, seed=0):  # AUROC over (pos, neg) pairs of the same scene; 95% scene-bootstrap CI
    P, W = [], []
    for _, x in d.groupby("scene"):
        p, m = x.q[x.y == 1].values, x.q[x.y == 0].values
        if len(p) and len(m):
            P.append(len(p) * len(m))
            W.append((p[:, None] > m).sum() + 0.5 * (p[:, None] == m).sum())
    if not P:
        return None
    P, W, rng = np.array(P), np.array(W), np.random.default_rng(seed)
    b = [W[i].sum() / P[i].sum() for i in (rng.integers(0, len(P), len(P)) for _ in range(n))]
    return {"auroc": round(W.sum() / P.sum(), 3), "ci95": np.percentile(b, [2.5, 97.5]).round(3),
            "pairs": int(P.sum()), "scenes": len(P)}


def within_scene_len(d, **kw):  # G1b (wiki/plan.md 3): within-scene AUROC of logit q minus its ln T line over negatives
    return within_scene(within_len_q(d), **kw)


def perm_p(d, n=2000, seed=0):  # within-scene permutation p-value: q shuffled within each scene, P(null >= observed)
    S = [(x.q.values, x.y.values == 1) for _, x in d.groupby("scene") if 0 < x.y.sum() < len(x)]
    def ws(S):  # same statistic as within_scene's point estimate
        W = sum((q[y][:, None] > q[~y]).sum() + 0.5 * (q[y][:, None] == q[~y]).sum() for q, y in S)
        return W / sum(y.sum() * (~y).sum() for _, y in S)
    obs, rng = ws(S), np.random.default_rng(seed)
    null = np.array([ws([(rng.permutation(q), y) for q, y in S]) for _ in range(n)])
    return round((1 + np.sum(null >= obs - 1e-12)) / (n + 1), 4)


def timing_hit(d, lo=-1, hi=3, n=2000, seed=0):  # G3: raw peak t in [g + lo, g + hi] on positives with g + 0.5 < dur
    p = d[(d.y == 1) & (d.g + 0.5 < d.dur)]
    hit = lambda t: ((t >= p.g.values + lo) & (t <= p.g.values + hi)).astype(float)
    cs = np.arange(0, 21.0)
    ref = [hit(np.minimum(c, p.dur.values)).mean() for c in cs]
    c = cs[int(np.argmax(ref))]
    diff, rng = hit(p.t.values) - hit(np.minimum(c, p.dur.values)), np.random.default_rng(seed)
    grp = p.scene.values if "scene" in p else np.arange(len(p))  # scene bootstrap of the paired difference
    u = np.unique(grp)
    idx = [np.flatnonzero(grp == g) for g in u]
    b = [diff[np.concatenate([idx[i] for i in rng.integers(0, len(u), len(u))])].mean() for _ in range(n)]
    return {"hit": round(hit(p.t.values).mean(), 3), "n": len(p), "best_constant_t": c, "best_constant_hit": round(max(ref), 3),
            "diff_ci95": np.percentile(b, [2.5, 97.5]).round(3)}


def boot(runs, n=1000, seed=0, **kw):  # runs: [(d, tau, delta)]; 2 runs -> paired difference
    rng, y = np.random.default_rng(seed), runs[0][0].y.values
    ip, ineg, out = np.where(y == 1)[0], np.where(y == 0)[0], []
    for _ in range(n):
        i = np.concatenate([rng.choice(ip, len(ip)), rng.choice(ineg, len(ineg))])
        v = [f1(counts(d.iloc[i], a, b, **kw)) for d, a, b in runs]
        out.append(v[0] - v[1] if len(v) == 2 else v[0])
    return np.percentile(out, [2.5, 50, 97.5]).round(3)


def within_len_q(d):  # d with q replaced by the G1b residual (for the permutation p-value)
    q = np.clip(d.q.values, 1e-6, 1 - 1e-6)
    z, lt, neg = np.log(q / (1 - q)), np.log(d.dur.values), d.y.values == 0
    b, a = np.polyfit(lt[neg], z[neg], 1)
    return d.assign(q=np.round(z - a - b * lt, 6))


def ref_duration_only(d):  # reference row: a score that knows only clip length (q = 1/(1 + T/20), t = 9)
    r = d.assign(q=1 / (1 + d.dur / 20), t=9.0)
    return {"xfit_f1_official_int": round(xfit(r, integer=True), 3),
            "within_scene_auroc": within_scene(r, n=1)["auroc"] if "scene" in d else None}


def report(d, integer):
    kw = dict(integer=integer)
    F, tau, delta = tune(d, **kw)
    pos, emit = d.y.values == 1, d.q.values >= tau
    p = pred(d, delta, integer)
    hit = (p >= d.g.values - 3) & (p <= d.g.values + 10)
    tp, fp, fn = counts(d, tau, delta, **kw)
    floor, cstar = blind(d, integer=integer)
    r = {
        "n": len(d), "n_pos": int(pos.sum()),
        "bad_onsets_g_ge_dur": d[pos & (d.g.values >= d.dur.values)].video.tolist(),
        "xfit_f1": {k: round(xfit(d, early=e, fpfn=f, **kw), 3) for k, (e, f) in VARIANTS.items()},
        "tuned": {"f1": round(F, 3), "tau": tau, "delta": delta, "P": round(tp / max(tp + fp, 1), 3),
                  "R": round(tp / max(tp + fn, 1), 3), "ci95": boot([(d, tau, delta)], **kw)},
        "at_official_settings": {k: round(f1(counts(d, tau, delta, e, f, integer)), 3) for k, (e, f) in VARIANTS.items()},
        "blind_floor": {k: round(blind(d, e, f, integer)[0], 3) for k, (e, f) in VARIANTS.items()},
        "auroc": round(auroc(d.y.values, d.q.values), 3),
        "within_scene_auroc": within_scene(d) if "scene" in d else None,
        "within_scene_auroc_len_ctrl_g1b": within_scene_len(d) if "scene" in d else None,
        "within_scene_perm_p": {"raw": perm_p(d), "len_ctrl": perm_p(within_len_q(d))} if "scene" in d else None,
        "timing_hit_g3": timing_hit(d),
        "ref_duration_only": ref_duration_only(d),
        "w_all_pos": round(hit[pos].mean(), 3),
        "w_emitted_pos": round(hit[pos & emit].mean(), 3) if (pos & emit).any() else None,
        "timing_err_p_minus_g_pct5_25_50_75_95":
            np.percentile((p - d.g.values)[pos & emit], [5, 25, 50, 75, 95]).round(2) if (pos & emit).any() else None,
        "median_abs_err_model_vs_blind": [round(np.median(np.abs(d.t.values + delta - d.g.values)[pos]), 2),
                                          round(np.median(np.abs(cstar - d.g.values)[pos]), 2)],
        "per_fold_f1": {str(k): round(f1(counts(d[d.fold == k], tau, delta, **kw)), 3) for k in sorted(d.fold.unique())},
    }
    for col in ("night", "dim", "event_type", "scene"):
        if col in d:
            g = d.assign(fp=emit & ~pos, tp=emit & pos & hit)
            r[f"by_{col}"] = {str(v): {"neg_fp_rate": round(x.fp[x.y == 0].mean(), 3) if (x.y == 0).any() else None,
                                       "pos_tp_rate": round(x.tp[x.y == 1].mean(), 3) if (x.y == 1).any() else None,
                                       "n": len(x)} for v, x in g.groupby(col)}
    return r, (tau, delta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--vs"); ap.add_argument("--int", action="store_true")
    ap.add_argument("--variant", choices=list(VARIANTS), default="official", help="variant of the --vs pairing")
    a = ap.parse_args()
    load = lambda f: pd.read_csv(f).sort_values("video").reset_index(drop=True)
    d = load(a.csv)
    r, (tau, delta) = report(d, a.int)
    if a.vs:  # paired bootstrap of F1(this) - F1(other) on the same videos
        o = load(a.vs)
        assert (o.video.values == d.video.values).all(), "runs must cover the same videos"
        e, f = VARIANTS[a.variant]
        kw = dict(early=e, fpfn=f, integer=a.int)
        (_, t1, d1), (_, ta, de) = tune(d, **kw), tune(o, **kw)  # official: the same tau, delta as before
        key = "paired_delta_f1_ci95" + ("" if a.variant == "official" else f"_{a.variant}")
        r[key] = boot([(d, t1, d1), (o, ta, de)], **kw)
    print(json.dumps(r, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))


if __name__ == "__main__":
    main()
