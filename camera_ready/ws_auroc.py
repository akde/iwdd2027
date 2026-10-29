"""BUDGET supplement: within-scene AUROC per arm and seed group, with paired scene-bootstrap intervals of the contrasts.
Within-scene AUROC = share of (positive, negative) pairs from the same scene in which the positive has the higher clip
score q (ties count 1/2), pooled over the 22 scenes holding both classes (254 pairs; one scene holds 108 of them).
Higher = better discrimination where camera identity carries no information. Bootstrap: the 22 scenes resampled with
replacement, 5000 draws, numpy default_rng(2), the same draws for both arms (paired). Clip discrimination, not event F1.
Run from the repository root: python camera_ready/ws_auroc.py"""
import json
import numpy as np
from common import OUT, load_oof
from budget import RUNS, CONTRASTS


def per_scene(d):  # scene -> (wins, pairs)
    out = {}
    for s, x in d.groupby("scene"):
        p, m = x.q[x.y == 1].values, x.q[x.y == 0].values
        if len(p) and len(m):
            out[s] = ((p[:, None] > m).sum() + 0.5 * (p[:, None] == m).sum(), len(p) * len(m))
    return out


S = {(arm, g): per_scene(load_oof(path / "oof.csv")) for arm, gs in RUNS.items() for g, path in gs.items()}
scenes = sorted(next(iter(S.values())))
assert all(sorted(v) == scenes for v in S.values())
draw = np.random.default_rng(2).integers(0, len(scenes), (5000, len(scenes)))
W = {k: np.array([v[s][0] for s in scenes]) for k, v in S.items()}
P = np.array([next(iter(S.values()))[s][1] for s in scenes])
auc = lambda w, i=slice(None): w[i].sum() / P[i].sum()
R = {"protocol": __doc__, "scenes": len(scenes), "pairs": int(P.sum()),
     "auroc": {f"{a} {g}": round(float(auc(w)), 4) for (a, g), w in W.items()}, "contrasts": {}}
for name, (a, b) in CONTRASTS.items():
    R["contrasts"][name] = {}
    for g in ("g0", "g1", "g2"):
        v = np.array([auc(W[a, g], i) - auc(W[b, g], i) for i in draw])
        R["contrasts"][name][g] = dict(d=round(float(auc(W[a, g]) - auc(W[b, g])), 4),
                                       scene_ci95=[round(float(x), 4) for x in np.percentile(v, [2.5, 97.5])])
(OUT / "ws_auroc.json").write_text(json.dumps(R, indent=1))
print(json.dumps({k: R[k] for k in ("scenes", "pairs", "auroc", "contrasts")}, indent=1))
