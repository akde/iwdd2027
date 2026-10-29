"""E4 (wiki/experiments.md, post hoc, disclosed): DINOv2 clip-level probe, spec frozen as found in the 2026-10-04 review.

x = per-dimension mean over the clip's stored 8 fps rows of |cls - median(cls)| (features/<video>.npz, 768-d).
Per outer fold: StandardScaler + LogisticRegression(C=0.01, max_iter=3000) fit on the 4 training folds; q = sigmoid of
the decision function on the held-out fold; t = 9.0 for every clip (the scorer's delta makes it a constant time).
python src/probe.py -> runs/e4_probe/: oof.csv, probe_f<k>.npz (scaler mean/scale, coef, intercept), config.json,
metrics.json.
"""
import hashlib, json, subprocess

import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data import ROOT
from features import OUT
from train import clips, report

C, T_CONST = 0.01, 9.0


def stat(v):
    x = np.load(OUT / f"{v}.npz")["cls"].astype(np.float32)
    return np.abs(x - np.median(x, 0)).mean(0)


def main():
    run = ROOT / "runs" / "e4_probe"
    run.mkdir(parents=True, exist_ok=True)
    R = clips()
    vids = sorted(R)
    X, y, fold = np.array([stat(v) for v in vids]), np.array([R[v].y for v in vids]), np.array([R[v].fold for v in vids])
    q = np.zeros(len(vids))
    for k in sorted(set(fold)):
        tr, te = fold != k, fold == k
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[tr]), y[tr])
        q[te] = 1 / (1 + np.exp(-m.decision_function(sc.transform(X[te]))))
        np.savez(run / f"probe_f{k}.npz", mean=sc.mean_, scale=sc.scale_, coef=m.coef_[0], intercept=m.intercept_)
    rows = [dict(video=v, fold=R[v].fold, y=R[v].y, g=R[v].g, dur=R[v].T, q=q[i], t=T_CONST, night=R[v].night,
                 dim=R[v].dim, event_type=R[v].event_type, scene=R[v].scene) for i, v in enumerate(vids)]
    pd.DataFrame(rows).to_csv(run / "oof.csv", index=False)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    cfg = dict(experiment="E4", C=C, t=T_CONST, git=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain", "src", "tools")),
               features_meta_sha256=hashlib.sha256((OUT / "meta.csv").read_bytes()).hexdigest())
    (run / "config.json").write_text(json.dumps(cfg, indent=1))
    o = pd.read_csv(run / "oof.csv")
    met = {k: report(o, i)[0] for k, i in (("int", True), ("float", False))}
    (run / "metrics.json").write_text(json.dumps(met, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    print(f"E4: xfit F1 (--int) {met['int']['xfit_f1']}, AUROC {met['int']['auroc']}, within-scene "
          f"{met['int']['within_scene_auroc']}, G1b {met['int']['within_scene_auroc_len_ctrl_g1b']}")


if __name__ == "__main__":
    main()
