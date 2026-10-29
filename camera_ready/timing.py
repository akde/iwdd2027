"""TIMING: does the detector's own timestamp beat a constant timestamp on exactly the same emitted clips?
Post-submission analysis. CPU only; reads the OOF predictions (q, t per clip) of the submitted recipe (group g0,
seeds 0-2) and, as a robustness check, of the four further seed groups g1..g4 of the same recipe.

Protocol, fixed before scoring:
- For each held-out fold k, (tau_k, delta_k) = iwdd_eval.tune on the other four folds (101 q-quantiles x delta in
  {-2, -1.5, ..., 6}, contest rule e5_fpfn, integer outputs): the paper's cross-fitted operating point.
- Emission mask on fold k: q >= tau_k. Both variants use this mask; the threshold is never re-tuned for the constant.
- Learned timing: p = max(0, min(floor(t + delta_k), floor(T))).
- Constant timing: p = max(0, min(c_k, floor(T))), c_k in {0, 1, ..., 30} s, chosen on the other four folds, whose
  clips are emitted at the same tau_k, by F1 under the contest rule; ties go to the smallest c.
- Primary endpoint: F1 on all 500 clips (held-out folds pooled), learned minus constant, with a paired scene bootstrap
  (183 scenes, 5000 resamples, numpy default_rng(1)) holding predictions and per-fold parameters fixed. The public-test
  subset (100 clips, 70 scenes) is reported separately; it is part of the development set.
Caveat: tau_k was tuned jointly with delta_k for the learned timestamp, so the shared mask was chosen for the learned
variant. This is the paper's cross-fitting of the operating point, not nested model selection (the recipe itself
was chosen on all 500 clips).
Run from the repository root: python camera_ready/timing.py"""
import json
import numpy as np, pandas as pd
from common import OUT, KW, SEEDS, RUN, run_dir, load_oof, outcomes, xfit_points, f1, SceneBoot, public_mask, to_output

GROUPS = RUN["MIXED_MATCHED"]
CS = np.arange(0, 31)


def const_p(d, c):
    return to_output(np.full(len(d), float(c)), 0.0, d.dur.values)


def run(d):
    pts = xfit_points(d)
    rows = []
    for k, (tau, delta) in pts.items():
        o = d[d.fold != k]
        eo = o.q.values >= tau
        fs = [f1(outcomes(o, eo, const_p(o, c)).sum(0)) for c in CS]
        c = int(CS[int(np.argmax(fs))])  # argmax returns the first (smallest c) of tied maxima
        h = d[d.fold == k]
        rows.append(h.assign(tau=tau, delta=delta, c=c, sel_f1_const=round(max(fs), 4),
                             emit=h.q.values >= tau, p_learned=to_output(h.t.values, delta, h.dur.values),
                             p_const=const_p(h, c)))
    return pd.concat(rows).sort_values("video").reset_index(drop=True), pts


def summary(r, m, sb):
    a, b = outcomes(r, r.emit.values, r.p_learned.values), outcomes(r, r.emit.values, r.p_const.values)
    pos, em = r.y.values == 1, r.emit.values
    hit = lambda p: (p >= r.g.values - 5) & (p <= r.g.values + 10)
    out = {}
    for name, x, p in (("learned", a, r.p_learned.values), ("constant", b, r.p_const.values)):
        c = x[m].sum(0)
        out[name] = dict(TP=int(c[0]), FP=int(c[1]), FN=int(c[2]), F1=round(f1(c), 4),
                         timing_success_emitted_positives=f"{int((hit(p) & pos & em & m).sum())}/{int((pos & em & m).sum())}",
                         timing_success_all_positives=f"{int((hit(p) & pos & em & m).sum())}/{int((pos & m).sum())}")
    pt, ci, zero = sb.ci(a, b)
    out["learned_minus_constant"] = dict(dF1=pt, scene_ci95=ci, zero_denominator_resamples=zero)
    out["emitted"] = f"{int((em & m).sum())}/{int(m.sum())}"
    return out


R = {"protocol": __doc__, "groups": {}}
allp = []
for gname, run_id in GROUPS.items():
    d = load_oof(run_dir("MIXED_MATCHED", gname) / "oof.csv")
    r, pts = run(d)
    pub = public_mask(r)
    res = {"run": run_id, "per_fold": {int(k): dict(tau=round(float(v[0]), 4), delta=float(v[1]),
                                                   c=int(r[r.fold == k].c.iloc[0]),
                                                   selection_f1_const_other_folds=float(r[r.fold == k].sel_f1_const.iloc[0]))
                                       for k, v in pts.items()},
           "all_500": summary(r, np.ones(len(r), bool), SceneBoot(r)),
           "public_100": summary(r, pub, SceneBoot(r, pub))}
    R["groups"][gname] = res
    oc = lambda p: np.where(r.y.values == 1, np.where(~r.emit.values, "FN", np.where((p >= r.g.values - 5) & (p <= r.g.values + 10), "TP", "FP+FN")),
                            np.where(r.emit.values, "FP", "TN"))
    allp.append(r.assign(group=gname, run=run_id, split=np.where(pub, "public_test", "train"),
                         outcome_learned=oc(r.p_learned.values), outcome_const=oc(r.p_const.values))
                [["video", "scene", "fold", "split", "group", "run", "y", "g", "dur", "q", "t", "tau", "delta", "c", "emit",
                  "p_learned", "p_const", "outcome_learned", "outcome_const"]])
    a = res["all_500"]
    print(f"{gname}: learned {a['learned']['F1']} vs constant {a['constant']['F1']}: {a['learned_minus_constant']}; "
          f"c by fold {[v['c'] for v in res['per_fold'].values()]}; timing on emitted positives "
          f"{a['learned']['timing_success_emitted_positives']} vs {a['constant']['timing_success_emitted_positives']}")
(OUT / "timing.json").write_text(json.dumps(R, indent=1))
pd.concat(allp).to_csv(OUT / "timing_predictions.csv", index=False)
