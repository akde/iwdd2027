"""Candidate row for the manuscript's Table 3 (tab:pairs): the REAL_EXTENDED arm (seeds 0-2, padded real epochs) with the
paper's columns, computed with the paper's definitions at cross-fitted operating points on all 500 clips. The two
existing fixed-5 rows (REAL_ORIGINAL g0 = row 3, MIXED_MATCHED g0 = row 4) are recomputed first as a check.
F1; AUC = pooled AUROC of q; AUC_ws = within-scene AUROC; R_s, R_d = share of static / dynamic positives detected inside
the window; FPR = share of the 250 negatives with an emission; F1_n = F1 on the 50 night clips.
Run from the repository root: python camera_ready/table3_row.py"""
import json
import numpy as np
from common import E, OUT, load_oof, outcomes, xfit_points, decide, f1
from budget import RUNS

rows = {}
for name, (arm, g) in {"fixed 5, no pairs (row 3)": ("REAL_ORIGINAL", "g0"), "fixed 5, pairs (row 4)": ("MIXED_MATCHED", "g0"),
                       "fixed 5, no pairs, padded to the pairs' updates (new)": ("REAL_EXTENDED", "g0")}.items():
    d = load_oof(RUNS[arm][g] / "oof.csv")
    emit, p = decide(d, xfit_points(d))
    o = outcomes(d, emit, p)
    hit = o[:, 0] == 1
    st, dy, neg, night = (d.event_type == "staticdumping").values, (d.event_type == "dynamicdumping").values, (d.y == 0).values, (d.night == 1).values
    rows[name] = dict(F1=round(f1(o.sum(0)), 3), AUC=round(float(E.auroc(d.y.values, d.q.values)), 3),
                      AUC_ws=E.within_scene(d)["auroc"], R_s=round(hit[st].mean(), 3), R_d=round(hit[dy].mean(), 3),
                      FPR=round(emit[neg].mean(), 3), F1_n=round(f1(o[night].sum(0)), 3), n_static=int(st.sum()),
                      n_dynamic=int(dy.sum()), n_night=int(night.sum()))
    print(name, rows[name])
(OUT / "table3_row.json").write_text(json.dumps(rows, indent=1))
