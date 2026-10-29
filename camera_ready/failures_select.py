"""FAILURES, example selection by a rule fixed before any clip is viewed (post-submission analysis).

Source: the submitted recipe's OOF predictions (run mixed_g0, seeds 0-2 averaged per fold, each clip scored by its
held-out fold's models) at the cross-fitted operating points (tau_k, delta_k), i.e. the rows of
output/timing_predictions.csv with group g0 and the learned timestamp. Curves: output/frame_logits_mixed_g0.npz
(camera_ready/frame_logits.py), which must reproduce oof.csv q and t (checked here).

Categories (contest rule): TP (emitted, p in [g-5, g+10]); MISS (positive, nothing emitted); EARLY (positive, emitted,
p < g-5); LATE (positive, emitted, p > g+10); FALSE ALARM (negative, emitted).
Rule per category: the key is |p - g| for TP / EARLY / LATE and q for MISS / FALSE ALARM. For each scene, take its clip
whose key is closest to the category median (ties: smaller video id); then take the 2 scenes whose representatives are
closest to the median (ties: smaller scene id). An empty category, or one with a single scene, is reported as such.
Unrelated-activity false alarms: the 20 false alarms with the highest q are listed for visual review; up to 3 of those
showing human activity unrelated to leaving an object are then chosen, from at least 2 scenes, highest q first.
Run from the repository root (after timing.py and frame_logits.py): python camera_ready/failures_select.py"""
import json
import numpy as np, pandas as pd
from common import OUT, ROOT
from head import score

P = pd.read_csv(OUT / "timing_predictions.csv")
P = P[P.group == "g0"].set_index("video")
LG = dict(np.load(OUT / "frame_logits_mixed_g0.npz"))
T4 = {v: np.load(ROOT / "features_movinet" / f"{v}.npz")["t"] for v in P.index}
dq = max(abs(score(LG[v], T4[v])[0] - P.q[v]) for v in P.index)
dt = max(abs(score(LG[v], T4[v])[1] - P.t[v]) for v in P.index)
assert dq < 1e-5 and dt < 1e-9, (dq, dt)

pos, emit, p, g = P.y == 1, P.emit.astype(bool), P.p_learned, P.g
cat = pd.Series("TN", index=P.index)
cat[pos & ~emit] = "MISS"
cat[pos & emit & (p >= g - 5) & (p <= g + 10)] = "TP"
cat[pos & emit & (p < g - 5)] = "EARLY"
cat[pos & emit & (p > g + 10)] = "LATE"
cat[~pos & emit] = "FALSE_ALARM"
P["category"] = cat
P["key"] = np.where(cat.isin(["TP", "EARLY", "LATE"]), (p - g).abs(), P.q)

sel, info = [], {}
for c in ["TP", "MISS", "EARLY", "LATE", "FALSE_ALARM"]:
    x = P[P.category == c].reset_index()
    info[c] = dict(clips=len(x), scenes=int(x.scene.nunique()))
    if not len(x):
        continue
    x["dist"] = (x.key - x.key.median()).abs()
    rep = x.sort_values(["dist", "video"]).groupby("scene", sort=False).head(1).sort_values(["dist", "scene"]).head(2)
    info[c].update(median_key=float(x.key.median()), selected=rep.video.tolist())
    sel += [dict(video=v, category=c) for v in rep.video]
fa = P[P.category == "FALSE_ALARM"].sort_values(["q", "scene"], ascending=[False, True]).head(20)
pd.DataFrame(sel).to_csv(OUT / "failures_selected_by_rule.csv", index=False)
fa.reset_index()[["video", "scene", "fold", "split", "q", "t", "p_learned", "dur"]].to_csv(OUT / "failures_fa_top20.csv", index=False)
P.reset_index()[["video", "scene", "fold", "split", "y", "g", "dur", "q", "t", "tau", "delta", "p_learned", "category"]].to_csv(
    OUT / "failures_categories_g0.csv", index=False)
json.dump(dict(categories=info, fa_top20=fa.index.tolist()), open(OUT / "failures_selection.json", "w"), indent=1)
print(json.dumps(info, indent=1))
