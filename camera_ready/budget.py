"""BUDGET + SEEDS: synthetic content vs extra optimisation (post-submission analysis).

Arms (all: submitted E3r recipe, form m, robust augmentation, 5 fixed epochs, final update = checkpoint, data/folds.csv):
  MIXED_MATCHED  real clips + synthetic minimal pairs (the submitted recipe), pool n + 2*floor(n/2) per epoch
  REAL_EXTENDED  real clips only, --pad-real: the same pool size, update count and cosine schedule as MIXED_MATCHED
  REAL_ORIGINAL  real clips only, pool n per epoch (about half the updates)
Seed groups (3 seeds per fold each; same seed = same initial weights in every arm): g0 = seeds 0-2, g1 = 3-5, g2 = 6-8.
Contrasts: content = MIXED_MATCHED - REAL_EXTENDED (primary); exposure = REAL_EXTENDED - REAL_ORIGINAL;
link = MIXED_MATCHED - REAL_ORIGINAL (the earlier confounded comparison, g0 = paper Tab. 3 rows 4 vs 3).
Each arm/group/seed uses its own cross-fitted operating points (iwdd_eval.tune on the other 4 folds, contest rule,
integer outputs). Primary: all 500 clips. Intervals: paired scene bootstrap (183 scenes; 5000 resamples; numpy
default_rng(1)), predictions and per-fold operating points held fixed; seed variability is shown separately by the
per-seed and per-group results. Public-test subset (100 clips, 70 scenes; part of the development set) separately.
GPU nondeterminism: an identical rerun of one fit (fold 0, seed 0) changed clip scores by 0.033 on average (parity.py),
so a seed pairs initial weights and data order, not identical training.
Run from the repository root: python camera_ready/budget.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from common import E, OUT, ROOT, SEEDS as ALL_SEEDS, run_dir, load_oof, outcomes, xfit_points, decide, f1, SceneBoot, public_mask

SEEDS = {g: ALL_SEEDS[g] for g in ("g0", "g1", "g2")}
RUNS = {arm: {g: run_dir(arm, g) for g in SEEDS} for arm in ("MIXED_MATCHED", "REAL_EXTENDED", "REAL_ORIGINAL")}
CONTRASTS = {"content (MIXED_MATCHED - REAL_EXTENDED)": ("MIXED_MATCHED", "REAL_EXTENDED"),
             "exposure (REAL_EXTENDED - REAL_ORIGINAL)": ("REAL_EXTENDED", "REAL_ORIGINAL"),
             "link (MIXED_MATCHED - REAL_ORIGINAL)": ("MIXED_MATCHED", "REAL_ORIGINAL")}


def evaluate(path):  # -> per clip outcomes at its own cross-fitted points, plus a summary
    d = load_oof(path)
    pts = xfit_points(d)
    emit, p = decide(d, pts)
    o = outcomes(d, emit, p)
    pub = public_mask(d)
    s = lambda m: dict(zip(("TP", "FP", "FN"), o[m].sum(0).astype(int).tolist())) | {"F1": round(f1(o[m].sum(0)), 4)}
    ws = E.within_scene(d)
    return d, o, dict(all_500=s(np.ones(500, bool)), public_100=s(pub), emit_rate=round(float(emit.mean()), 3),
                      auroc=round(float(E.auroc(d.y.values, d.q.values)), 4),
                      within_scene_auroc=dict(auroc=ws["auroc"], pairs=ws["pairs"], scenes=ws["scenes"]),
                      points={k: [round(float(a), 4), float(b)] for k, (a, b) in pts.items()})



def main():
    R = {"protocol": __doc__, "status": {}, "groups": {}, "seeds": {}, "contrasts_group": {}, "contrasts_seed": {}}
    O, D = {}, {}
    for arm, gs in RUNS.items():
        for g, path in gs.items():
            ok = (path / "oof.csv").exists()
            R["status"][f"{arm} {g}"] = ("done: " + str(path)) if ok else "MISSING (not finished)"
            if not ok:
                continue
            cfg = json.loads((path / "config.json").read_text())
            assert tuple(cfg["seeds"]) == SEEDS[g] and cfg.get("epochs_fixed") == 5 and cfg["aug"]["lowq"] == 0.5, path
            assert ("synthetic" in cfg) == (arm == "MIXED_MATCHED") and ("pad_real" in cfg) == (arm == "REAL_EXTENDED"), path
            D[arm, g], O[arm, g], R["groups"][f"{arm} {g}"] = evaluate(path / "oof.csv")
            for s in SEEDS[g]:
                _, O[arm, g, s], R["seeds"][f"{arm} seed {s}"] = evaluate(path / f"oof_seed{s}.csv")

    ref = next(iter(D.values()))
    SB = {"all_500": SceneBoot(ref), "public_100": SceneBoot(ref, public_mask(ref))}
    for name, (a, b) in CONTRASTS.items():
        R["contrasts_group"][name], R["contrasts_seed"][name] = {}, {}
        for g, seeds in SEEDS.items():
            if (a, g) in O and (b, g) in O:
                R["contrasts_group"][name][g] = {sp: dict(zip(("dF1", "scene_ci95", "zero_denominator_resamples"),
                                                              sb.ci(O[a, g], O[b, g]))) for sp, sb in SB.items()}
                for s in seeds:
                    R["contrasts_seed"][name][s] = round(R["seeds"][f"{a} seed {s}"]["all_500"]["F1"]
                                                         - R["seeds"][f"{b} seed {s}"]["all_500"]["F1"], 4)
        gd = [v["all_500"]["dF1"] for v in R["contrasts_group"][name].values()]
        sd = list(R["contrasts_seed"][name].values())
        R["contrasts_group"][name]["summary"] = dict(groups=len(gd), mean=round(float(np.mean(gd)), 4) if gd else None,
                                                     min=min(gd, default=None), max=max(gd, default=None),
                                                     seeds=len(sd), seed_mean=round(float(np.mean(sd)), 4) if sd else None,
                                                     seed_min=min(sd, default=None), seed_max=max(sd, default=None),
                                                     seeds_positive=int(sum(x > 0 for x in sd)))
        (OUT / "budget.json").write_text(json.dumps(R, indent=1, default=float))
    for k, v in R["status"].items():
        print(f"{k:22s} {v[:60]}")
    for name in CONTRASTS:
        print(name, {g: (x["all_500"]["dF1"], x["all_500"]["scene_ci95"]) for g, x in R["contrasts_group"][name].items() if g != "summary"},
              R["contrasts_group"][name]["summary"])


if __name__ == "__main__":
    main()
