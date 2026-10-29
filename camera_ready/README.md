# Camera-ready analyses

Post-submission analyses added for the camera-ready version of the paper. They answer two questions about the
submitted recipe on the 500 labelled development clips (400 training + 100 public-test clips), with the five
scene-grouped folds of `data/folds.csv`:

- **TIMING**: does the detector's own timestamp beat a constant timestamp on exactly the same emitted clips?
- **BUDGET**: does adding the synthetic minimal pairs help once the number of optimiser updates is matched?

and select failure cases by a fixed rule (**FAILURES**). Event-level F1 follows the contest rule (a hit iff
g-5 <= p <= g+10; a mistimed detection on a positive counts as FP + FN; integer outputs). Every model uses its own
cross-fitted operating points (tau, delta tuned on the other four folds). Intervals are paired scene-bootstrap 95%
intervals (183 scenes, 5000 resamples) with the predictions and operating points held fixed.

## Training runs

All runs use the submitted recipe (`--form m --robust`, 5 fixed epochs, the checkpoint after the last update). Seed
group gK is seeds 3K, 3K+1, 3K+2. Prepare the data and caches as in the main README (Train), then:

```
X=cache/frames5_synth/release_v3/index.csv
# MIXED_MATCHED: real clips + synthetic pairs (the submitted recipe); g0..g4
python src/finetune.py run --form m --robust --extra $X --cap 1.0 --epochs 5 --seeds 0,1,2 --run mixed_g0
python src/finetune.py run --form m --robust --extra $X --cap 1.0 --epochs 5 --seeds 3,4,5 --run mixed_g1   # ... to g4 (seeds 12-14)
# REAL_EXTENDED: real clips only, padded to the mixed runs' pool size, update count and learning-rate schedule; g0..g2
python src/finetune.py run --form m --robust --epochs 5 --pad-real --seeds 0,1,2 --run realext_g0             # ... g1, g2
# REAL_ORIGINAL: real clips only; g0..g2
python src/finetune.py run --form m --robust --epochs 5 --seeds 0,1,2 --run realorig_g0                       # ... g1, g2
```

`mixed_g0` is the out-of-fold run of the shipped models (the run `tests/test_package.py` calls
`runs/s5_movinet_ft_r_v23_fe5`). `--pad-real` adds 2 floor(n/2) further real clips per epoch, drawn without replacement,
so that each fold's pool (784-805 clips), update count (245-255; batch 16, no drop-last) and cosine schedule equal those
of the mixed run; REAL_ORIGINAL takes 125-130 updates. Equal updates do not mean equal real exposure: REAL_EXTENDED shows
each real clip about twice per epoch. After an interruption, rerun the same command with `--resume`: finished fits are
loaded from their checkpoints (written atomically) instead of retrained. If the synthetic release directory has another
name, set `SYNTH_INDEX` in `camera_ready/common.py`.

## Analyses (repository root, in this order)

```
python camera_ready/timing.py              # TIMING -> output/timing.json, timing_predictions.csv
python camera_ready/budget.py              # BUDGET per seed group and per seed -> output/budget.json
python camera_ready/exposure.py            # updates, presentations, unique pairs per fit -> output/exposure.csv
python camera_ready/ws_auroc.py            # within-scene AUROC with paired intervals -> output/ws_auroc.json
python camera_ready/table3_row.py          # the paper's synthetic-pairs table rows -> output/table3_row.json
python camera_ready/tables.py              # output/results.csv, output/results.tex
python camera_ready/budget_predictions.py  # per-clip records of every arm, group and seed
python camera_ready/frame_logits.py --run mixed_g0   # GPU: out-of-fold frame logits rebuilt from the fold models
python camera_ready/failures_select.py     # rule-based failure examples -> output/failures_*.csv
python camera_ready/failures_figure.py VIDEO:"(a) Detected" VIDEO:"(b) Missed" ...   # output/failure_cases.pdf
python camera_ready/parity.py --run realorig_g0      # GPU, optional: run-to-run variation of one retrained fit
```

`camera_ready/output/` holds per-clip labels and is not under version control. The failure examples in the paper were
chosen by `failures_select.py` and then checked by hand; their descriptions are not included because they describe
contest clips.

## Reference results (all 500 clips, cross-fitted operating points)

| Comparison | g0 | g1 | g2 |
|---|---|---|---|
| TIMING: learned minus constant timestamp, same emitted clips | +0.004 [-0.039, +0.045] | +0.016 [-0.021, +0.053] | +0.020 [-0.019, +0.059] |
| BUDGET content: MIXED_MATCHED minus REAL_EXTENDED | -0.013 [-0.070, +0.034] | -0.030 [-0.091, +0.024] | -0.018 [-0.068, +0.029] |
| BUDGET exposure: REAL_EXTENDED minus REAL_ORIGINAL | -0.040 [-0.087, +0.000] | -0.038 [-0.082, +0.005] | -0.021 [-0.072, +0.024] |
| MIXED_MATCHED minus REAL_ORIGINAL | -0.053 [-0.112, -0.006] | -0.068 [-0.133, -0.011] | -0.039 [-0.089, +0.003] |

TIMING for g3 and g4: +0.012 [-0.031, +0.055] and +0.030 [-0.005, +0.066]. The constants chosen on the other folds
are 11-13 s. All rows: `reference_results.csv` (the `results.csv` these scripts write).

GPU training is not bit-reproducible even with cuDNN's deterministic mode: an identical retraining of one fit changed
clip scores by 0.033 on average and moved the peak in 24 of 98 clips. A shared seed therefore pairs the initial weights
and the data order across arms, not identical training; reruns reproduce these numbers only approximately.
