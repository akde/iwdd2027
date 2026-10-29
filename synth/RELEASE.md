# Synthetic IWDD clips, release v3 (2026-10-08)

The synthetic set the shipped models were trained on, built in the authors' working tree. The clips are not part of
this repository, which ships the inputs of the MEVA plates only, so the build and check commands below are a record.
Dnn and Fn refer to the authors' design log (not included).

Location (not in git): `synth/data/release_v3/` = `clips/<clip>.mp4`, `labels/<clip>.json`, `manifest.csv` (hard links
into the batch directories; make_release.py also writes `synth/release_v3_manifest.csv`). Built with
`python synth/scripts/make_release.py synth/data/release_v3 synth/data/batch_s34 synth/data synth/data/batch_v3r`;
checked with `python synth/scripts/check_release.py synth/data/release_v3` (every clip: label vs manifest, timestamp
rule, fold vs data/folds.csv, decoder duration, pair members identical but for the outcome; 1722 clips, 0 failures).

## Content
- 1722 clips = 861 minimal pairs, 381 MB, 9.05 h of video (1 positive + 1 negative, identical until the outcome;
  DECISIONS.md D14, D22, D27).
- Positives: `place_leave` 386 (on foot: carry in, put down, leave) and `unload_vehicle_leave` 475 (from a real parked
  car in the plate: carry 2.5-7 m, put down, return to the car; D26).
- Negatives: `carry_through` 194, `place_pickup` 117, `pickup_preexisting` 75 (on foot); `carry_from_car` 199,
  `unload_reload` 145, `load_preexisting` 131 (vehicle).

| Batch (seed) | Groups rendered | Accepted by QC | Kept after the full screen | Generator |
|---|---|---|---|---|
| s3 + s4 (release v1.2) | 300 | 159 | 84 | on foot, before D29 |
| s15 | 150 | 143 | 129 | vehicle unloading, before D29 |
| s16 | 100 | 91 | 51 | on foot, before D29 |
| s17 | 60 | 46 | 28 | on foot, longer plates, before D29 |
| v3 (seed 9910, `batch_v3r`) | 600 | 594 | 569 | both families, every D29 fix (D30-D46) |

- v3 batch: rendered from the fixed generator (structure masks, black bars, walk-through and crouch gates, distractor
  allowlist, upright endings, moving vehicles, sun from real shadows, vehicles as 3D solids for occlusion, the
  object's own exposure), re-composited from its kept renders with the final compositor (D45/D46). QC rejected 6
  groups (actor not visible 5, object small 1). Pre-render screen, structure, black-bar and leg-gate re-checks on all
  1200 members: 0 problems. The full visual screen of every accepted pair
  excluded 19 pairs (3.2%), and 6 more for real people on two plates (D47): 25 of 594 (4.2%; the old batches lost
  33%). One line per excluded pair in `synth/data/exclude_pairs.csv` (batch and the check that found it; 172 rows).
- Backgrounds: real person-free CCTV segments; 20 scenes (15 contest scenes, 5 MEVA cameras), 76 plates; MEVA 69% of
  clips (v3 batch 74%). Most-used scenes: meva_G506 172, meva_G341 170, meva_G336 131, s153 97, meva_G300 91 pairs.
- Durations: on foot median 10.8 s (10-90%: 5.3-26.0), vehicle 20.7 s (13.3-35.4); real clips 18.9 s.
  Positive timestamps: median 3 s (on foot 4, vehicle 2); real median about 7 s.
- Night 4.8% of clips (v3 batch 3.5%, real 10%); frame rates 6-15 fps after degradation; 11 objects (bags and sacks
  67%), 54 avatars.
- Balance: labels 50/50; within every pair plate, duration, avatar, object, frame rate, resolution, CRF and fold are
  identical, so day/night, lighting, IR, duration and every object/avatar have the same share in both labels.

## Label JSON
Contest keys (`label` "true"/"false", `timestamp` integer s or -1, `metadata` with video_id, resolution, frame_rate,
duration, event_type, time_of_day, lighting) plus `t_onset_exact`, `t_release`, `t_dec`, `family`, `outcome`,
`pair_id`, `plate_id`, `scene_id`, `fold`, `seed`, `events`, `qc`. timestamp = round(t_onset_exact); t_onset_exact =
start of the bend + 0.76 s (on foot) or start of the walk from the car + 1.00 s (unloading) (taxonomy, D20, D26).
All times are seconds on the clip's own PTS axis. `t_dec` (first moment the outcome is visible) is null when the clip
ends before it: 25% of positives, 14% of negatives (v3 batch positives 18%).

## Degradation
Already applied (gen/degrade.py: real resolution/fps law capped by the source, H.264 CRF 28-42, noise, frame drops).
Invariant 2: apply the same `degrade.degrade_file(real, out, degrade.sample_params(...), seed)` law to real clips in
training; never degrade only the synthetic ones.

## Using it without leakage
- `fold` = the fold of the contest scene the background comes from (data/folds.csv); `-1` = MEVA (in no contest fold).
- When validating contest fold f, train on synthetic clips with `fold != f`; MEVA clips (fold -1) are always allowed.
- Pairs per fold: -1: 594, 0: 25, 1: 58, 2: 174, 3: 8, 4: 2. The contest-plate pairs sit mostly in fold 2 (scene s153
  alone has 97), so per-fold comparisons see very different amounts of in-domain synthetic data.
- The `batch` column tells the generator: `batch_s34` and `data` pairs come from the pre-D29 generator (screened and
  cleaned, D29), `batch_v3r` from the fixed one. An ablation can use the v3 batch alone (569 pairs).
- Keep or drop (spec): keep synthetic data only if real+synthetic beats real-only on scene-grouped CV by more than
  one fold standard deviation; otherwise train on synthetic only and test on real (AUC < 0.6 = realism bottleneck).

## Known limitations and deviations
- Occlusion: only still parked vehicles hide avatars (3D solids fitted to their masks, D45). Thin structures
  (branches, poles, railings) are drawn behind avatars; walls, pillars, roofs, piles and fences are avoided, not used
  as occluders (structure masks, D34). In car mode, positives are partly hidden behind their own car slightly more
  often than negatives (5.5% vs 4.3% of frames), because the dumper returns to the car (D46).
- Short on-foot positives can end while the actor stands next to the object or just starts to walk away (by design,
  D32/D35); the screen excluded those where the object is never seen apart from the actor (D47).
- Night 4.8% vs 10% real; car mode has no night plate. IR night exposure of avatars varies (dark and light
  silhouettes, D47); small dark or light objects can be faint after degradation (excluded when invisible).
- Scene diversity: 20 scenes (calibration needs real people); 69% MEVA; within a plate the drop spots cluster
  (e.g. the G341 building corner); contest folds 3 and 4 have 8 and 2 pairs.
- Actor size below the real law (on foot median ~0.13 of the frame height in the v3 batch vs 0.23 real; D34/D37).
- Sun: only 1 of 115 plates has hard real shadows (D42), so almost every v3 group is overcast or night.
- Rendered at <= 15 fps and <= 960x540 (D17); no car animation (doors and boot stay closed).
- Older batches (`batch_s34`, `data`, 292 pairs): composited before D39-D45 (no 3D-solid occlusion, shared actor/object
  gain, the shadow border box). Their pre-existing pairs do not share one foreground gain (D38; measured on release v2:
  median 0.1 grey levels, mixed sign). They are kept as release v2 had them.

## Earlier releases (history)
- Release v2 (`synth/data/release_v2/`, 292 pairs): the old batches
  after D29; all of it is part of v3.
- Release v1 (`synth/data/release_v1/`, v1.2: 159 pairs): 75 of its pairs fail D29.
