# IWDD 2027: illegal waste dumping detection

Code and trained models of team HACETTEPE (Burak Akdemir, Seniha Esen Yüksel Erdem; Hacettepe University) for the
[IWDD 2027 contest](https://mivia.unisa.it/iwddcontest2027/) of the
[WasteVision 2027 workshop](https://mivia.unisa.it/wastevision2027/) at WACV 2027.

For each video the method writes one integer, the predicted onset of the dumping in seconds from the start, or an
empty file. Frames are decoded at 5 fps (times from the PTS). A MoViNet-A2 network (Kinetics-600 weights) runs with
its trunk frozen and its last blocks fine-tuned, and a dilated temporal convolution head turns its output into one
logit per frame. The frame logits give a video score q and a time t (`submission/test.py`, `src/head.py`). A video
gets an answer iff q >= TAU, and the answer is `max(0, min(floor(t + DELTA), floor(T)))` (`src/decision.py`). The
shipped models `submission/model/g<k>_s<s>.pt` are 15 networks (5 groups x 3 seeds), each trained on the 500
labelled contest clips plus synthetic clips drawn from a pool of 1722 (250 minimal pairs per epoch); their outputs
are averaged. The 15 cross-validation models `f<k>_s<s>.pt` of the same recipe serve only the per-fold mode
(`--folds data/folds.csv`) of the package test.

## Install
Python 3.12 with a CUDA build of PyTorch (tested with torch 2.11.0), then

```
pip install -r requirements.txt   # PyAV, OpenCV, einops (pinned), pandas
```

## Run
```
python test.py --videos <video dir>/ --results <result dir>/
```

`test.py` writes `<result dir>/<video file name>.txt` for every file in `<video dir>` (for example `vid0041.mp4.txt`):
one integer with no newline, or nothing. It installs `av` and `einops` itself if they are missing. The checkpoints
are checked against the SHA256 in `submission/model/MANIFEST.json` when loaded (the MoViNet weights against
`src/features.py`). Without a CUDA device, or if a model file fails its check, every video gets
`floor(min(13, duration))`. On Google Colab (GPU runtime) open `test.ipynb`, set `ROOT`, `VIDEOS` and `RESULTS`, and
run all cells.

## Train
The MIVIA-IWDD videos and labels come from the contest organizers and are not in this repository. Put them as

```
data/train/videos/<video>.mp4         data/train/labels/<video>.json
data/public_test/videos/<video>.mp4   data/public_test/labels/<video>.json
```

`data/folds.csv` holds the 5 scene-grouped folds used for training and cross-validation.

```
python src/data.py                     # cache/index.csv: decoder duration of every clip
python src/features.py --movinet       # features_movinet/: frozen MoViNet-A2 features (read by run and full)
python src/finetune.py cache           # cache/frames5/: 5 fps frames
python src/finetune.py cache --release <synthetic release dir>   # cache/frames5_synth/<release>/: frames, index.csv
python src/finetune.py run --form m --robust --extra cache/frames5_synth/<release>/index.csv --cap 1.0 --epochs 5 --run <cv id>
python src/finetune.py full --form m --robust --extra cache/frames5_synth/<release>/index.csv --cap 1.0 --epochs 5 --run <full id>
```

`--form m` subtracts each sequence's median from its features, and `--robust` adds the low-quality and frame-shape
augmentation. `run` trains the recipe per fold and writes the out-of-fold predictions `runs/<cv id>/oof.csv` and the
fold models `f<k>_s<s>.pt`. `full` trains the 15 models on all clips (`g<k>_s<s>.pt`); give it a different id from
`run`. Scoring and the operating point use the out-of-fold predictions:

```
python tools/iwdd_eval.py runs/<cv id>/oof.csv --int                     # all metrics; the contest rule is variant e5_fpfn
python tools/operating_point.py runs/<cv id>/oof.csv --variant e5_fpfn   # TAU, DELTA
python tools/check_scorer.py                                             # scorer checks on synthetic cases, no data
```

To package retrained models, copy the f- and g-files into `submission/model/`, update `MANIFEST.json` (run, files,
full.run, full.epochs, full.files with their SHA256, full.insample_sha256) and set TAU, DELTA, RUN and VARIANT in
`src/decision.py`. `tests/test_package.py`
checks that `test.py` reproduces the training-time predictions clip by clip. It needs the data, `cache/` and the two
training runs behind the shipped models (`runs/s5_movinet_ft_r_v23_fe5/`, `runs/s5_movinet_ft_r_v23_full5/`), which
are not included because they contain the contest labels.

`src/probe.py`, `src/train.py --inputs cls|cls_step|grid` and `src/features.py` without `--movinet` belong to an
earlier DINOv2 baseline. They need facebookresearch/dinov2 at commit 7b187bd in `vendor/dinov2-7b187bd/` and
`weights/dinov2_vitb14_reg4_pretrain.pth`, which are not included, and `src/probe.py` needs scikit-learn.

## Camera-ready analyses
`camera_ready/` holds the post-submission analyses of the camera-ready paper: the detector's timestamp against a
constant timestamp on the same emitted clips, the synthetic pairs against real-only training with the same number of
optimiser updates (`src/finetune.py run --pad-real`), and rule-based failure examples. `camera_ready/README.md` lists
the training runs, the commands and the reference results. `src/finetune.py run --resume` continues an interrupted
run from its saved fold models.

## Synthetic data
`synth/` holds the generator of the synthetic training clips (`synth/gen`, `synth/scripts`; the set the shipped
models used: `synth/RELEASE.md`). Each clip is one member of a minimal pair: a positive and a negative that are
identical until the outcome. Actors (Microsoft Rocketbox avatars and animations) and objects (Poly Haven meshes) are
rendered with Blender 4.2 and composited onto person-free background segments ("plates") of real CCTV videos. The
shipped models used plates of the contest videos and of MEVA; this repository holds the inputs of the 136 MEVA
plates only (the contest plates would reveal contest labels; 66 MEVA plates from 6 cameras pass the calibration
check and are sampled): camera calibrations (`synth/calib/`), structure masks
(`synth/calib/structures/`), the plate, light, distractor-line and asset tables, the MEVA keys and the person
detections (`synth/data/`). The generator still draws clip durations, onset times, actor sizes, person brightness and
the video degradation from the contest clips, so it needs the contest data too. The rendered clips, the assets and the
videos are not included.
The rendered pairs on MEVA backgrounds (594 of the 861 pairs of release v3, 1188 clips, CC BY 4.0) are released on
Zenodo: https://doi.org/10.5281/zenodo.23365921. The 267 pairs rendered on contest backgrounds are not released.

Requirements: an NVIDIA GPU with OptiX, Blender 4.2.23 at `~/opt/blender-4.2.23-linux-x64/blender` (`BLENDER` in
`synth/gen/run_batch.py`), a systemd user session (each Blender call runs in `systemd-run --user --scope`), the contest
data in `data/` as above, and `yolo11x.pt` and `yolo11x-seg.pt` in `synth/assets/` (URLs in
`synth/assets/LICENSES.csv`). From the repository root:

```
pip install -r synth/requirements.txt && pip install --no-deps ultralytics==8.3.253 ultralytics-thop==2.2.2
python synth/scripts/fetch_rocketbox.py Assets/Animations Assets/Avatars/Adults \
  Assets/Avatars/Professions/Construction Assets/Avatars/Professions/Delivery Assets/Avatars/Professions/Gardener \
  Assets/Avatars/Professions/Sports Assets/Avatars/Professions/Wood          # 6.7 GB
python synth/scripts/fetch_objects.py
~/opt/blender-4.2.23-linux-x64/blender -b -P synth/scripts/build_assets.py   # library.blend (also rewrites the avatar and object tables)
python synth/scripts/fetch_meva.py get synth/data/meva/plate_keys.txt       # 35 videos, about 6 GB
python src/data.py                               # cache/index.csv
python synth/scripts/taxonomy.py sheets          # synth/data/taxonomy/clips.csv (and contact sheets)
python synth/scripts/person_ratio.py             # brightness of real people in the contest clips
python synth/scripts/plate_structures.py raster $(ls synth/calib/structures | sed 's/\.json$//')
python synth/scripts/plate_obstacles.py
python synth/scripts/plate_vehicles.py
python synth/scripts/plate_traffic.py
python synth/scripts/plate_motion.py
python synth/gen/run_batch.py --groups 600 --seed 9920 --workers 5 --data-dir synth/data/<batch>
python synth/scripts/make_release.py synth/data/<release> synth/data/<batch>
python synth/scripts/check_release.py synth/data/<release>
```

Use a seed other than those of release v3 (3, 4, 15, 16, 17, 9910), whose pair ids are in
`synth/data/exclude_pairs.csv`. Do not rerun the plate-building scripts (find_plates, verify_plates, extend_plates,
calibrate, check_calib): they rebuild the plate tables from scratch, contest plates included. This makes new
MEVA-background pairs. It does not rebuild release v3: its contest-background pairs need the contest
plates, and 292 of its 861 pairs came from earlier versions of the generator.

## License
The code in this repository is released under the MIT License (`LICENSE`). Third-party parts keep their own terms:

- `vendor/movinet-c2d1edf/`: MoViNet-pytorch by Atze00 (MIT, `vendor/movinet-c2d1edf/LICENSE`); `iwdd.patch` lists
  our changes, already applied.
- `submission/model/movinet_a2_base_statedict_v3` is MoViNet-pytorch's `modelA2_statedict_v3`, unchanged: the
  MoViNet-A2 Kinetics-600 weights of Kondratyuk et al. (CVPR 2021), converted from the TensorFlow Model Garden
  release, Apache License 2.0 (`submission/model/LICENSE-Apache-2.0`). The fine-tuned checkpoints
  `submission/model/*.pt` are derived from them and are released under the same Apache License 2.0.
- The synthetic-data scripts use Blender (GPL-3.0), Microsoft Rocketbox (MIT), Poly Haven assets (CC0), MEVA
  (Corona et al., WACV 2021; CC BY 4.0) and Ultralytics YOLO11 (AGPL-3.0); none of them is included.
- The MIVIA-IWDD dataset (Greco et al., IMAVIS 2026; Bouwmans et al., WACVW 2026) belongs to its authors and is
  available from the contest organizers. `data/folds.csv` (our scene-grouped folds) and `synth/data/det/vid*.npz`
  (person detections and frame statistics of the contest clips) are derived from it.
- The plate files in `synth/calib/` and `synth/data/` (calibrations, masks, plate tables, the MEVA detections and
  keys) are derived from MEVA (CC BY 4.0, Kitware / IARPA).
