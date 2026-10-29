"""Assemble a release from accepted clips of several batch directories.
python synth/scripts/make_release.py OUT_DIR BATCH_DIR [BATCH_DIR ...]
OUT_DIR/clips/<clip>.mp4 and OUT_DIR/labels/<clip>.json (hard links), OUT_DIR/manifest.csv (accepted clips of all
batches, with the batch directory); a copy of the manifest goes to synth/<OUT_DIR name>_manifest.csv (not in git).
Clip names are unique across batches (group ids carry the seed). Pairs listed in synth/data/exclude_pairs.csv
(visual screening and geometric checks, DECISIONS.md D29) are left out."""
import os, pathlib, sys
import pandas as pd

SYN = pathlib.Path(__file__).resolve().parents[1]
COLS = ["clip", "y", "timestamp", "t_onset_exact", "t_dec", "family", "outcome", "pair_id", "plate_id", "scene_id",
        "fold", "source", "avatar", "object", "duration", "frame_rate", "resolution", "time_of_day", "lighting", "crf",
        "object_px", "height_ratio"]


def main(out, batches):
    out = pathlib.Path(out)
    (out / "clips").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(exist_ok=True)
    parts = []
    drop = set(pd.read_csv(SYN / "data/exclude_pairs.csv").pair_id)
    for bd in map(pathlib.Path, batches):
        m = pd.read_csv(bd / "synth_manifest.csv")
        a = m[m.qc & ~m.pair_id.isin(drop)].copy(); a["batch"] = bd.name
        print(f"{bd}: {m[m.qc].pair_id.nunique()} accepted pairs, {m[m.qc].pair_id.isin(drop).sum() // 2} excluded (exclude_pairs.csv)")
        for c in a["clip"]:
            for sub, ext in (("clips", "mp4"), ("labels", "json")):
                dst = out / sub / f"{c}.{ext}"
                if not dst.exists():
                    os.link(bd / sub / f"{c}.{ext}", dst)
        parts.append(a)
    A = pd.concat(parts, ignore_index=True)
    assert A["clip"].is_unique and (A.groupby("pair_id").y.agg(["min", "max", "size"]) == [0, 1, 2]).all().all()
    A[COLS + ["batch"]].to_csv(out / "manifest.csv", index=False)
    A[COLS + ["batch"]].to_csv(SYN / f"{out.name}_manifest.csv", index=False)
    print(f"{out}: {len(A)} clips, {A.pair_id.nunique()} pairs, labels {A.y.value_counts().to_dict()}, "
          f"families {A[A.y == 1].family.value_counts().to_dict()}, scenes {A.scene_id.nunique()}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
