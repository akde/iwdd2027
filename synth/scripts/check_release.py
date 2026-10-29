"""Consistency check of an assembled release (every clip, not a sample).
python synth/scripts/check_release.py synth/data/release_v2
manifest <-> files (no missing, no extra), label JSON <-> manifest (label string, timestamp rule, ids, fold), folds from
data/folds.csv (MEVA -1), pairs (1 positive + 1 negative sharing plate, duration, avatar, object, rate, size, CRF),
decoder duration T = last PTS + median step (src/data.py rule) vs the label's metadata duration, timestamp < T.
Exit 1 on any failure."""
import json, pathlib, sys
import av, numpy as np, pandas as pd

SYN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYN / "gen"))
from scenario import quantize


def decoder_T(path):
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        ts = sorted(float(p.pts * s.time_base) for p in c.demux(s) if p.pts is not None)
    ts = np.array(ts) - ts[0]
    return ts[-1] + float(np.median(np.diff(ts)))


def main(rel):
    rel = pathlib.Path(rel)
    m = pd.read_csv(rel / "manifest.csv")
    folds = pd.read_csv(SYN.parent / "data/folds.csv").set_index("video").fold
    plates = pd.read_csv(SYN / "data/plates.csv", dtype={"label": str}).set_index("plate_id")
    bad = []
    for sub, ext in (("clips", "mp4"), ("labels", "json")):
        have = {p.stem for p in (rel / sub).glob(f"*.{ext}")}
        bad += [f"missing {sub}/{c}" for c in set(m["clip"]) - have] + [f"extra {sub}/{c}" for c in have - set(m["clip"])]
    for r in m.itertuples():
        L = json.loads((rel / f"labels/{r.clip}.json").read_text())
        p = plates.loc[r.plate_id]
        fold = -1 if p.source == "meva" else int(folds[p.video])
        ts = quantize(r.t_onset_exact) if r.y == 1 else -1
        T = decoder_T(rel / f"clips/{r.clip}.mp4")
        checks = {"label": L["label"] == ("true" if r.y == 1 else "false"),
                  "timestamp": L["timestamp"] == ts == r.timestamp and isinstance(L["timestamp"], int),
                  "ids": (L["pair_id"], L["plate_id"], L["scene_id"], L["family"]) == (r.pair_id, r.plate_id, r.scene_id, r.family),
                  "video_id": L["metadata"]["video_id"] == f"{r.clip}.mp4",
                  "event_type": (L["metadata"]["event_type"] == "normal") == (r.y == 0),
                  "fold": L["fold"] == r.fold == fold and p.scene_id == r.scene_id,
                  "duration": abs(T - L["metadata"]["duration"]) <= 1.0 / L["metadata"]["frame_rate"] + 0.05,
                  "ts<T": r.y == 0 or L["timestamp"] < T}
        bad += [f"{r.clip}: {k}" for k, ok in checks.items() if not ok]
    same = ["plate_id", "duration", "avatar", "object", "frame_rate", "resolution", "crf", "fold"]
    for pid, g in m.groupby("pair_id"):
        if sorted(g.y) != [0, 1] or (g[same].nunique() > 1).any():
            bad.append(f"pair {pid}: members differ beyond the outcome")
    print(f"{rel}: {len(m)} clips, {m.pair_id.nunique()} pairs checked; {len(bad)} failures")
    print("\n".join(bad[:50]))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1])
