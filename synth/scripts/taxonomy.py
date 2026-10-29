"""Stage 1c: per-clip contact sheets for the taxonomy, and the statistics of the filled taxonomy.
python synth/scripts/taxonomy.py sheets   -> synth/data/taxonomy/<video>_{ov,s1,s2}.jpg for all 500 contest clips
   ov: 12 frames evenly over the clip; s1, s2: positives 4 fps over [g-3, g+5), negatives min(1, 32/T) fps over the clip.
python synth/scripts/taxonomy.py stats    -> synth/reports/taxonomy_stats.md from synth/data/taxonomy.csv
Clip list, y, g and decoder T from cache/index.csv (src/data.py); times are PTS seconds."""
import sys, json, pathlib
from types import SimpleNamespace
from multiprocessing import Pool
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from contact_sheet import frames_at, sheet

ROOT = pathlib.Path(__file__).resolve().parents[2]
SYN = ROOT / "synth"
OUT = SYN / "data/taxonomy"


def clips():
    d = pd.read_csv(ROOT / "cache/index.csv")
    meta = []
    for r in d.itertuples():
        m = json.loads((ROOT / f"data/{r.split}/labels/{r.video}.json").read_text())["metadata"]
        meta.append(dict(event_type=m["event_type"].replace(" ", ""), time_of_day=m["time_of_day"],
                         lighting=m["lighting"], resolution=m["resolution"]))
    return pd.concat([d, pd.DataFrame(meta)], axis=1)


def strip_times(r):
    if r.y == 1:
        return [t for t in np.arange(r.g - 3, r.g + 5, 0.25) if 0 <= t < r.T]
    fps = min(1.0, 32 / r.T)
    return list(np.arange(0.5 / fps, r.T, 1 / fps))[:32]


def make(r):
    r = SimpleNamespace(**r)  # attribute access; r.T would transpose a Series
    path = ROOT / f"data/{r.split}/videos/{r.video}.mp4"
    ov = frames_at(path, [r.T * (k + 0.5) / 12 for k in range(12)], 520)
    sheet([f for _, f in ov], [f"{r.video} {t:.2f}s" for t, _ in ov], OUT / f"{r.video}_ov.jpg", cols=3)
    st = frames_at(path, strip_times(r), 390)
    for k in range(0, len(st), 16):
        part = st[k:k + 16]
        sheet([f for _, f in part], [f"{t:.2f}s" for t, _ in part], OUT / f"{r.video}_s{k // 16 + 1}.jpg", cols=4)
    return r.video


def stats():
    """Merge synth/data/taxonomy/rows/A*.json (+ blind onsets B*.json) with the labels -> synth/data/taxonomy.csv
    and synth/reports/taxonomy_stats.md. Low-confidence rows are kept in the CSV but excluded from statistics."""
    rows = []
    for f in sorted((OUT / "rows").glob("A*.json")):
        rows += json.loads(f.read_text())
    t = pd.DataFrame(rows).rename(columns={"clip": "video"})
    c = pd.read_csv(OUT / "clips.csv")
    t = c[["video", "split", "y", "g", "T", "event_type", "time_of_day", "lighting"]].merge(t, on="video")
    t["label"] = np.where(t.y == 1, "true", "false")
    t = t.rename(columns={"video": "clip"})
    cols = ["clip", "label", "family", "object_type", "object_size", "near_bin", "vehicle", "onset_seen_s", "pause_s",
            "walkaway_visible", "confidence", "notes", "face_blur", "n_people", "y", "g", "T", "event_type", "split"]
    t[cols].to_csv(SYN / "data/taxonomy.csv", index=False)
    hi = t[t.confidence == "high"]
    P, N = hi[hi.y == 1], hi[hi.y == 0]
    L = [f"# Taxonomy statistics (stage 1c)", "",
         f"Clips annotated: {len(t)} of 500 ({(t.y == 1).sum()} positive, {(t.y == 0).sum()} negative); "
         f"high confidence: {len(hi)} ({len(P)} positive, {len(N)} negative). Statistics use high-confidence rows.", ""]

    def table(s, title):
        v = s.value_counts()
        L.extend([f"## {title}", "", "| value | n | share |", "|---|---|---|"]
                 + [f"| {k} | {n} | {n / v.sum():.2f} |" for k, n in v.items()] + [""])
    table(P.family, "Positive families"); table(N.family, "Negative families")
    table(P.object_type, "Dumped object types"); table(P.object_size, "Dumped object sizes")
    table(P.vehicle, "Vehicle involved (positives)"); table(N.vehicle, "Vehicle involved (negatives)")
    table(P.near_bin, "Left near a bin (positives)"); table(N.near_bin, "Near a bin (negatives)")
    table(P.walkaway_visible, "Walk-away visible (positives)"); table(t.face_blur, "Face blur (all rows)")
    pp = hi[hi.family == "place_pickup"].pause_s.dropna()
    L += ["## Pause (place_pickup)", "", f"n = {len(pp)}; values: {sorted(pp.round(1).tolist())}", ""]
    on = P.dropna(subset=["onset_seen_s"])
    d = on.g - on.onset_seen_s
    L += ["## Onset offset", "", f"n = {len(d)} high-confidence positives with an onset.",
          f"- g - onset_seen_s: median {d.median():.2f}, mean {d.mean():.2f}, IQR [{d.quantile(.25):.2f}, "
          f"{d.quantile(.75):.2f}], 10-90% [{d.quantile(.1):.2f}, {d.quantile(.9):.2f}]",
          f"- by event type: " + ", ".join(f"{k} {v:.2f} (n={n})" for k, v, n in
                                          zip(*[on.groupby("event_type").apply(lambda x: x.name).values,
                                                (on.g - on.onset_seen_s).groupby(on.event_type).median().values,
                                                on.groupby("event_type").size().values])),
          f"- median(onset_seen_s - g) = {(-d).median():.2f}; fraction of positives with g - 1 < onset <= g: "
          f"{((d >= 0) & (d < 1)).mean():.2f}; with |onset - g| <= 0.5: {(d.abs() <= 0.5).mean():.2f}", ""]
    bf = sorted((OUT / "rows").glob("B*.json"))
    if bf:
        b = pd.DataFrame(sum((json.loads(f.read_text()) for f in bf), [])).rename(columns={"clip": "clip"})
        m = t.merge(b, on="clip", suffixes=("", "_b")).dropna(subset=["onset_seen_s", "onset_seen_s_b"])
        diff = (m.onset_seen_s - m.onset_seen_s_b).abs()
        L += ["## Blind second annotation of the onset (pass B)", "",
              f"n = {len(m)}; |A - B| median {diff.median():.2f} s, 75% {diff.quantile(.75):.2f} s, "
              f"share <= 0.5 s {(diff <= 0.5).mean():.2f}, <= 1 s {(diff <= 1).mean():.2f}", ""]
    (SYN / "reports").mkdir(exist_ok=True)
    (SYN / "reports/taxonomy_stats.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "sheets":
        OUT.mkdir(parents=True, exist_ok=True)
        d = clips()
        d.to_csv(OUT / "clips.csv", index=False)
        with Pool(16) as p:
            for i, v in enumerate(p.imap_unordered(make, d.to_dict("records"), chunksize=1)):
                if i % 50 == 0:
                    print(i, v, flush=True)
    else:
        stats()
