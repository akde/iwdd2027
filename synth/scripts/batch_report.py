"""Stage-6/7 checks on BATCH_DIR/synth_manifest.csv -> OUT/batch_report.md (+ contact sheets OUT/batch_sheet_<k>.jpg).
python synth/scripts/batch_report.py [--sheet-frac 0.05] [--seed 0] [--data-dir BATCH_DIR=synth/data] [--out OUT]
OUT defaults to BATCH_DIR/reports, so only an explicit --out synth/reports writes the report of batches
s15-s17 (release v2; D38, review R3/R4); old batch_sheet_*.jpg in OUT are removed first.
Reject rate and reasons (stage 6); accepted clips: label balance, day/night/IR, length and onset distributions per
label, objects and avatars per label (invariant 7), clips per fold and scene; a contact sheet of a random fraction
of accepted clips (degraded clips; 3 frames: before the event, after the release, the end)."""
import argparse, json, pathlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from contact_sheet import frames_at, sheet

SYN = pathlib.Path(__file__).resolve().parents[1]
D = SYN / "data"


def reject_reasons(m):
    """Reject reasons over the members (a group fails if any member fails) -> value counts; empty without rejections
    (an all-empty qc_why column is read as float)."""
    why = m[~m.qc].qc_why.astype(object).fillna("").astype(str)
    return why.replace("", "(other member)").str.split(";").explode().value_counts()


def main(a):
    D = pathlib.Path(a.data_dir)
    out_dir = pathlib.Path(a.out) if getattr(a, "out", None) else D / "reports"
    m = pd.read_csv(D / "synth_manifest.csv", dtype={"qc_why": str})
    sc = {r.clip: json.loads((D / f"scenarios/{r.clip}.json").read_text()) for r in m.itertuples()}
    m["ir"] = [sc[c]["group"]["light"]["ir"] for c in m["clip"]]
    g = m.groupby("pair_id").qc.first()
    L = ["# Synthetic batch report", "", f"Groups: {len(g)}; accepted {int(g.sum())} ({g.mean():.0%}); clips accepted "
         f"{int(m.qc.sum())} of {len(m)}.", "", "## Reject reasons (members; a group fails if any member fails)", ""]
    why = reject_reasons(m)
    L += [f"- {k}: {v}" for k, v in why.items()] + [""]
    A = m[m.qc].copy()
    L += ["## Accepted clips", "", f"- labels: {A.y.value_counts().to_dict()}"]
    for col in ("time_of_day", "lighting", "ir", "source"):
        t = pd.crosstab(A[col], A.y, normalize="columns").round(3)
        L.append(f"- {col} share by label (columns y=0, y=1): " + "; ".join(f"{i}: {r[0]:.3f} / {r[1]:.3f}"
                                                                            for i, r in t.iterrows()))
    q = lambda s: np.percentile(s, [10, 25, 50, 75, 90]).round(1).tolist()
    L += [f"- duration quantiles 10/25/50/75/90: y=0 {q(A[A.y == 0].duration)}, y=1 {q(A[A.y == 1].duration)}",
          f"- frame rate quantiles: y=0 {q(A[A.y == 0].frame_rate)}, y=1 {q(A[A.y == 1].frame_rate)}",
          f"- positive timestamps: quantiles {q(A[A.y == 1].timestamp)}; t_dec available "
          f"{A[A.y == 1].t_dec.notna().mean():.2f}"]
    ob = pd.crosstab(A.object, A.y); av = pd.crosstab(A.avatar, A.y)
    L += [f"- objects: max |n(y=1) - n(y=0)| per object = {int((ob[1] - ob[0]).abs().max())} "
          f"(invariant 7; {len(ob)} objects)",
          f"- avatars: max |n(y=1) - n(y=0)| per avatar = {int((av[1] - av[0]).abs().max())} ({len(av)} avatars)",
          f"- negative families: {A[A.y == 0].family.value_counts().to_dict()}",
          f"- folds (accepted clips): {A.fold.value_counts().sort_index().to_dict()}",
          f"- scenes: {A.scene_id.nunique()}; plates: {A.plate_id.nunique()}; max uses of a plate: "
          f"{A.groupby('plate_id').pair_id.nunique().max()} groups", ""]
    rng = np.random.default_rng(a.seed)
    pick = A[A.y == 1].pair_id.drop_duplicates()
    pick = pick.sample(max(1, int(round(a.sheet_frac * len(pick)))), random_state=a.seed).tolist()
    ims, caps = [], []
    for pid in pick:
        for c in sorted(A[A.pair_id == pid]["clip"], key=lambda c: "leave" not in c):
            mm = sc[c]["member"]; ev = mm["events"]; dur = float(A[A["clip"] == c].duration.iloc[0])
            ts = [max(0.0, ev["t_event"] - 1.0), min(dur - 0.2, ev.get("t_stand", ev["t_event"]) + 1.0), dur - 0.2]
            for t, f in frames_at(D / f"clips/{c}.mp4", ts, 480):
                ims.append(f); caps.append(f"{c[-16:]} {t:.1f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("batch_sheet_*.jpg"):  # sheets of an earlier, larger report must not stay next to this one
        f.unlink()
    out = []
    for k in range(0, len(ims), 24):
        p = out_dir / f"batch_sheet_{k // 24}.jpg"; sheet(ims[k:k + 24], caps[k:k + 24], p, cols=6); out.append(p.name)
    L += [f"Contact sheets ({len(pick)} accepted groups = {a.sheet_frac:.0%}, positive row then negative row, "
          f"degraded clips): {', '.join(out)}", ""]
    (out_dir / "batch_report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=str(D))
    ap.add_argument("--out", help="report directory (default BATCH_DIR/reports)")
    main(ap.parse_args())
