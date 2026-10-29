"""Submission-output checker for IWDD 2027: scores a folder of result TXT files.

NOT the authoritative scorer. tools/iwdd_eval.py is the single source of truth for every
reported metric (OOF CSVs, threshold/offset tuning, cross-fitted F1, CIs, subgroups).
Use this file only to check that the final test.py outputs score as expected.
Its default rule matches tools/iwdd_eval.py "official".

Ground truth: one JSON per video (labels/vidXXXX.json, "timestamp" in seconds,
-1 for negatives). Predictions: one TXT per video named like the official
sample, "vidXXXX.mp4.txt", holding the instant in seconds or nothing. A detection is a TP if
g - early <= p <= g + 10 (seconds); any other detection is an FP.

Mistimed detections in positive videos:
- default ("official", contest website wording): FP only.
- --fpfn: FP and also FN, as in the organizers' papers (Greco et al. IMAVIS 2026, Bouwmans et al. WACVW 2026).
Early tolerance: 3 s (website, default); the papers used 5 s (--early 5).

Usage: python src/evaluate.py --gt LABELS_DIR --results RESULTS_DIR [--early 3] [--fpfn] [--verbose]
"""
import argparse
import json
import os

EARLY_TOL = 3.0   # seconds before g still accepted
LATE_MAX = 10.0   # seconds after g still accepted


def read_label(path):
    """Return (instant or None, metadata dict) from a ground-truth JSON."""
    with open(path) as f:
        label = json.load(f)
    g = label["timestamp"] if label["label"] == "true" else None
    return g, label["metadata"]


def read_instant(path):
    """Return the instant stored in a result TXT file, or None if missing/empty."""
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        content = f.read().strip()
    if not content:
        return None
    return float(content.split()[0])


def evaluate(gt_dir, results_dir, early=EARLY_TOL, fpfn=False):
    tp = fp = fn = 0
    delays = []
    per_video = {}
    for name in sorted(os.listdir(gt_dir)):
        if not name.endswith(".json"):
            continue
        g, meta = read_label(os.path.join(gt_dir, name))
        name = meta["video_id"]
        p = read_instant(os.path.join(results_dir, name + ".txt"))
        if p is None:
            outcome = "FN" if g is not None else "TN"
        elif g is None:
            outcome = "FP"
        elif g - early <= p <= g + LATE_MAX:
            outcome = "TP"
            delays.append(abs(p - g))
        else:
            outcome = "FP+FN" if fpfn else "FP"
        tp += outcome == "TP"
        fp += outcome in ("FP", "FP+FN")
        fn += outcome in ("FN", "FP+FN")
        per_video[name] = (g, p, outcome)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_delay = sum(delays) / len(delays) if delays else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1,
                mean_delay=mean_delay, per_video=per_video)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="folder of ground-truth JSON labels")
    ap.add_argument("--results", required=True, help="folder of predicted TXT files")
    ap.add_argument("--early", type=float, default=EARLY_TOL, help="early tolerance in seconds")
    ap.add_argument("--fpfn", action="store_true",
                    help="count a mistimed detection on a positive as FP and FN (organizers' papers)")
    ap.add_argument("--verbose", action="store_true", help="print per-video outcomes")
    args = ap.parse_args()

    r = evaluate(args.gt, args.results, early=args.early, fpfn=args.fpfn)
    if args.verbose:
        for name, (g, p, outcome) in r["per_video"].items():
            print(f"{name}\tg={g}\tp={p}\t{outcome}")
    print(f"TP={r['tp']} FP={r['fp']} FN={r['fn']}")
    print(f"Precision={r['precision']:.4f} Recall={r['recall']:.4f} F1={r['f1']:.4f} "
          f"MeanDelay={r['mean_delay']:.2f}s")


if __name__ == "__main__":
    main()
