"""Stage-5 check: members of a group are near-identical before the divergence time and differ after it.
python synth/scripts/check_pairs.py [--data-dir BATCH_DIR=synth/data]   (groups rendered with --keep-renders)
Per negative member with divergence > 0: mean |RGBA difference| (8-bit units) between its render passes and the
positive's, per frame; pass if every frame before divergence (minus one frame of motion blur) is < 0.5 (the spec's
0.5/255) and after it some frame has >= 50 pixels differing by > 20 (only an object/actor-sized region changes). Also reported on the clean composites (look and noise are shared per group)."""
import json, pathlib, sys
import cv2, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "gen"))
import degrade

D = pathlib.Path(sys.argv[sys.argv.index("--data-dir") + 1] if "--data-dir" in sys.argv else
                 pathlib.Path(__file__).resolve().parents[1] / "data")


def rgba(rd, k):
    f = rd / f"rgba_{k:04d}.png"
    return cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.float32) if f.exists() else None


def main():
    rows = []
    for sc in sorted((D / "scenarios").glob("*_leave.json")):
        gid = sc.name.split("_")[0]
        for neg in sorted((D / "scenarios").glob(f"{gid}_*.json")):
            m = json.loads(neg.read_text())["member"]
            if m["y"] == 1 or not m["divergence"]:
                continue
            g = json.loads(neg.read_text())["group"]; fps = g["render_fps"]
            pa, pb = D / f"renders/{gid}/leave", D / f"renders/{gid}/{m['outcome']}"
            if not (pa / "rgba_0001.png").exists() and not any(pa.glob("rgba_*.png")):
                continue
            kd = int(np.floor(m["divergence"] * fps))  # frame index of the divergence
            before, after = [], []
            for k in range(int(g["duration"] * fps)):
                a, b = rgba(pa, k + 1), rgba(pb, k + 1)
                if a is None and b is None:
                    d, c = 0.0, 0
                else:
                    a = a if a is not None else np.zeros_like(b); b = b if b is not None else np.zeros_like(a)
                    diff = np.abs(a - b)
                    d, c = float(diff.mean()), int((diff.max(2) > 20).sum())
                if k < kd - 1:
                    before.append(d)
                else:
                    after.append(c)
            ca, ta = degrade.decode(D / f"clean/{gid}_leave.mp4")
            cb, tb = degrade.decode(D / f"clean/{gid}_{m['outcome']}.mp4")
            cbef = max(float(np.abs(x.astype(np.float32) - y.astype(np.float32)).mean()) for x, y in
                       zip(ca[:max(1, kd - 1)], cb[:max(1, kd - 1)]))
            ok = max(before or [0]) < 0.5 and max(after or [0]) >= 50
            rows.append((gid, m["outcome"], round(m["divergence"], 2), round(max(before or [0]), 4),
                         round(max(after or [0]), 2), round(cbef, 3), ok))
    for r in rows:
        print("%s %-11s div=%6.2fs  render before: max mean|diff| %.4f  after: max changed px %d  clean before %.3f  %s" %
              (*r[:6], "PASS" if r[6] else "FAIL"))
    print(f"{sum(r[6] for r in rows)}/{len(rows)} pairs pass")


if __name__ == "__main__":
    main()
