"""Stage-1 check: N random plates (seed), 3 frames each (start, middle, end), as contact sheets of 5 plates.
python synth/scripts/check_plates.py [N=20] [SEED=0] -> synth/data/checks/plates_<k>.jpg"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from contact_sheet import frames_at, sheet

SYN = pathlib.Path(__file__).resolve().parents[1]
n, seed = (int(sys.argv[1]) if len(sys.argv) > 1 else 20), (int(sys.argv[2]) if len(sys.argv) > 2 else 0)
p = pd.read_csv(SYN / "data/plates.csv", dtype={"label": str}).sample(n, random_state=seed)
(SYN / "data/checks").mkdir(parents=True, exist_ok=True)
imgs, caps = [], []
for r in p.itertuples():
    for t in (r.start + 0.05, (r.start + r.end) / 2, r.end - 0.1):
        tt, f = frames_at(r.path, [t], 480)[0]
        imgs.append(f); caps.append(f"{r.plate_id} {tt:.1f}s")
for k in range(0, len(imgs), 15):
    sheet(imgs[k:k + 15], caps[k:k + 15], SYN / f"data/checks/plates_{k // 15}.jpg", cols=3)
print(p[["plate_id", "dur", "source", "scene_id"]].to_string(index=False))
