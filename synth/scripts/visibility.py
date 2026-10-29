"""Actor visibility during the event: mean |composite - plate| luma (8-bit) inside the actor's box, over the frames
from t_event to t_release, on the clean composite of the positive member (pairs share it until divergence).
python synth/scripts/visibility.py [BATCH_DIR=synth/data] -> BATCH_DIR/visibility.csv (pair_id, vis, luma_box)"""
import json, pathlib, sys
import cv2, numpy as np, pandas as pd
GEN = pathlib.Path(__file__).resolve().parents[1] / "gen"
sys.path.insert(0, str(GEN))
import composite, degrade, qc

SYN = GEN.parent


def visibility(bd, gid, plates):
    S = json.loads((bd / f"scenarios/{gid}_leave.json").read_text()); g, m = S["group"], S["member"]
    mk = np.load(bd / f"renders/{gid}/leave/masks.npz")
    fps = g["render_fps"]; ev = m["events"]
    ks = [k for k in range(int(np.floor(ev["t_event"] * fps)), int(np.ceil(ev["t_release"] * fps)) + 1)
          if k < len(mk["act_box"]) and mk["act_box"][k][0] >= 0]
    if not ks:
        return 0.0, 0.0
    fr, _ = degrade.decode(bd / f"clean/{gid}_leave.mp4")
    pl = composite.plate_frames(plates.loc[g["plate_id"], "path"], g["plate_start"], [k / fps for k in ks],
                                g["render_size"])
    lum = [cv2.cvtColor(fr[k], cv2.COLOR_RGB2GRAY)[mk["act_box"][k][1]:mk["act_box"][k][3] + 1,
                                                    mk["act_box"][k][0]:mk["act_box"][k][2] + 1].mean() for k in ks]
    return qc.visibility(fr, pl, mk["act_box"], ks), float(np.median(lum))


if __name__ == "__main__":
    bd = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else SYN / "data")
    plates = pd.read_csv(SYN / "data/plates.csv").set_index("plate_id")
    man = pd.read_csv(bd / "synth_manifest.csv")
    rows = []
    for gid in sorted(man[man.qc].pair_id.unique()):
        v, l = visibility(bd, gid, plates)
        rows.append(dict(pair_id=gid, vis=round(v, 2), luma_box=round(l, 1)))
    df = pd.DataFrame(rows); df.to_csv(bd / "visibility.csv", index=False)
    print(df.describe().round(2).to_string()); print(df.sort_values("vis").head(15).to_string(index=False))
