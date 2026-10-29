"""Pair-identity check on a RELEASE (no renders needed): members of every pair must be identical before the
divergence and differ after it. Works on the clean composites (--clean; one H.264 encode at CRF 14) or on the
degraded release clips (default; what the detector sees).
python synth/scripts/check_release_pairs.py synth/data/release_v3 OUT.csv [--clean]
Per pair, frames paired by PTS (grey): before the divergence (minus one frame of motion blur) no frame may have
>= 50 pixels differing by > 20 grey levels (an object- or actor-sized change); after it some frame must. Pairs
whose negative diverges at frame 0 (pre-existing object) must differ in the first frame. Mean |diff| is reported
too: the clean encodes differ by encoder noise only, the degraded ones by the second encode's reaction to it.
Release v3 (2026-10-13): clean 0 of 655 pairs with a pre-divergence change, worst-frame mean |diff| median 0.56
levels; degraded: mean |diff| median 1.5, 648 of 655 pairs with >= 50 changed pixels in some frame (encoder blocks,
flat over time, not larger near the divergence); every pair differs after the divergence; 205 of 206 pre-existing
pairs differ in frame 0 by the 50-pixel criterion."""
import json, pathlib, sys
from multiprocessing import Pool
import av, numpy as np, pandas as pd

R = pathlib.Path(__file__).resolve().parents[2]
SC = {"data": R / "synth/data/scenarios", "batch_s34": R / "synth/data/batch_s34/scenarios",
      "batch_v3r": R / "synth/data/batch_v3r/scenarios"}
CLEAN = {k: v.parent / "clean" for k, v in SC.items()}
REL = pathlib.Path(sys.argv[1]); MODE = "clean" if "--clean" in sys.argv else "degraded"


def frames(path):
    c = av.open(str(path)); tb = c.streams.video[0].time_base; out = {}
    for f in c.decode(video=0):
        out[round(float(f.pts * tb), 3)] = f.to_ndarray(format="gray").astype(np.int16)
    c.close(); return out


def check(args):
    pid, batch, pos_clip, neg_clip, fps = args
    div = json.loads((SC[batch] / f"{neg_clip}.json").read_text())["member"].get("divergence") or 0.0
    d = CLEAN[batch] if MODE == "clean" else REL / "clips"
    A, B = frames(d / f"{pos_clip}.mp4"), frames(d / f"{neg_clip}.mp4")
    ts = sorted(set(A) & set(B)); margin = 1.0 / fps
    def stats(tt):
        md, big = [0.0], [0]
        for t in tt:
            x = np.abs(A[t] - B[t]); md.append(float(x.mean())); big.append(int((x > 20).sum()))
        return max(md), max(big)
    bm, bb = stats([t for t in ts if t < div - margin]); am, ab = stats([t for t in ts if t >= div])
    return dict(pair=pid, batch=batch, divergence=div, same_pts=set(A) == set(B), n_frames=len(ts),
                before_mean_max=bm, before_big_max=bb, after_big_max=ab,
                first_frame_big=int((np.abs(A[ts[0]] - B[ts[0]]) > 20).sum()))


def main():
    m = pd.read_csv(REL / "manifest.csv")
    jobs = [(pid, g.iloc[0].batch, g[g.y == 1].iloc[0]["clip"], g[g.y == 0].iloc[0]["clip"], float(g.iloc[0].frame_rate))
            for pid, g in m.groupby("pair_id")]
    with Pool(8) as p:
        df = pd.DataFrame(p.map(check, jobs, chunksize=4))
    df.to_csv(sys.argv[2], index=False)
    nd, pre = df[df.divergence > 0], df[df.divergence == 0]
    print(f"{MODE}: pairs {len(df)}, with divergence {len(nd)}, pre-existing {len(pre)}; same PTS sets {int(df.same_pts.sum())}")
    print(f"before divergence: pairs with a frame of >= 50 px > 20 levels {int((nd.before_big_max >= 50).sum())}; "
          f"worst-frame mean|diff| median {nd.before_mean_max.median():.3f} p90 {nd.before_mean_max.quantile(.9):.3f} max {nd.before_mean_max.max():.3f}")
    print(f"after divergence: pairs with no such frame {int((nd.after_big_max < 50).sum())}; "
          f"pre-existing pairs differing in frame 0: {int((pre.first_frame_big >= 50).sum())} / {len(pre)}")


if __name__ == "__main__":
    main()
