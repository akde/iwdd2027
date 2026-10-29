"""Contact sheet of minimal pairs: per group, positive and negative rows at the same key times, cropped around
the spot (clean composites, or the degraded clips with --clips).
python synth/scripts/pair_sheet.py OUT.jpg GROUP [GROUP ...] [--clips] [--data-dir BATCH_DIR=synth/data]"""
import sys, json, pathlib
import cv2, numpy as np
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parent), str(pathlib.Path(__file__).resolve().parents[1] / "gen")]
from contact_sheet import frames_at, sheet
from camera import Camera, group_calib

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


def main(out, gids, clips, D=DATA):
    ims, caps = [], []
    for gid in gids:
        mem = sorted(D.glob(f"scenarios/{gid}_*.json"), key=lambda p: "leave" not in p.name)
        pos = json.loads(mem[0].read_text()); g, m = pos["group"], pos["member"]
        ev = m["events"]
        ts = [ev["t_arrive"] - 1.0, ev["t_event"] + 0.4, ev["t_release"], ev["t_stand"] + 1.0,
              (m["t_dec"] or g["duration"]) , g["duration"] - 0.15]
        ts = [float(np.clip(t, 0, g["duration"] - 0.15)) for t in ts]
        cal = group_calib(g)
        W, H = g["render_size"]
        (u, v), = Camera(cal, (W, H)).project(np.array([[*g["path"]["X"], 0.0]]))[0]
        for p in mem:
            name = p.stem
            vid = D / ("clips" if clips else "clean") / f"{name}.mp4"
            fr = frames_at(vid, ts, max(W, H) if not clips else 4096)
            for t, f in fr:
                k = f.shape[1] / W
                cu, cv = int(u * k), int(v * k); s = int(0.22 * f.shape[0])
                crop = f[max(0, cv - 2 * s):cv + s, max(0, cu - 2 * s):cu + 2 * s]
                ims.append(cv2.resize(crop, (360, 270), interpolation=cv2.INTER_AREA)); caps.append(f"{name[-14:]} {t:.1f}")
    sheet(ims, caps, out, cols=6)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--clips"]
    kw = {}
    if "--data-dir" in a:
        i = a.index("--data-dir"); kw["D"] = pathlib.Path(a[i + 1]); del a[i:i + 2]
    main(a[0], a[1:], "--clips" in sys.argv, **kw)
