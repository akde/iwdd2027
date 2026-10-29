"""Moving foreground per usable plate (F5, DECISIONS.md D32): what YOLO's 2-fps traffic boxes miss (a car cut by the
frame edge at the plate start, a reversing car's bottom rows, a person at a departing car's tailgate).
python synth/scripts/plate_motion.py [--workers 6] -> synth/data/motion/<plate_id>.npz: t (s, plate PTS), fg (packbits
along the last axis of [T, h, w] bool), shape, thr, mad. CPU only.
Every plate that passed the stage-2 check (calib_check.csv ok), sampled at FPS on the PTS grid (contact_sheet.frames_at,
long side LONG), grey + Gaussian sigma 1, background = temporal median over the plate window; foreground = |I - B| >
max(MIN_THR, 6 x 1.4826 x median |I - B|), opened 3x3. qc.motion_regions keeps only blobs attributed to a vehicle."""
import argparse, pathlib, sys
import concurrent.futures as cf
import cv2, numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from contact_sheet import frames_at

SYN = pathlib.Path(__file__).resolve().parents[1]
OUT = SYN / "data/motion"
FPS, LONG, MIN_THR = 3.0, 480, 18.0


def foreground(gray):
    """[T, h, w] uint8 grey frames -> bool foreground masks, threshold, MAD."""
    g = np.stack([cv2.GaussianBlur(x, (0, 0), 1.0) for x in gray]).astype(np.int16)
    D = np.abs(g - np.median(g, 0).astype(np.int16)[None])
    mad = float(np.median(D))
    thr = max(MIN_THR, 6 * 1.4826 * mad)
    k = np.ones((3, 3), np.uint8)
    return np.stack([cv2.morphologyEx((d > thr).astype(np.uint8), cv2.MORPH_OPEN, k) for d in D]).astype(bool), thr, mad


def one(plate_id, path, start, end):
    f = OUT / f"{plate_id}.npz"
    if f.exists():
        return plate_id, None
    fr = frames_at(path, list(np.arange(start, end, 1 / FPS)), LONG)
    M, thr, mad = foreground([cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for _, x in fr])
    np.savez_compressed(f, t=np.array([t for t, _ in fr]), fg=np.packbits(M, axis=-1), shape=np.array(M.shape),
                        thr=thr, mad=mad)
    return plate_id, f"{len(fr)} frames thr {thr:.1f} fg {M.mean():.4f}"


def load(plate_id):
    """(t, fg [T, h, w] bool) of a plate, or None if it has no motion file."""
    f = OUT / f"{plate_id}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z["t"], np.unpackbits(z["fg"], axis=-1)[..., :int(z["shape"][2])].astype(bool)


def main(a):
    P = pd.read_csv(SYN / "data/plates.csv")
    chk = pd.read_csv(SYN / "data/calib_check.csv")
    P = P[P.plate_id.isin(chk[chk.ok].plate_id)]
    OUT.mkdir(parents=True, exist_ok=True)
    with cf.ProcessPoolExecutor(a.workers) as ex:
        for fu in cf.as_completed([ex.submit(one, r.plate_id, r.path, r.start, r.end) for r in P.itertuples()]):
            pid, msg = fu.result()
            if msg:
                print(pid, msg, flush=True)
    print(f"{sum((OUT / f'{p}.npz').exists() for p in P.plate_id)} of {len(P)} plates have motion files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    main(ap.parse_args())
