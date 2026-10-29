"""Content box per plate: the part of the frame that is camera image, without black pillarbox/letterbox bars
(DECISIONS.md D31, PLAN_D29_FIXES.md F2).
python synth/scripts/plate_content.py   -> 'content': [x0, y0, x1, y1] (image_size px) in every synth/calib/*.json
                                           (the full frame [0, 0, W, H] when there are no bars), synth/data/plate_content.csv
Rule (bar_check.py's rule made stricter): temporal median of N_FRAMES frames of the plate window at native resolution;
on each side the leading run of lines with mean luma < 12, line p99 < 24 and p99 of the temporal maximum < 32
(uniform and constantly black, so dark night edges are not bars); runs under MIN_BAR px count as 0. The box takes the
median bar widths over a video's plates (rounded up: bars belong to the video). calibrate.py writes the same field
(content_boxes), extend_plates.py copies it with the calib."""
import json, math, os, pathlib, sys
import cv2, numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
SYN = HERE.parent
sys.path.insert(0, str(HERE))
from contact_sheet import frames_at

MIN_BAR, N_FRAMES = 4, 15


def _run(ok):
    k = 0
    while k < len(ok) and ok[k]:
        k += 1
    return k


def bars(gray):
    """Bar widths [left, top, right, bottom] in px of grey frames [N, H, W] of one plate."""
    med, mx = np.median(gray, 0), gray.max(0)
    out = []
    for m, x in ((med.T, mx.T), (med, mx), (med.T[::-1], mx.T[::-1]), (med[::-1], mx[::-1])):
        k = _run((m.mean(1) < 12) & (np.percentile(m, 99, axis=1) < 24) & (np.percentile(x, 99, axis=1) < 32))
        out.append(k if k >= MIN_BAR else 0)
    return out


def plate_bars(path, start, end, size, n=N_FRAMES):
    W, H = size
    fr = frames_at(path, list(np.linspace(start + 0.1, end - 0.1, n)), max(W, H))
    g = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for _, f in fr]).astype(np.float32)
    assert g.shape[1:] == (H, W), (path, g.shape, size)
    return bars(g)


def content_boxes(plates):
    """Rows of plates.csv (plate_id, video, path, start, end, w, h) -> per-plate bar widths (DataFrame L, T, R, B) and
    {plate_id: content box [x0, y0, x1, y1]} from the median widths of the plate's video."""
    b = pd.DataFrame([plate_bars(r.path, r.start, r.end, (int(r.w), int(r.h))) for r in plates.itertuples()],
                     columns=list("LTRB"), index=plates.plate_id)
    b["video"] = plates.video.to_numpy()
    med = b.groupby("video")[list("LTRB")].median().map(lambda v: math.ceil(v) if v >= MIN_BAR else 0)
    box = {}
    for r in plates.itertuples():
        L, T, R, B = med.loc[r.video]
        box[r.plate_id] = [int(L), int(T), int(r.w) - int(R), int(r.h) - int(B)]
    return b, box


def main():
    files = sorted((SYN / "calib").glob("*.json"))
    P = pd.read_csv(SYN / "data/plates.csv").set_index("plate_id").loc[[f.stem for f in files]].reset_index()
    b, box = content_boxes(P)
    rows = []
    for f in files:
        cal = json.loads(f.read_text())
        cal["content"] = box[f.stem]
        tmp = f.with_suffix(".tmp")  # readers of synth/calib/*.json never see a half-written file
        tmp.write_text(json.dumps(cal)); os.replace(tmp, f)
        W, H = cal["image_size"]
        rows.append(dict(plate_id=f.stem, video=cal["video"], w=W, h=H, **{f"bar_{s}": int(b.loc[f.stem, s]) for s in "LTRB"},
                         x0=box[f.stem][0], y0=box[f.stem][1], x1=box[f.stem][2], y1=box[f.stem][3]))
    df = pd.DataFrame(rows)
    df.to_csv(SYN / "data/plate_content.csv", index=False)
    bar = df[(df.x0 > 0) | (df.y0 > 0) | (df.x1 < df.w) | (df.y1 < df.h)]
    print(bar.to_string(index=False))
    print(f"{len(df)} calib JSONs; {len(bar)} plates in {bar.video.nunique()} videos have bars >= {MIN_BAR} px")


if __name__ == "__main__":
    main()
