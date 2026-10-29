"""Frame grids for visual checks. Times are from PTS (seconds from the first frame).
frames_at(path, times, long_side) -> list of (t_pts, RGB uint8) taking, for each t, the last frame with pts <= t.
sheet(images, captions, out, cols) writes a JPG grid with a caption on each tile.
CLI: python synth/scripts/contact_sheet.py VIDEO OUT.jpg [N=12]   (N frames evenly over the clip)"""
import sys
import av, cv2, numpy as np


def frames_at(path, times, long_side=480):
    times = sorted(times)
    out, prev, p0, i = [], None, None, 0

    def take(f, t):
        k = long_side / max(f.width, f.height)
        img = f.reformat(width=round(f.width * k) // 2 * 2, height=round(f.height * k) // 2 * 2, format="rgb24",
                         interpolation="AREA").to_ndarray()
        return (t, img)

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        if times and times[0] > 4.0:  # time zero = first decoded frame's PTS, then seek near the first target
            p0 = next(f.pts for f in c.decode(s) if f.pts is not None)
            c.seek(int((times[0] - 3.0) / s.time_base) + p0, stream=s, backward=True)
        for f in c.decode(s):
            if f.pts is None or (prev is not None and f.pts <= prev[0].pts):
                continue
            p0 = f.pts if p0 is None else p0
            t = float((f.pts - p0) * s.time_base)
            while i < len(times) and times[i] < t and prev is not None:
                out.append(take(*prev)); i += 1
            prev = (f, t)
            if i == len(times):
                break
        while i < len(times) and prev is not None:
            out.append(take(*prev)); i += 1
    return out


def duration(path):
    """Last PTS + median step (as src/data.py)."""
    ts, p0 = [], None
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        for pkt in c.demux(s):
            if pkt.pts is not None:
                p0 = pkt.pts if p0 is None else min(p0, pkt.pts)
                ts.append(pkt.pts)
        ts = np.unique(np.array(ts, dtype=np.int64))
        ts = (ts - ts[0]) * float(s.time_base)
    return float(ts[-1] + (np.median(np.diff(ts)) if len(ts) > 1 else 0.0))


def sheet(images, captions, out, cols=4, scale=1.0):
    h = max(i.shape[0] for i in images); w = max(i.shape[1] for i in images)
    rows = -(-len(images) // cols)
    g = np.full((rows * (h + 4), cols * (w + 4), 3), 255, np.uint8)
    for k, (img, cap) in enumerate(zip(images, captions)):
        r, c = divmod(k, cols)
        y, x = r * (h + 4), c * (w + 4)
        g[y:y + img.shape[0], x:x + img.shape[1]] = img
        cv2.rectangle(g, (x, y), (x + 9 * len(cap) + 6, y + 20), (0, 0, 0), -1)
        cv2.putText(g, cap, (x + 3, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    if scale != 1.0:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out), cv2.cvtColor(g, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    path, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    T = duration(path)
    fr = frames_at(path, [T * (k + 0.5) / n for k in range(n)])
    sheet([f for _, f in fr], [f"{t:.2f}s" for t, _ in fr], out)
