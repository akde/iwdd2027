"""Pinhole camera from a calib JSON (synth/calib/<plate_id>.json); pure numpy.
camera_world is a Blender camera matrix (looks along local -Z, local +Y up); world Z up, metres, ground z = 0.
Pixel coordinates are in the calib's image_size frame unless `size` is given (e.g. the render size).
The camera image is the calib's content box (synth/scripts/plate_content.py): black pillarbox/letterbox bars of the
plate lie outside it and count as outside the image (visible, in_image, edge; the compositor masks them).
A scenario group carries a snapshot of its plate's camera taken at sampling time (gr['cam']), so a later
recalibration or a new mask cannot change how an old scenario is re-read; group_calib() is the one way to get it."""
import json, pathlib
import numpy as np

CALIB = pathlib.Path(__file__).resolve().parents[1] / "calib"
SNAPSHOT = ("camera_world", "focal_px", "image_size", "render_size", "pitch_deg", "height_m", "walkable", "content")


def plate_calib(plate_id):
    return json.loads((CALIB / f"{plate_id}.json").read_text())


def snapshot(cal):
    """The camera fields of a calib that a scenario group stores (gr['cam']); 'content' only if the calib has it.
    Not the real-person fit_boxes (up to 0.4 MB; only the informative YOLO height ratio reads them)."""
    return {k: cal[k] for k in SNAPSHOT if k in cal}


def group_calib(gr):
    """The camera a scenario group was sampled with: its snapshot, else (scenarios written before 2026-10-07) the
    plate's current calib JSON."""
    return gr["cam"] if "cam" in gr else plate_calib(gr["plate_id"])


class Camera:
    def __init__(self, calib, size=None):
        self.M = np.array(calib["camera_world"], float)
        W, H = calib["image_size"]
        self.W, self.H = (size if size is not None else (W, H))
        self.f = calib["focal_px"] * self.W / W
        self.R, self.C = self.M[:3, :3], self.M[:3, 3]
        x0, y0, x1, y1 = calib.get("content", (0, 0, W, H))  # without the key: the full frame
        self.box = np.array([x0 * self.W / W, y0 * self.H / H, x1 * self.W / W, y1 * self.H / H], float)
        self.bars = np.array([x0 > 0, y0 > 0, x1 < W, y1 < H])  # left, top, right, bottom side of the box is a bar

    def project(self, P):
        """World points [N,3] -> pixels [N,2], depth [N] (> 0 in front of the camera)."""
        P = np.atleast_2d(np.asarray(P, float))
        c = (P - self.C) @ self.R  # camera coords: x right, y up, z backward
        z = -c[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = np.stack([self.W / 2 + self.f * c[:, 0] / z, self.H / 2 - self.f * c[:, 1] / z], 1)
        return uv, z

    def ground(self, uv):
        """Pixels [N,2] -> ground points [N,2] (nan above the horizon)."""
        uv = np.atleast_2d(np.asarray(uv, float))
        d = np.stack([(uv[:, 0] - self.W / 2) / self.f, -(uv[:, 1] - self.H / 2) / self.f, -np.ones(len(uv))], 1) @ self.R.T
        t = np.where(d[:, 2] < -1e-9, -self.C[2] / np.where(d[:, 2] < -1e-9, d[:, 2], -1), np.nan)
        return (self.C[None, :2] + t[:, None] * d[:, :2])

    def person_px(self, xy, height=1.7):
        """Pixel foot and head points of a standing person of `height` m at ground points xy [N,2]."""
        xy = np.atleast_2d(np.asarray(xy, float))
        foot, zf = self.project(np.c_[xy, np.zeros(len(xy))])
        head, zh = self.project(np.c_[xy, np.full(len(xy), height)])
        return foot, head, np.minimum(zf, zh)

    def visible(self, xy, height=1.8, margin=0.03):
        """Whole standing person inside the image (the content box, with a margin as a fraction of its size) and in
        front."""
        foot, head, z = self.person_px(xy, height)
        x0, y0, x1, y1 = self.box
        mx, my = margin * (x1 - x0), margin * (y1 - y0)
        return ((z > 0.5) & (foot[:, 0] > x0 + mx) & (foot[:, 0] < x1 - mx) & (foot[:, 1] < y1 - my)
                & (head[:, 1] > y0 + my) & (head[:, 0] > x0 + mx) & (head[:, 0] < x1 - mx))

    def in_image(self, uv, margin=0.0):
        """Pixels [N,2] inside the content box (shrunk by margin px)."""
        uv = np.atleast_2d(np.asarray(uv, float)); x0, y0, x1, y1 = self.box
        return ((uv[:, 0] >= x0 + margin) & (uv[:, 0] <= x1 - margin) & (uv[:, 1] >= y0 + margin)
                & (uv[:, 1] <= y1 - margin))

    def edge(self, uv):
        """Distance of pixels [N,2] to the nearest side of the content box (negative outside) / its larger side."""
        uv = np.atleast_2d(np.asarray(uv, float)); x0, y0, x1, y1 = self.box
        return np.minimum.reduce([uv[:, 0] - x0, x1 - uv[:, 0], y1 - uv[:, 1], uv[:, 1] - y0]) / max(x1 - x0, y1 - y0)


def inside(poly, xy):
    """Even-odd point-in-polygon for ground points xy [N,2]."""
    poly = np.asarray(poly, float); xy = np.atleast_2d(np.asarray(xy, float))
    x, y = xy[:, 0:1], xy[:, 1:2]
    x0, y0 = poly[:, 0][None], poly[:, 1][None]
    x1, y1 = np.roll(poly[:, 0], -1)[None], np.roll(poly[:, 1], -1)[None]
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0)
    return cross.sum(1) % 2 == 1
