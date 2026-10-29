"""Walker ground tracks (pure numpy; also imported inside Blender): the one model of where every walker's root is,
shared by rendering (build_scene, leg_overlap), sampling (scenario) and the pre-render screen (postqc).
Root keys are (t, x, y, yaw), piecewise linear and held constant outside the keys (motion.root_track)."""
import math
import numpy as np


def walker_keys(path, t0, speed):
    """Root keys of a distractor walking straight from E = path[0] (at t0) to B = path[1] at `speed`, the line
    continued 60 s before E and after B."""
    E, B = np.array(path[0], float), np.array(path[1], float)
    u = (B - E) / np.linalg.norm(B - E); a = math.atan2(u[1], u[0])
    tb = t0 + np.linalg.norm(B - E) / speed
    return [(t0 - 60, *(E - u * speed * 60), a), (t0, *E, a), (tb, *B, a), (tb + 60, *(B + u * speed * 60), a)]


def track(keys, ts):
    """Ground positions [T,2] of root keys at times ts."""
    k = np.array(keys, float)
    return np.c_[np.interp(ts, k[:, 0], k[:, 1]), np.interp(ts, k[:, 0], k[:, 2])]


def walkers(gr, m, ts=None):
    """Ground tracks [T,2] at times ts (default every 0.1 s over the clip): member m's actor, then the group's
    distractors."""
    ts = np.arange(0, gr["duration"], 0.1) if ts is None else np.asarray(ts, float)
    return [track(m["root_keys"], ts)] + [track(walker_keys(D["path"], D["t0"], D["speed"]), ts)
                                          for D in gr["distractors"]]


def heading(keys, ts):
    """Facing angle (rad, 0 = +x) at times ts: the keys' yaw unwrapped, linear in between (motion.root_track)."""
    k = np.array(keys, float)
    return np.interp(ts, k[:, 0], np.unwrap(k[:, 3]))


def moving(keys, ts, settle=0.35):
    """The root moves at times ts or moved within the previous `settle` s (piecewise linear keys; held outside them):
    the walking envelope of a carried object (structures.carry_box, CARRY_SETTLE)."""
    k = np.array(keys, float); k = k[np.argsort(k[:, 0], kind="stable")]
    ts = np.asarray(ts, float); out = np.zeros(len(ts), bool)
    for (t0, x0, y0), (t1, x1, y1) in zip(k[:-1, :3], k[1:, :3]):
        if t1 > t0 and math.hypot(x1 - x0, y1 - y0) > 1e-9:
            out |= (ts >= t0) & (ts < t1 + settle)
    return out
