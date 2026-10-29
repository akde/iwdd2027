"""Stage 6 decision layer (wiki/baseline.md): the integer test.py writes.

to_output(t, delta, T) = max(0, min(floor(t + delta), floor(T))), for scalars or arrays; tools/iwdd_eval.py --int uses it.
Emit iff q >= TAU, then write to_output(t, DELTA, T). TAU, DELTA: tools/operating_point.py --variant VARIANT on
runs/RUN/oof.csv, the shipped run (submission/model/MANIFEST.json), copied into the line below.
History: baseline-v0 runs/s5_cls_inner 0.299, 4.0 (official); constant runs/blind_k13 1.0, 0.0.
"""
import numpy as np

TAU, DELTA, RUN, VARIANT = 0.363, -1.0, "s5_movinet_ft_r_v23_fe5", "e5_fpfn"


def to_output(t, delta, T):
    return np.maximum(0, np.minimum(np.floor(np.asarray(t) + delta), np.floor(T))).astype(int)
