"""IWDD 2027 submission (wiki/plan.md section 8; wiki/experiments.md E3r, E3s): python test.py --videos V --results R

POLICY = "model": the run in submission/model/MANIFEST.json (src/decision.py RUN). Per clip: PyAV decode at 5 fps
(src/data.py; times from PTS, T = last PTS + median step) -> RGB, short side 224 (src/features.py prep_rgb) -> frozen
MoViNet-A2 trunk in fp16, one 4 s window per forward, centred every 1 s (src/finetune.py trunk_out) -> each of the 15
fine-tuned tails + TCN heads (5 folds x 3 seeds, submission/model/f<k>_s<s>.pt), each sequence minus its own median
(finetune.head_in, form m): frame logits at 5 Hz -> L_k = the mean of fold k's 3 seeds -> q = sigmoid(mean over k of
video_logit(L_k)), t = time of the max of the mean of the L_k (src/head.py) -> if q >= TAU one integer
to_output(t, DELTA, T) with no newline, else an empty file (src/decision.py, operating point of RUN under VARIANT).
--folds data/folds.csv: per-fold mode (tests/test_package.py): q, t = score(L_k) of the clip's own fold = RUN's oof.csv.
E3f (wiki/experiments.md): when MANIFEST.json holds "full", full mode uses its 15 refit models instead (g<k>_s<s>.pt,
5 groups x 3 seeds, each trained on every clip; the same formula with L_k = the mean of group k's 3 seeds); RUN is then
the fixed-epoch CV run whose OOF gives TAU, DELTA and whose 15 fold models serve the per-fold mode.
POLICY = "constant", no CUDA device (the model on the CPU runs 30-40x real time) or a model that fails to load: every
clip writes to_output(13.0, 0.0, T) = floor(min(13, T)). A per-clip error writes the constant when a duration is
readable (decoder T, else the container's duration), else an empty TXT (p must stay <= the duration).
Missing pinned packages (av, einops; requirements.txt) are installed first. Writes R/<file name>.txt for every file in V.
"""
import argparse, csv, hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for mod, pin in (("av", "av==19.0.0"), ("einops", "einops==0.8.2")):  # if the notebook's pip cell was skipped
    if importlib.util.find_spec(mod) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pin], check=False)

import av, cv2, numpy as np, torch

sys.path.insert(0, str(ROOT / "src"))
from data import decode
from decision import DELTA, RUN, TAU, VARIANT, to_output
from head import Head, full_score, score  # full_score: shared with src/finetune.py full (E3f)

FOLDS, SEEDS, D, W_MAX = range(5), range(3), 640, 240  # W_MAX: windows above which the tail runs in chunks of 64
MODEL = ROOT / "submission" / "model"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
POLICY, T_CONST = "model", 13.0  # "constant" | "model"


def constant(T):  # the fallback: floor(min(13, T)) through the decision layer's to_output (delta 0)
    return str(int(to_output(T_CONST, 0.0, T)))


def container_T(path):  # duration header of an undecodable file, or None
    try:
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            d = s.duration * s.time_base if s.duration else (c.duration / 1e6 if c.duration else None)
            return float(d) if d and d > 0 else None
    except Exception:
        return None


def load_models(full=False):  # -> finetune module, frozen trunk, group -> [(tail, head)] in seed order
    import finetune as E
    from features import load_movinet  # checks the MoViNet weights' SHA256
    man = json.loads((MODEL / "MANIFEST.json").read_text())
    assert man["run"] == RUN, (man["run"], RUN)
    files = {**man["files"], **man.get("full", {}).get("files", {})}
    for n, h in files.items():  # every listed file, whichever models this mode loads
        if hashlib.sha256((MODEL / n).read_bytes()).hexdigest() != h:
            raise ValueError(f"{n}: SHA256 mismatch")
    pre = "g" if full and "full" in man else "f"  # E3f refit groups in full mode, else RUN's fold models
    print(f"models: {pre}<k>_s<s>.pt of {man['full']['run'] if pre == 'g' else RUN}", flush=True)
    base = load_movinet(DEV)
    trunk, models = E.split(base), {}
    for k in FOLDS:
        for s in SEEDS:
            f = MODEL / f"{pre}{k}_s{s}.pt"
            if f.name not in files:
                raise ValueError(f"{f.name}: not in MANIFEST.json")
            ck = torch.load(f, map_location="cpu", weights_only=True)
            tail, head = E.new_tail(base), Head(D, torch.zeros(D), torch.ones(D)).to(DEV)
            tail.load_state_dict(ck["tail"]); head.load_state_dict(ck["head"])
            models.setdefault(k, []).append((tail.eval(), head.eval()))
    return E, trunk, models


def rows(E, tail, h, n):  # finetune.head_in (form m); above W_MAX windows the tail runs in chunks (long clips only)
    if len(h) <= W_MAX:
        return E.head_in(tail, h, n, "m")
    f = E.interp(torch.cat([tail(h[i:i + 64].float()).mean((2, 3, 4)) for i in range(0, len(h), 64)]), n)
    return f - f.quantile(0.5, dim=0)


@torch.no_grad()
def fold_logits(E, trunk, models, path, folds=FOLDS):  # -> {k: L_k [n]} at 5 Hz, times [n], T
    frames, _, T = decode(path, fps=E.MFPS)
    x = E.prep_rgb(frames)
    h = E.trunk_out(trunk, E.to_x(x), wb=1)
    lg = {k: np.mean([hd(rows(E, tl, h, len(x))).cpu().numpy() for tl, hd in models[k]], 0) for k in folds}
    return lg, np.arange(len(x)) / E.MFPS, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--folds", help="data/folds.csv: per-fold mode (tests/test_package.py only)")
    a = ap.parse_args()
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False  # as in training
    print(f"torch {torch.__version__}, av {av.__version__}, cv2 {cv2.__version__}, numpy {np.__version__}, {DEV}; "
          f"run {RUN}, TAU {TAU}, DELTA {DELTA} ({VARIANT})", flush=True)
    os.makedirs(a.results, exist_ok=True)
    policy, model = POLICY, None
    if policy == "model" and DEV != "cuda":
        print("WARNING: no CUDA device; the model needs a GPU runtime. Constant for every clip.", file=sys.stderr,
              flush=True)
        policy = "constant"
    if policy == "model":
        try:
            model = load_models(full=not a.folds)
        except Exception as e:  # e.g. a SHA256 mismatch or a missing file: the constant for every clip
            print(f"model load failed, constant for every clip: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            policy = "constant"
    failed = 0
    fold = {r["video"]: int(r["fold"]) for r in csv.DictReader(open(a.folds))} if a.folds else None
    for name in sorted(os.listdir(a.videos)):
        path = os.path.join(a.videos, name)
        if not os.path.isfile(path):
            continue
        out, info, t0 = "", "", time.time()
        if DEV == "cuda":
            torch.cuda.reset_peak_memory_stats()
        T = None
        try:
            if policy == "constant":
                T = decode(path)[2]
                out, info = constant(T), f"constant, T {T:.2f}, "
            else:
                lg, t5, T = fold_logits(*model, path, [fold[Path(name).stem]] if fold else FOLDS)
                q, t = score(next(iter(lg.values())), t5) if fold else full_score(lg, t5)
                out = str(int(to_output(t, DELTA, T))) if q >= TAU else ""
                info = f"q {q:.4f}, t {t}, T {T:.2f}, "
        except Exception as e:  # the constant if a duration is readable, else an empty TXT; go on
            failed += 1
            T = T if T is not None else container_T(path)
            out = constant(T) if T is not None else ""
            print(f"{name}: {type(e).__name__}: {e} -> '{out}'", file=sys.stderr, flush=True)
        with open(os.path.join(a.results, name + ".txt"), "w") as f:
            f.write(out)
        if DEV == "cuda":
            info += (f"peak alloc {torch.cuda.max_memory_allocated() / 2**20:.0f} MB, "
                     f"reserved {torch.cuda.max_memory_reserved() / 2**20:.0f} MB, ")
        print(f"{name}\t'{out}'\t{info}{time.time() - t0:.1f} s", flush=True)
        if DEV == "cuda":
            torch.cuda.empty_cache()
    print(f"policy {policy}; failures {failed}", flush=True)


if __name__ == "__main__":
    main()
