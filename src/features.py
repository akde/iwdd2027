"""Stage 4 features (wiki/baseline.md): frozen DINOv2-B/14-reg in fp16 on gray+CLAHE frames at t = i/8 s.

extract(model, x, step=True) -> t [N] (= i/8), cls [N,768] fp16 (CLS minus the clip's per-dimension median),
                     step [N,5] (top-5 per-patch cosine distances between the per-dimension medians of
                     rows i-32..i-8 and i+8..i+32, i.e. [t-4, t-1] s vs [t+1, t+4] s; rows clamped to the clip);
                     step=False returns t and cls only (stage 7).
grid(model, x)    -> t [N], grid [N,64,768] fp16 (E1, wiki/experiments.md): patch tokens average-pooled to 8x8 cells
                     (adaptive pooling over the H'/14 x W'/14 patch grid; cell = row * 8 + col), each cell minus its
                     clip median.
movinet(model, x) -> t [N] (= j/5), movinet [N,640] fp16 (E2, wiki/experiments.md): frozen MoViNet-A2 base on RGB
                     frames at 5 fps, short side 224 (prep_rgb); for frame j the 4 s window of frames j-10..j+9
                     (clamped to the clip), conv7 output averaged over time and space. Frames stay on the CPU; MBATCH
                     windows at a time go to the GPU as uint8 (the package's settings, wiki/plan.md 4.1).
movinet_clip(model, x, r) -> movinet() of the clip, plus crops [K,2] (start row, rows) and crop<i> [rows,640]: K training
                     sub-crops drawn once by src/train.py's crop rule, each extracted as its own clip, so its edge windows
                     are clamped at its own edges as a test clip's are (edge fix, wiki/plan.md 4.2).
                     Positives with g >= 8 and g + 0.5 < T also get pre [2] and pre_crop: one sub-crop inside [0, g - 2]
                     (E2p pre-onset negative). Negatives also get gray and dark: the full clip's features on a gray
                     view (luma x 3) and a dark view (x^gamma * a + noise, seed per clip) for the shortcut audit.
demed(a)          -> a minus its per-dimension median over rows (E2m: per sequence, at load time).
Build all 500: python src/features.py [--grid | --movinet] -> features/ (features_grid/, features_movinet/):
<video>.npz, meta.csv (incl. peak GPU memory per clip).
"""
import argparse, hashlib, sys, time
from multiprocessing import Pool
from pathlib import Path

import cv2, numpy as np, pandas as pd, torch, torch.nn.functional as F

from data import FPS, ROOT, decode, index

HUB = ROOT / "vendor" / "dinov2-7b187bd"  # facebookresearch/dinov2 hubconf.py, LICENSE, dinov2/ at commit 7b187bd
WEIGHTS = ROOT / "weights" / "dinov2_vitb14_reg4_pretrain.pth"
SHA256 = "73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71"
OUT, OUT_GRID, GRID = ROOT / "features", ROOT / "features_grid", 8
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
NEAR, FAR, TOPK = FPS, 4 * FPS, 5  # step windows: rows i-FAR..i-NEAR and i+NEAR..i+FAR
BATCH, SLAB, CHUNK = 32, 64, 4  # backbone batch; rows per GPU slab of patch tokens; rows per median
MOVINET = ROOT / "vendor" / "movinet-c2d1edf"  # Atze00/MoViNet-pytorch at c2d1edf + iwdd.patch (wiki/decisions.md)
MOVINET_WEIGHTS = ROOT / "submission" / "model" / "movinet_a2_base_statedict_v3"  # tracked: the package ships it
MOVINET_SHA256 = "4c1cbef685413caffecd7dcdc656a8e009836868a0e5de47ce42a8486cc91659"
OUT_MOVINET, MFPS, WIN, SHORT, MBATCH, HALF, K = ROOT / "features_movinet", 5, 20, 224, 1, True, 3  # HALF: tests/test_movinet.py


def load_backbone(device="cuda"):
    if hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() != SHA256:
        raise ValueError(f"{WEIGHTS}: SHA256 mismatch")
    m = torch.hub.load(str(HUB), "dinov2_vitb14_reg", source="local", pretrained=False)
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))  # strict
    return m.half().to(device).eval()  # fp16 on the CPU, then to the GPU


def load_movinet(device="cuda", half=None):  # fp16 on the GPU (HALF); always fp32 on the CPU (no fp16 avg_pool3d there)
    half = HALF and device != "cpu" if half is None else half
    if hashlib.sha256(MOVINET_WEIGHTS.read_bytes()).hexdigest() != MOVINET_SHA256:
        raise ValueError(f"{MOVINET_WEIGHTS}: SHA256 mismatch")
    sys.path.insert(0, str(MOVINET))
    from movinets import MoViNet
    from movinets.config import _C
    # what pretrained=True sets (it would download from GitHub): 600 classes, 3d convs, TF-like padding
    m = MoViNet(_C.MODEL.MoViNetA2, causal=False, pretrained=False, num_classes=600, conv_type="3d", tf_like=True)
    m.load_state_dict(torch.load(MOVINET_WEIGHTS, map_location="cpu", weights_only=True))  # strict
    return (m.half() if half else m).to(device).eval()


def prep_rgb(frames):
    """RGB uint8 [N,H,W,3] -> RGB uint8 [N,H',W',3], short side 224, aspect kept, no crop."""
    h, w = frames.shape[1:3]
    H, W = round(h * SHORT / min(h, w)), round(w * SHORT / min(h, w))
    return np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR) for f in frames])


def prep(frames):
    """RGB uint8 [N,H,W,3] -> gray+CLAHE uint8 [N,H',W'], H' and W' rounded to multiples of 14."""
    h, w = frames.shape[1:3]
    H, W = max(14, round(h / 14) * 14), max(14, round(w / 14) * 14)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        out.append(clahe.apply(g if (H, W) == (h, w) else cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)))
    return np.stack(out)


@torch.inference_mode()
def extract(model, x, device="cuda", step=True):
    N = len(x)
    mean, std = (torch.tensor(v, device=device).view(1, 3, 1, 1) for v in (MEAN, STD))
    cls, pat = [], []
    for i in range(0, N, BATCH):
        b = torch.from_numpy(x[i:i + BATCH]).to(device)[:, None].expand(-1, 3, -1, -1).float().div(255)
        o = model.forward_features(((b - mean) / std).half())
        cls.append(o["x_norm_clstoken"].float())
        if step:
            pat.append(o["x_norm_patchtokens"].cpu())  # whole clip on CPU; GPU holds one slab at a time
    cls = torch.cat(cls)
    cls = cls - cls.median(0).values
    if not step:
        return dict(t=np.arange(N) / FPS, cls=cls.half().cpu().numpy())
    pat = torch.cat(pat)
    before, after = torch.arange(-FAR, -NEAR + 1, device=device), torch.arange(NEAR, FAR + 1, device=device)
    step = []
    for s in range(0, N, SLAB):
        lo = max(0, s - FAR)
        p = pat[lo:min(N, s + SLAB + FAR)].to(device)
        for i in range(s, min(s + SLAB, N), CHUNK):
            r = torch.arange(i, min(i + CHUNK, N), device=device)[:, None]
            m0, m1 = (p[(r + o).clamp(0, N - 1) - lo].median(1).values.float() for o in (before, after))
            d = (1 - F.cosine_similarity(m0, m1, dim=-1)).clamp(0, 2)
            step.append(d.topk(TOPK, dim=1).values.cpu())
    return dict(t=np.arange(N) / FPS, cls=cls.half().cpu().numpy(), step=torch.cat(step).numpy())


@torch.inference_mode()
def grid(model, x, device="cuda"):
    N, h, w = len(x), x.shape[1] // 14, x.shape[2] // 14
    mean, std = (torch.tensor(v, device=device).view(1, 3, 1, 1) for v in (MEAN, STD))
    out = []
    for i in range(0, N, BATCH):
        b = torch.from_numpy(x[i:i + BATCH]).to(device)[:, None].expand(-1, 3, -1, -1).float().div(255)
        p = model.forward_features(((b - mean) / std).half())["x_norm_patchtokens"].float()  # [B, h*w, 768], row-major
        p = F.adaptive_avg_pool2d(p.unflatten(1, (h, w)).permute(0, 3, 1, 2), GRID)  # [B, 768, 8, 8]
        out.append(p.flatten(2).transpose(1, 2))  # [B, 64, 768]
    g = torch.cat(out)
    g = g - g.median(0).values
    return dict(t=np.arange(N) / FPS, grid=g.half().cpu().numpy())


@torch.inference_mode()
def movinet(model, x, device="cuda"):
    N, dt = len(x), next(model.parameters()).dtype
    v = torch.from_numpy(x).permute(3, 0, 1, 2)  # [3, N, H, W] uint8, stays on the CPU
    off = torch.arange(-WIN // 2, WIN // 2)
    out = []
    for i in range(0, N, MBATCH):
        idx = (torch.arange(i, min(i + MBATCH, N))[:, None] + off).clamp(0, N - 1)  # [B, 20]
        b = v[:, idx].permute(1, 0, 2, 3, 4).to(device).to(dt).div_(255)  # [B, 3, 20, H, W] in [0, 1], no mean/std
        out.append(model.conv7(model.blocks(model.conv1(b))).float().mean((2, 3, 4)))
    return dict(t=np.arange(N) / MFPS, movinet=torch.cat(out).half().cpu().numpy())


def subcrops(r, n):  # K training sub-crops (start row, rows), drawn once per clip by src/train.py's crop rule
    from train import crop  # here, not at the top: train.py imports this module
    rng, t = np.random.default_rng([0, int(r.video[3:])]), np.arange(n) / MFPS
    return np.array([(c.start, c.stop - c.start) for c in (crop(rng, t, r) for _ in range(K))])


def precrop(r):  # E2p: one sub-crop inside [0, g - 2], length U[6, min(20, g - 2)] s, or None
    if not r.y or r.g < 8 or r.g + 0.5 >= r.T:
        return None
    rng, end = np.random.default_rng([1, int(r.video[3:])]), int((r.g - 2) * MFPS)  # rows with t < g - 2
    L = min(end, int(round(rng.uniform(6, min(20, r.g - 2)) * MFPS)))
    s = int(rng.integers(0, end - L + 1))
    return np.array([s, L])


def gray_view(x):
    return np.repeat(np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in x])[..., None], 3, -1)


def dark_view(x, video):  # x^gamma (gamma U[2, 3]) * U(0.4, 0.8) + Gaussian noise (sigma U[0.02, 0.04]), seed per clip
    rng = np.random.default_rng([2, int(video[3:])])
    gam, a, sig = rng.uniform(2, 3), rng.uniform(0.4, 0.8), rng.uniform(0.02, 0.04)
    f = lambda y: np.clip((y / 255) ** gam * a + rng.normal(0, sig, y.shape).astype(np.float32), 0, 1)
    return np.stack([np.round(f(y.astype(np.float32)) * 255).astype(np.uint8) for y in x])


def movinet_clip(model, x, r, device="cuda"):
    z = movinet(model, x, device)
    z["crops"] = subcrops(r, len(x))
    for i, (s, L) in enumerate(z["crops"]):
        z[f"crop{i}"] = movinet(model, x[s:s + L], device)["movinet"]
    pre = precrop(r)
    if pre is not None:
        z["pre"], z["pre_crop"] = pre, movinet(model, x[pre[0]:pre.sum()], device)["movinet"]
    if not r.y:
        z["gray"] = movinet(model, gray_view(x), device)["movinet"]
        z["dark"] = movinet(model, dark_view(x, r.video), device)["movinet"]
    return z


def demed(a):
    return a - np.median(a, 0)


def _load(path):
    return prep(decode(path)[0])


def _load_rgb(path):
    return prep_rgb(decode(path, fps=MFPS)[0])


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--grid", action="store_true", help="E1: 8x8 patch-token grid -> features_grid/")
    g.add_argument("--movinet", action="store_true", help="E2: MoViNet-A2 4 s window features -> features_movinet/")
    a = ap.parse_args()
    out, fn, load, backbone = ((OUT_MOVINET, movinet_clip, _load_rgb, load_movinet) if a.movinet else
                               (OUT_GRID, lambda m, x, r: grid(m, x), _load, load_backbone) if a.grid else
                               (OUT, lambda m, x, r: extract(m, x), _load, load_backbone))
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False  # as submission/test.py
    d = index()
    out.mkdir(exist_ok=True)
    paths = [ROOT / "data" / r.split / "videos" / f"{r.video}.mp4" for r in d.itertuples()]
    rows = []
    with Pool(8) as pool:  # forked before CUDA init; workers only decode
        model = backbone()
        for r, x in zip(d.itertuples(), pool.imap(load, paths)):
            v, t0 = r.video, time.time()
            torch.cuda.reset_peak_memory_stats()
            np.savez(out / f"{v}.npz", **fn(model, x, r))
            rows.append(dict(video=v, N=len(x), H=x.shape[1], W=x.shape[2],
                             peak_alloc_mb=torch.cuda.max_memory_allocated() / 2**20,
                             peak_reserved_mb=torch.cuda.max_memory_reserved() / 2**20, gpu_s=time.time() - t0))
    m = pd.DataFrame(rows)
    m.to_csv(out / "meta.csv", index=False)
    print(f"{len(m)} clips, {m.N.sum()} frames, GPU {m.gpu_s.sum():.0f} s; peak alloc max {m.peak_alloc_mb.max():.0f} MB, "
          f"reserved max {m.peak_reserved_mb.max():.0f} MB")


if __name__ == "__main__":
    main()
