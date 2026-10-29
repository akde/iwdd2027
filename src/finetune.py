"""E3 (wiki/plan.md section 6, wiki/experiments.md): MoViNet-A2 B0 fine-tune.

Frozen trunk: conv1 + blocks b0_l0..b4_l4 in fp16, no gradients. Trainable tail: blocks.b4_l5, b4_l6 and conv7 in fp32,
every BatchNorm in eval mode (frozen running statistics); then the baseline-v0 TCN head (src/head.py).
Windows: 20 frames (4 s) centred every 1 s (rows c = 0, 5, 10, ... of the 5 fps sequence; frames c-10..c+9 clamped to
the sequence being processed: a training crop or the full clip); conv7 output averaged over time and space, linearly
interpolated to every 5 Hz row (rows after the last centre take its value). --form: raw, m (each sequence minus its own
per-dimension median, as E2m) or p (m plus E2p's pre-onset sub-crops as negatives with probability 0.25).
Training, per crop (src/train.py crop rule), all classes: gray p 0.2, dark p 0.3, brightness and contrast x U(0.8, 1.2),
horizontal flip p 0.5; baseline-v0 losses; AdamW (tail 1e-4, head 1e-3, weight decay 0.01), gradient clip 1, 16 crops
per step, cosine over 5 epochs. Epochs per outer fold k: inner split (train on k's training folds except (k+1) mod 5,
validate on fold (k+1) mod 5, full clips), E_k = argmin of the seed-mean inner validation loss; retrain on the 4
training folds for E_k epochs. Outer-fold metrics are not printed. Full clips (validation, OOF, train_pred) use trunk
outputs computed one window per forward, as the package will.

python src/finetune.py cache              -> cache/frames5/<video>.npy: 5 fps prep_rgb uint8 (the extractor's frames)
python src/finetune.py probe --form m     -> one inner-split epoch on outer fold 0: wall time, peak memory, finite loss
python src/finetune.py run --form m [--seeds 0,1,2] -> runs/s5_movinet_ft/: oof.csv, oof_seed<s>.csv, train_pred.csv,
                                             f<k>_s<s>.pt (tail + head), train.log, config.json, metrics.json
python src/finetune.py audit --form m     -> runs/s5_movinet_ft/view_q.csv: q of each OOF negative on the original, gray and
                                             dark views through its own fold models; edge.json: mean frame logit on rows
                                             within 2 s of a clip edge vs interior (plan 4.4; read by reports/e2/e2_audit.py)
python src/finetune.py robust_audit --form m [--robust] -> robust_q.csv: q and t of all 500 OOF clips on the original, a
                                             low-quality view (short side 128 and back, noise 0.01, seed per clip) and the
                                             other frame shape (16:9 <-> 4:3), through each clip's own fold models
--robust (E3r, wiki/experiments.md): runs/s5_movinet_ft_r; training adds, per crop and for all classes, a low-quality
look (p 0.5: short side U[112, 224] and back, bilinear with antialiasing, Gaussian noise sigma U[0, 0.02]) and the other
frame shape (p 0.3: width resized 16:9 <-> 4:3, no content cut), so a source's look cannot stand in for the label.

Synthetic training clips (E3s, wiki/experiments.md):
python src/finetune.py cache --release DIR -> cache/frames5_synth/<release>/<clip>.npy + index.csv (training only)
python src/finetune.py run --form m --robust --extra IDX [IDX ...] [--cap 1.0] --run ID -> runs/ID/
Each epoch of a fit adds min(available, floor(cap * real clips / 2)) whole minimal pairs, drawn without replacement by
a separate seeded generator. Fold rule (synth/RELEASE.md): the inner fit of outer fold k (validation fold j) uses
synthetic fold not in {k, j}, the outer fit fold != k; MEVA (fold -1) is always allowed. Inner validation, OOF,
train_pred and the head's standardization use the 500 real clips only. Without --extra a run is E3 / E3r as run.
--pad-real (camera-ready BUDGET control, real clips only): each epoch adds floor(cap * real clips / 2) * 2 further real
clips, drawn without replacement from the fit's real clips by the generator that draws the pairs, so the epoch pool, the
update count and the cosine schedule equal those of the same run with --extra.
--resume (run --epochs only, after a crash or power loss): rerun the same command with --resume; every fit whose
checkpoint f<k>_s<s>.pt exists is loaded instead of retrained (its OOF and train_pred logits are recomputed from it),
the config must match the interrupted one, and train.log is appended to. Checkpoints are written atomically (temporary
file, fsync, rename), so a checkpoint that exists is complete.

Final form E3f (wiki/experiments.md "E3f"), the shipped arm on all its clips:
python src/finetune.py run --form m --robust [--extra IDX] --epochs E --run ID   FE: every outer fold E epochs, no inner
                                             split (the OOF for the gates, TAU and DELTA)
python src/finetune.py full --form m --robust [--extra IDX] --epochs E --run ID  R: 5 groups x 3 seeds (seeds 0..14) on
                                             every real (+ synthetic) clip -> g<k>_s<i>.pt, insample_full.csv (package
                                             parity only, never scored)
python src/finetune.py synth_probe --form m --robust --extra IDX --runs RUN ... --out CSV   full-mode and per-group q, t
                                             of the MEVA synthetic clips (fold -1) through each run's 15 models
"""
import argparse, copy, hashlib, json, os, subprocess, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

from data import ROOT, decode, index
from features import MFPS, OUT_MOVINET, WIN, _load_rgb, dark_view, gray_view, load_movinet, prep_rgb
from head import Head, full_score, score
from train import BATCH, CLIP, DEV, clips, crop, loss, rows

sys.path.insert(0, str(ROOT / "tools"))
from iwdd_eval import auroc, report

CACHE, STRIDE, EPOCHS, LR_TAIL, LR_HEAD, WD = ROOT / "cache" / "frames5", 5, 5, 1e-4, 1e-3, 0.01
SYNTH = ROOT / "cache" / "frames5_synth"
P_GRAY, P_DARK, P_FLIP, JIT, PPRE, TRAIN_WB = 0.2, 0.3, 0.5, 0.2, 0.25, 64
P_LOWQ, P_ASPECT = 0.0, 0.0  # E3r (--robust): 0.5, 0.3


def build_cache():
    d = index()
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / "data" / r.split / "videos" / f"{r.video}.mp4" for r in d.itertuples()]
    with Pool(8) as pool:
        for v, x in zip(d.video, pool.imap(_load_rgb, paths)):
            np.save(CACHE / f"{v}.npy", x)
    print(f"{len(d)} clips cached in {CACHE}")


def _load_rgb_T(path):  # _load_rgb plus the decoder T
    x, _, T = decode(path, fps=MFPS)
    return prep_rgb(x), T


def build_cache_synth(rel):  # synthetic release -> SYNTH/<release>/<clip>.npy + index.csv, written last
    rel = Path(rel).resolve()
    m, out = pd.read_csv(rel / "manifest.csv"), SYNTH / rel.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.csv").unlink(missing_ok=True)
    rows = []
    with Pool(8) as pool:
        for r, (x, T) in zip(m.itertuples(), pool.imap(_load_rgb_T, [rel / "clips" / f"{c}.mp4" for c in m["clip"]])):
            d = json.loads((rel / "labels" / f"{r.clip}.json").read_text())
            if d["label"] not in ("true", "false"):
                raise ValueError(f"{r.clip}: label {d['label']!r}")
            y, g = int(d["label"] == "true"), int(d["timestamp"])
            assert (y, g, d["fold"], d["pair_id"]) == (r.y, r.timestamp, r.fold, r.pair_id), r.clip
            assert len(x) / MFPS >= T > (len(x) - 1) / MFPS and (not y or g >= 0), r.clip
            np.save(out / f"{r.clip}.npy", x)
            rows.append(dict(clip=r.clip, y=y, g=g, T=T, n=len(x), fold=int(r.fold), scene=r.scene_id,
                             pair=f"{rel.name}/{r.pair_id}", family=r.family))
    d = pd.DataFrame(rows)
    p = d.groupby("pair").agg(n=("y", "size"), y=("y", "sum"), folds=("fold", "nunique"))
    assert (p.n == 2).all() and (p.y == 1).all() and (p.folds == 1).all(), "pairs: one positive, one negative, one fold"
    d.to_csv(out / "index.csv", index=False)
    (out / "source.json").write_text(json.dumps(dict(release=str(rel), manifest_sha256=hashlib.sha256(
        (rel / "manifest.csv").read_bytes()).hexdigest(), clips=len(d)), indent=1))
    print(f"{len(d)} synthetic clips ({len(p)} pairs) cached in {out}")


def split(base):  # -> frozen fp16 trunk (shared by all models), names of the tail blocks
    names = [n for n, _ in base.blocks.named_children()]
    trunk = nn.Sequential(base.conv1, *[getattr(base.blocks, n) for n in names[:-2]]).eval()
    for p in trunk.parameters():
        p.requires_grad_(False)
    return trunk


def new_tail(base):  # fp32 copy of blocks.b4_l5, b4_l6, conv7; eval mode = frozen BatchNorm statistics
    return nn.Sequential(copy.deepcopy(base.blocks.b4_l5), copy.deepcopy(base.blocks.b4_l6),
                         copy.deepcopy(base.conv7)).float().to(DEV).eval().requires_grad_(True)


def to_x(frames):  # uint8 [n, H, W, 3] (numpy) -> [3, n, H, W] fp16 in [0, 1] on the GPU (as features.movinet)
    return torch.from_numpy(np.ascontiguousarray(frames)).to(DEV).permute(3, 0, 1, 2).half().div_(255)


@torch.no_grad()
def trunk_out(trunk, x, wb):  # [3, n, H, W] -> trunk outputs of the windows centred at rows 0, 5, 10, ...
    n = x.shape[1]
    idx = (torch.arange(0, n, STRIDE)[:, None] + torch.arange(-WIN // 2, WIN // 2)).clamp(0, n - 1)
    return torch.cat([trunk(x[:, idx[i:i + wb]].permute(1, 0, 2, 3, 4)) for i in range(0, len(idx), wb)])


def interp(f, n):  # [m, d] at the window centres -> [n, d] at every row; rows after the last centre take its value
    r = torch.arange(n, device=f.device)
    j = (r // STRIDE).clamp(max=len(f) - 1)
    w = torch.where(j == len(f) - 1, 0.0, (r - j * STRIDE) / STRIDE).to(f.dtype)[:, None]
    return f[j] * (1 - w) + f[(j + 1).clamp(max=len(f) - 1)] * w


def head_in(tail, h, n, form):  # trunk outputs -> head input rows [n, 640]
    f = interp(tail(h.float()).mean((2, 3, 4)), n)
    return f if form == "raw" else f - f.quantile(0.5, dim=0)  # quantile 0.5 = np.median (features.demed)


def lowq(x, s, sigma, gen=None):  # [3, n, H, W] float in [0, 1]: at short side s and back (less detail), plus noise
    H, W = x.shape[-2:]
    k = s / min(H, W)
    y = F.interpolate(x.transpose(0, 1), size=(max(8, round(H * k)), max(8, round(W * k))), mode="bilinear",
                      antialias=True, align_corners=False)
    y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False).transpose(0, 1)
    return (y + torch.randn(y.shape, device=y.device, generator=gen) * sigma).clamp(0, 1)


def other_aspect(x):  # [3, n, H, W]: width resized to the other frame shape (16:9 <-> 4:3), no content cut
    H, W = x.shape[-2:]
    W2 = round(H * 4 / 3) if W / H > 1.55 else round(H * 16 / 9)
    return F.interpolate(x.transpose(0, 1), size=(H, W2), mode="bilinear", antialias=True, align_corners=False).transpose(0, 1)


def augment(x, rng):  # one draw per crop; x [3, n, H, W] fp16 in [0, 1]
    x = x.float()
    if P_LOWQ and rng.random() < P_LOWQ:  # E3r: the look of a low-quality source, all classes (no draw when off: E3 as run)
        x = lowq(x, rng.uniform(112, 224), rng.uniform(0, 0.02))
    if P_ASPECT and rng.random() < P_ASPECT:  # E3r: the other frame shape, all classes
        x = other_aspect(x)
    if rng.random() < P_GRAY:
        x = (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]).expand(3, *x.shape[1:])
    if rng.random() < P_DARK:  # as features.dark_view
        x = (x ** rng.uniform(2, 3) * rng.uniform(0.4, 0.8) + torch.randn_like(x) * rng.uniform(0.02, 0.04)).clamp(0, 1)
    x = (x * rng.uniform(1 - JIT, 1 + JIT)).clamp(0, 1)
    m = x.mean()
    x = ((x - m) * rng.uniform(1 - JIT, 1 + JIT) + m).clamp(0, 1)
    return (x.flip(-1) if rng.random() < P_FLIP else x).half()


class E3:
    def __init__(self, form, extra=()):
        self.form, self.R, self.X = form, clips(), {}  # R: the 500 real clips; X: synthetic training clips (E3s)
        self.base = load_movinet()
        self.trunk = split(self.base)
        self.frames = {v: np.load(CACHE / f"{v}.npy", mmap_mode="r") for v in self.R}
        self.t, self.stats, self.pre, self.full = {}, {}, {}, {}
        for v in self.R:
            z = np.load(OUT_MOVINET / f"{v}.npz")
            assert len(z["t"]) == len(self.frames[v]), v
            self.t[v] = z["t"]
            f = z["movinet"].astype(np.float32)
            self.stats[v] = torch.from_numpy(f if form == "raw" else f - np.median(f, 0))
            if form == "p" and "pre" in z:
                self.pre[v] = slice(int(z["pre"][0]), int(z["pre"].sum()))
        for p in map(Path, extra):
            for r in pd.read_csv(p).itertuples(index=False):
                assert r.clip not in self.R and r.clip not in self.X, f"duplicate clip {r.clip}"
                self.X[r.clip] = r
                self.frames[r.clip] = np.load(p.parent / f"{r.clip}.npy", mmap_mode="r")
                self.t[r.clip] = np.arange(len(self.frames[r.clip])) / MFPS

    def cache_full(self, vids):  # trunk outputs of full clips, one window per forward (the package's way), on the GPU
        for v in vids:
            if v not in self.full:
                self.full[v] = trunk_out(self.trunk, to_x(self.frames[v]), wb=1)

    @torch.no_grad()
    def logits(self, tail, head, vids):
        head.eval()
        self.cache_full(vids)
        return {v: head(head_in(tail, self.full[v], len(self.t[v]), self.form)).cpu().numpy() for v in vids}

    @torch.no_grad()
    def val(self, tail, head, vids):  # full clips -> mean loss, AUROC
        head.eval()
        self.cache_full(vids)
        out = [loss(head, head_in(tail, self.full[v], len(self.t[v]), self.form), self.t[v], self.R[v]) for v in vids]
        return np.mean([float(l) for l, _ in out]), auroc(np.array([self.R[v].y for v in vids]), np.array([q for _, q in out]))

    def syn(self, *out):  # fold rule (synth/RELEASE.md): synthetic clips with fold not in out; MEVA (-1) always allowed
        return [v for v in self.X if self.X[v].fold not in out]

    def pairs(self, extra):  # synthetic clips -> {pair: [members]}, sorted
        out = {}
        for v in sorted(extra):
            out.setdefault(self.X[v].pair, []).append(v)
        return out

    def fit(self, vids, seed, stop=None, val=(), log=print, tag="", extra=(), cap=1.0, pad=False):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        tail = new_tail(self.base)
        X = torch.cat([self.stats[v] for v in vids])  # head standardization: real clips only
        head = Head(X.shape[1], X.mean(0), X.std(0) + 1e-6).to(DEV)
        opt = torch.optim.AdamW([{"params": tail.parameters(), "lr": LR_TAIL}, {"params": head.parameters(), "lr": LR_HEAD}],
                                weight_decay=WD)
        pairs = self.pairs(extra)  # E3s: whole pairs per epoch, from their own generator (no draw without extra)
        ids, npair, srng = sorted(pairs), min(len(pairs), int(cap * len(vids)) // 2), np.random.default_rng([seed, 1])
        assert not (pad and pairs), "--pad-real is a real-only control"
        npad = 2 * (int(cap * len(vids)) // 2) if pad else 0  # --pad-real: real clips in the place of the pairs' clips
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * -(-(len(vids) + 2 * npair + npad) // BATCH))
        params, curve = list(tail.parameters()) + list(head.parameters()), []
        for ep in range(stop or EPOCHS):
            head.train()
            pool = vids + [v for p in (srng.choice(ids, npair, replace=False) if npair else []) for v in pairs[p]]
            pool += [str(v) for v in srng.choice(vids, npad, replace=False)] if npad else []
            t0, order, tl, ty, tq = time.time(), [pool[i] for i in rng.permutation(len(pool))], [], [], []
            for i in range(0, len(order), BATCH):
                opt.zero_grad()
                b, bl = order[i:i + BATCH], 0.0
                for v in b:
                    r, t = self.R[v] if v in self.R else self.X[v], self.t[v]
                    if v in self.pre and rng.random() < PPRE:  # pre-onset sub-crop of a positive: a same-scene negative
                        c, r = self.pre[v], r._replace(y=0)
                    else:
                        c = crop(rng, t, r)
                    x = augment(to_x(self.frames[v][c]), rng)
                    l, q = loss(head, head_in(tail, trunk_out(self.trunk, x, TRAIN_WB), x.shape[1], self.form), t[c], r)
                    (l / len(b)).backward()
                    bl += float(l.detach()) / len(b)
                    ty.append(r.y); tq.append(q)
                torch.nn.utils.clip_grad_norm_(params, CLIP)
                opt.step()
                sched.step()
                tl.append(bl)
            msg = (f"{tag} ep {ep} pool {len(pool)} updates {-(-len(pool) // BATCH)} train_loss {np.mean(tl):.4f} train_auc_crops {auroc(np.array(ty), np.array(tq)):.3f} "
                   f"{time.time() - t0:.0f} s")
            if len(val):
                vl, va = self.val(tail, head, val)
                msg += f" val_loss {vl:.4f} val_auc {va:.3f}"
                curve.append(vl)
            log(msg)
        return tail.eval(), head.eval(), curve


def save_ckpt(tail, head, path):  # atomic: a checkpoint that exists is complete, even after a power loss
    tmp = path.with_suffix(".pt.tmp")
    with open(tmp, "wb") as fh:
        torch.save({"tail": tail.state_dict(), "head": head.state_dict()}, fh)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)
    d = os.open(path.parent, os.O_RDONLY); os.fsync(d); os.close(d)


def load_ckpt(e, path):  # -> (tail, head) in eval mode, as returned by E3.fit
    ck = torch.load(path, map_location=DEV, weights_only=True)
    d = next(iter(e.stats.values())).shape[1]
    tail, head = new_tail(e.base), Head(d, torch.zeros(d), torch.ones(d)).to(DEV)
    tail.load_state_dict(ck["tail"]); head.load_state_dict(ck["head"])
    return tail.eval(), head.eval()


def full(e, a, cfg, run, seeds, log):  # E3f: every real (+ synthetic) clip, --epochs E, nothing validated
    vids, n = list(e.R), len(seeds)
    files = {f"g{g}_s{i}.pt": g * n + s for g in range(a.groups) for i, s in enumerate(seeds)}  # seeds 0..14 by default
    cfg.update(arm=cfg["experiment"], experiment="E3f", epochs=a.epochs, seeds_by_file=files)
    (run / "config.json").write_text(json.dumps(cfg, indent=1))
    lg = {v: {} for v in vids}
    for name, s in files.items():
        g = int(name[1:name.index("_")])
        tail, head, _ = e.fit(vids, s, stop=a.epochs, log=log, tag=f"E3f group {g} seed {s}", extra=e.syn(), cap=a.cap)
        torch.save({"tail": tail.state_dict(), "head": head.state_dict()}, run / name)
        for v, l in e.logits(tail, head, vids).items():  # E3.logits: the OOF's path
            lg[v].setdefault(g, []).append(l)
    out = [dict(video=v, fold=e.R[v].fold, y=e.R[v].y, g=e.R[v].g, dur=e.R[v].T,
                **dict(zip("qt", full_score({k: np.mean(ls, 0) for k, ls in lg[v].items()}, e.t[v])))) for v in vids]
    pd.DataFrame(out).sort_values("video").to_csv(run / "insample_full.csv", index=False)
    log(f"E3f: {len(files)} models, {len(e.syn())} synthetic clips allowed; insample_full.csv (parity only, never scored)")


def load_groups(e, run):  # run dir -> {group or fold: [(tail, head)] in seed order}: g-files if present, else f-files
    out, pre = {}, "g" if (run / "g0_s0.pt").exists() else "f"
    for f in sorted(run.glob(f"{pre}*_s*.pt"), key=lambda f: [int(x) for x in f.stem[1:].split("_s")]):
        ck = torch.load(f, map_location=DEV, weights_only=True)
        tail, head = new_tail(e.base), Head(640, torch.zeros(640), torch.ones(640)).to(DEV)
        tail.load_state_dict(ck["tail"]); head.load_state_dict(ck["head"])
        out.setdefault(int(f.stem[1:].split("_s")[0]), []).append((tail.eval(), head.eval()))
    return out


@torch.no_grad()
def synth_probe(e, runs, out):  # E3f rule 5: the MEVA synthetic clips (fold -1), unseen by real-arm runs
    models = {r.name: load_groups(e, r) for r in runs}
    rows = []
    for v in e.syn(*range(5)):
        r, t = e.X[v], e.t[v]
        h = trunk_out(e.trunk, to_x(e.frames[v]), wb=1)
        o = dict(clip=v, y=r.y, g=r.g, dur=r.T, scene=r.scene, family=r.family)
        for name, groups in models.items():
            lg = {k: np.mean([hd(head_in(tl, h, len(t), e.form)).cpu().numpy() for tl, hd in m], 0) for k, m in groups.items()}
            o[f"q_{name}"], o[f"t_{name}"] = full_score(lg, t)
            o.update({f"q_{name}_g{k}": score(l, t)[0] for k, l in lg.items()})
        rows.append(o)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"synth_probe: {len(rows)} clips x {len(models)} runs ({', '.join(f'{k}: {len(v)} groups' for k, v in models.items())}) -> {out}")


@torch.no_grad()
def audit(e, run):  # plan 4.4 for E3: views of the OOF negatives through each clip's own fold models (tails + heads)
    cfg, o = json.loads((run / "config.json").read_text()), pd.read_csv(run / "oof.csv")
    models = {}
    for k in sorted(o.fold.unique()):
        for s in cfg["seeds"]:
            ck = torch.load(run / f"f{k}_s{s}.pt", map_location=DEV, weights_only=True)
            tail, head = new_tail(e.base), Head(640, torch.zeros(640), torch.ones(640)).to(DEV)
            tail.load_state_dict(ck["tail"]); head.load_state_dict(ck["head"])
            models.setdefault(int(k), []).append((tail.eval(), head.eval()))
    rows, edge, inner = [], [], []
    for r in o[o.y == 0].itertuples():
        x, t, q = e.frames[r.video], e.t[r.video], {}
        for vw, fr in (("orig", x), ("gray", gray_view(x)), ("dark", dark_view(np.asarray(x), r.video))):
            h = trunk_out(e.trunk, to_x(fr), wb=1)
            lg = np.mean([hd(head_in(tl, h, len(t), e.form)).cpu().numpy() for tl, hd in models[r.fold]], 0)
            q[vw] = score(lg, t)[0]
            if vw == "orig":
                m = (t < 2) | (t > r.dur - 2)
                edge += lg[m].tolist(); inner += lg[~m].tolist()
        rows.append(dict(video=r.video, night=r.night, **q))
    v = pd.DataFrame(rows)
    v.to_csv(run / "view_q.csv", index=False)
    (run / "edge.json").write_text(json.dumps([float(np.mean(edge)), float(np.mean(inner))]))
    print(f"audit: {len(v)} negatives, max |q_orig - oof q| {np.abs(v.orig.values - o[o.y == 0].q.values).max():.1e}")


@torch.no_grad()
def robust_audit(e, run):  # E3r report: all OOF clips on the original, a low-quality view and the other frame shape
    cfg, o = json.loads((run / "config.json").read_text()), pd.read_csv(run / "oof.csv")
    models = {}
    for k in sorted(o.fold.unique()):
        for s in cfg["seeds"]:
            ck = torch.load(run / f"f{k}_s{s}.pt", map_location=DEV, weights_only=True)
            tail, head = new_tail(e.base), Head(640, torch.zeros(640), torch.ones(640)).to(DEV)
            tail.load_state_dict(ck["tail"]); head.load_state_dict(ck["head"])
            models.setdefault(int(k), []).append((tail.eval(), head.eval()))
    rows = []
    for r in o.itertuples():
        x, t, out = to_x(e.frames[r.video]), e.t[r.video], dict(video=r.video, y=r.y)
        gen = torch.Generator(device=DEV).manual_seed(int(r.video[3:]))
        for vw, xv in (("orig", x), ("lowq", lowq(x.float(), 128, 0.01, gen).half()), ("aspect", other_aspect(x.float()).half())):
            h = trunk_out(e.trunk, xv, wb=1)
            lg = np.mean([hd(head_in(tl, h, len(t), e.form)).cpu().numpy() for tl, hd in models[r.fold]], 0)
            out[f"q_{vw}"], out[f"t_{vw}"] = score(lg, t)
        rows.append(out)
    v = pd.DataFrame(rows)
    v.to_csv(run / "robust_q.csv", index=False)
    print(f"robust_audit: {len(v)} clips, max |q_orig - oof q| {np.abs(v.q_orig.values - o.q.values).max():.1e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "probe", "run", "audit", "robust_audit", "full", "synth_probe"])
    ap.add_argument("--robust", action="store_true", help="E3r: low-quality look and frame-shape augmentation")
    ap.add_argument("--form", choices=["raw", "m", "p"], default="m")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--release", help="cache: a synthetic release dir (manifest.csv, clips/, labels/)")
    ap.add_argument("--extra", nargs="*", default=[], help="run: synthetic index.csv files (E3s)")
    ap.add_argument("--cap", type=float, default=1.0, help="run: synthetic clips per real training clip and epoch")
    ap.add_argument("--pad-real", action="store_true", help="run: real-only control with the --extra run's pool size")
    ap.add_argument("--resume", action="store_true", help="run --epochs: reuse the finished fits of an interrupted run")
    ap.add_argument("--run", help="run id under runs/ (default: s5_movinet_ft_r with --robust, else s5_movinet_ft)")
    ap.add_argument("--epochs", type=int, help="run: E epochs in every outer fold, no inner split (E3f FE); full: required")
    ap.add_argument("--groups", type=int, default=5, help="full: groups of len(--seeds) models")
    ap.add_argument("--runs", nargs="*", default=[], help="synth_probe: run ids under runs/")
    ap.add_argument("--out", help="synth_probe: output CSV")
    a = ap.parse_args()
    if a.cmd == "cache":
        return build_cache_synth(a.release) if a.release else build_cache()
    global P_LOWQ, P_ASPECT
    if a.robust:
        P_LOWQ, P_ASPECT = 0.5, 0.3
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    seeds = tuple(int(s) for s in a.seeds.split(","))
    e = E3(a.form, a.extra if a.cmd in ("run", "full", "synth_probe") else ())
    R, folds = e.R, sorted({r.fold for r in e.R.values()})
    if a.cmd == "probe":  # outer fold 0's inner split only: train on folds 2-4, validate on fold 1
        torch.cuda.reset_peak_memory_stats()
        tr, va = [v for v in R if R[v].fold not in (0, 1)], [v for v in R if R[v].fold == 1]
        t0 = time.time()
        _, _, c = e.fit(tr, 0, stop=1, val=va, log=lambda s: None, tag="probe")  # no AUROC printed (plan 6: time only)
        print(f"probe: {time.time() - t0:.0f} s for 1 epoch on {len(tr)} clips + validation on {len(va)} (incl. trunk "
              f"cache), peak alloc {torch.cuda.max_memory_allocated() / 2**30:.2f} GB (+ about 4.4 GB for the 500-clip "
              f"trunk cache in the run), val loss finite: {bool(np.isfinite(c[0]))}")
        return
    run = ROOT / "runs" / (a.run or ("s5_movinet_ft_r" if a.robust else "s5_movinet_ft"))
    if a.cmd == "audit":
        return audit(e, run)
    if a.cmd == "robust_audit":
        return robust_audit(e, run)
    if a.cmd == "synth_probe":
        return synth_probe(e, [ROOT / "runs" / r for r in a.runs], Path(a.out))
    if (run / ("insample_full.csv" if a.cmd == "full" else "oof.csv")).exists():
        sys.exit(f"{run} already holds a finished run; choose another --run")
    run.mkdir(parents=True, exist_ok=True)
    git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    cfg = dict(experiment="E3r" if a.robust else "E3", form=a.form, seeds=seeds, epochs_max=EPOCHS, batch=BATCH, lr_tail=LR_TAIL, lr_head=LR_HEAD,
               wd=WD, clip=CLIP, stride_rows=STRIDE, aug=dict(gray=P_GRAY, dark=P_DARK, flip=P_FLIP, jitter=JIT, lowq=P_LOWQ, aspect=P_ASPECT),
               git=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain", "src", "tools")),
               folds_sha256=sha(ROOT / "data" / "folds.csv"), features_meta_sha256=sha(OUT_MOVINET / "meta.csv"))
    if e.X:
        cfg.update(experiment="E3s", synthetic=dict(
            index=a.extra, index_sha256=[sha(Path(p)) for p in a.extra], cap=a.cap, clips=len(e.X),
            source=[json.loads((Path(p).parent / "source.json").read_text()) for p in a.extra],
            pool={int(k): dict(inner=len(e.syn(k, (k + 1) % len(folds))), outer=len(e.syn(k))) for k in folds}))
    if a.epochs:
        cfg["epochs_fixed"] = a.epochs
    if a.pad_real:
        assert a.cmd == "run" and a.epochs and not a.extra, "--pad-real: run, --epochs, no --extra"
        cfg["pad_real"] = dict(cap=a.cap, rule="per epoch 2*floor(cap*n/2) extra real clips, no replacement")
    if a.resume and (run / "config.json").exists():  # same recipe as the interrupted run, or stop
        assert a.cmd == "run" and a.epochs, "--resume: run --epochs only"
        old = json.loads((run / "config.json").read_text())
        same = lambda c: {k: v for k, v in json.loads(json.dumps(c)).items() if k not in ("git", "dirty", "epochs_chosen", "resumed")}
        diff = sorted(k for k in set(same(old)) | set(same(cfg)) if same(old).get(k) != same(cfg).get(k))
        if diff:
            sys.exit(f"--resume: {run}/config.json differs from this command in {diff}; not resuming")
        cfg["resumed"] = old.get("resumed", []) + [dict(previous_git=old["git"], previous_dirty=old["dirty"],
                                                         at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), reused={})]
    (run / "config.json").write_text(json.dumps(cfg, indent=1))
    f = open(run / "train.log", "a" if a.resume else "w")
    log = lambda s: (print(s, file=f, flush=True), print(s, flush=True))
    if cfg.get("resumed"):
        log(f"RESUME at {cfg['resumed'][-1]['at']} (git {cfg['git'][:7]}): finished fits are loaded, not retrained")
    if a.cmd == "full":
        assert a.epochs, "full needs --epochs"
        return full(e, a, cfg, run, seeds, log)
    times = {v: (None, e.t[v]) for v in R}
    oof, per_seed, train_rows = [], {s: [] for s in seeds}, []
    for k in folds:
        tr, te = [v for v in R if R[v].fold != k], [v for v in R if R[v].fold == k]
        if a.epochs:  # E3f FE: fixed epochs, no inner split
            E = cfg.setdefault("epochs_chosen", {})[int(k)] = a.epochs
            log(f"fold {k}: fixed {E} epochs (--epochs)")
        else:
            j = (k + 1) % len(folds)
            curves = [e.fit([v for v in tr if R[v].fold != j], s, val=[v for v in tr if R[v].fold == j], log=log,
                            tag=f"E3 fold {k} inner {j} seed {s}", extra=e.syn(k, j), cap=a.cap)[2] for s in seeds]
            mean = np.mean(curves, 0)
            E = cfg.setdefault("epochs_chosen", {})[int(k)] = int(mean.argmin()) + 1
            log(f"fold {k}: mean inner val loss by epoch {mean.round(4).tolist()} -> {E} epochs")
        lte, ltr = {v: [] for v in te}, {v: [] for v in tr}
        for s in seeds:
            ck = run / f"f{k}_s{s}.pt"
            if a.resume and ck.exists():
                tail, head = load_ckpt(e, ck)
                cfg.setdefault("resumed", [dict(previous_git=None, previous_dirty=None, at=None, reused={})])
                cfg["resumed"][-1]["reused"][ck.name] = sha(ck)
                log(f"E3 fold {k} seed {s}: reused {ck.name} (sha256 {sha(ck)[:12]})")
            else:
                tail, head, _ = e.fit(tr, s, stop=E, log=log, tag=f"E3 fold {k} seed {s}", extra=e.syn(k), cap=a.cap,
                                      pad=a.pad_real)
                save_ckpt(tail, head, ck)
            for lg, vids in ((lte, te), (ltr, tr)):
                for v, l in e.logits(tail, head, vids).items():
                    lg[v].append(l)
            per_seed[s] += rows(R, {v: [lte[v][-1]] for v in te}, times)
        oof += rows(R, lte, times)
        train_rows += rows(R, ltr, times, fold=k)
        (run / "config.json").write_text(json.dumps(cfg, indent=1))
    pd.DataFrame(oof).sort_values("video").to_csv(run / "oof.csv", index=False)
    for s in seeds:
        pd.DataFrame(per_seed[s]).sort_values("video").to_csv(run / f"oof_seed{s}.csv", index=False)
    pd.DataFrame(train_rows).to_csv(run / "train_pred.csv", index=False)
    o = pd.read_csv(run / "oof.csv").sort_values("video").reset_index(drop=True)
    met = {k: report(o, i)[0] for k, i in (("int", True), ("float", False))}
    (run / "metrics.json").write_text(json.dumps(met, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    log(f"E3: xfit F1 (--int) {met['int']['xfit_f1']}, AUROC {met['int']['auroc']}")


if __name__ == "__main__":
    main()
