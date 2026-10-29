"""Stage 5 training (wiki/baseline.md): 5 scene-grouped folds x 3 seeds -> out-of-fold predictions.

python src/train.py --inputs cls|cls_step|grid|movinet|movinet_m|movinet_p [--inner] -> runs/s5_<inputs>[_inner]/: oof.csv (seed-averaged logits), oof_seed<s>.csv,
train_pred.csv (each fold's ensemble on its own training clips), f<k>_s<s>.pt, train.log, config.json, metrics.json.
movinet (E2), movinet_m (E2m: each sequence minus its own median), movinet_p (E2p: E2m + pre-onset negatives: a
positive with a stored pre-onset sub-crop uses it with probability 0.25 per epoch, video and frame targets 0): training draws one of the clip's stored sub-crops
per epoch instead of cropping rows (edge fix, wiki/plan.md 4.2); inner validation and OOF use the full clip.
"""
import argparse, hashlib, json, subprocess, sys

import numpy as np, pandas as pd, torch, torch.nn.functional as F

from data import ROOT, index
from features import OUT, OUT_GRID, OUT_MOVINET, demed
from head import Head, score, to4, video_logit

sys.path.insert(0, str(ROOT / "tools"))
from iwdd_eval import auroc, report

EPOCHS, BATCH, LR, WD, CLIP, SEEDS = 30, 16, 1e-3, 0.01, 1.0, (0, 1, 2)
LMIN, LMAX, SIGMA = 6, 20, 1.0  # crop length range (s), Gaussian target sigma (s)
DEV = "cuda"
FEAT = dict(cls=OUT, cls_step=OUT, grid=OUT_GRID, movinet=OUT_MOVINET, movinet_m=OUT_MOVINET, movinet_p=OUT_MOVINET)
SUBS, PRE, PPRE = {}, {}, 0.25  # video -> [(x, times)] of its training sub-crops; -> (x, times) of its pre-onset sub-crop (movinet_p)


def clips():  # video -> row (y, g, T, fold, scene + analysis-only night, dim, event_type from the label JSONs)
    m = pd.DataFrame([dict(video=p.stem, **json.loads(p.read_text())["metadata"])
                      for p in sorted((ROOT / "data").glob("*/labels/*.json"))]).set_index("video")
    d = index().merge(pd.read_csv(ROOT / "data" / "folds.csv"), on="video")
    d = d.assign(night=d.video.map(m.time_of_day.eq("night")).astype(int),
                 dim=d.video.map(m.lighting.eq("low")).astype(int),
                 event_type=d.video.map(m.event_type.str.replace(" ", "")))
    return {r.video: r for r in d.itertuples(index=False)}


def load(inputs, videos):  # video -> (x [n,d] on the GPU, times [n]): 4 fps (DINOv2, 8 fps averaged) or 5 fps (movinet)
    out = {}
    for v in videos:
        z = np.load(FEAT[inputs] / f"{v}.npz")
        if inputs.startswith("movinet"):  # E2, E2m: one row per 5 fps frame already
            f = (lambda a: a) if inputs == "movinet" else demed
            gpu = lambda a: torch.from_numpy(f(a.astype(np.float32))).to(DEV)
            out[v] = (gpu(z["movinet"]), z["t"])
            SUBS[v] = [(gpu(z[f"crop{i}"]), z["t"][s:s + L]) for i, (s, L) in enumerate(z["crops"])]
            if inputs == "movinet_p" and "pre" in z:
                PRE[v] = (gpu(z["pre_crop"]), z["t"][z["pre"][0]:z["pre"].sum()])
            continue
        x = z[inputs] if inputs != "cls_step" else np.concatenate([z["cls"].astype(np.float32), z["step"]], 1)
        out[v] = (torch.from_numpy(to4(x)).to(DEV), to4(z["t"]))
    return out


def target(t, r):  # frame targets at times t, or None (no frame loss: peak at or past the clip end)
    if not r.y:
        return np.zeros_like(t)
    return None if r.g + 0.5 >= r.T else np.exp(-0.5 * ((t - r.g - 0.5) / SIGMA) ** 2)


def crop(rng, t, r):  # one rule for all: L ~ U[6, 20] s capped at the clip; positives keep [g-2, min(g+4, T)]
    n = len(t)
    L = min(n, int(round(rng.uniform(LMIN, LMAX) / (t[1] - t[0]))))  # rows per second from the times: 4 (DINOv2), 5 (movinet)
    lo, hi = 0, n - L
    if r.y:
        b = min(r.g + 4, r.T)
        a = min(max(0, r.g - 2), b)
        ia, ib = np.searchsorted(t, a, "left"), np.searchsorted(t, b, "right")  # frames with a <= t <= b
        L = min(n, max(L, ib - ia))  # at 5 fps [g-2, g+4] spans 31 rows and U[6, 6.1) s rounds to 30 (never at 4 fps)
        lo = max(0, ib - L)
        hi = max(lo, min(ia, n - L))
    s = int(rng.integers(lo, hi + 1))
    return slice(s, s + L)


def loss(model, x, t, r):  # -> (video BCE + frame BCE, q)
    logits = model(x)
    vl = video_logit(logits)
    l = F.binary_cross_entropy_with_logits(vl, torch.tensor(float(r.y), device=DEV))
    tg = target(t, r)
    if tg is not None:
        l = l + F.binary_cross_entropy_with_logits(logits, torch.as_tensor(tg, dtype=torch.float32, device=DEV))
    return l, float(torch.sigmoid(vl.detach()))


@torch.no_grad()
def evaluate(model, data, R, vids):  # full clips -> mean loss, AUROC
    model.eval()
    out = [loss(model, *data[v], R[v]) for v in vids]
    model.train()
    return np.mean([float(l) for l, _ in out]), auroc(np.array([R[v].y for v in vids]), np.array([q for _, q in out]))


def fit(data, R, vids, seed, epochs=EPOCHS, val=(), log=print, tag="", stop=None, curve=None):
    """Cosine schedule over `epochs`; train the first `stop` of them (all if None); val losses go to `curve`."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = torch.cat([data[v][0] for v in vids]).flatten(0, -2)  # grid: one mean/std over all cells and frames
    model = Head(X.shape[1], X.mean(0), X.std(0) + 1e-6).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * -(-len(vids) // BATCH))
    for ep in range(stop or epochs):
        order = [vids[i] for i in rng.permutation(len(vids))]
        tl, ty, tq = [], [], []
        for i in range(0, len(order), BATCH):
            opt.zero_grad()
            ls = []
            for v in order[i:i + BATCH]:
                x, t = data[v]
                r = R[v]
                if v in PRE and rng.random() < PPRE:  # pre-onset sub-crop of a positive: a same-scene negative
                    (x, t), r = PRE[v], r._replace(y=0)
                elif v in SUBS:  # one of the clip's sub-crops, extracted as its own clip
                    x, t = SUBS[v][rng.integers(len(SUBS[v]))]
                else:
                    c = crop(rng, t, r)
                    x, t = x[c], t[c]
                l, q = loss(model, x, t, r)
                ls.append(l); ty.append(r.y); tq.append(q)
            l = torch.stack(ls).mean()
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            sched.step()
            tl.append(float(l.detach()))
        msg = f"{tag} ep {ep:2d} train_loss {np.mean(tl):.4f} train_auc_crops {auroc(np.array(ty), np.array(tq)):.3f}"
        if len(val):
            vl, va = evaluate(model, data, R, val)
            msg += f" val_loss {vl:.4f} val_auc {va:.3f}"
            if curve is not None:
                curve.append(vl)
        log(msg)
    return model.eval()


@torch.no_grad()
def logits(model, data, vids):
    return {v: model(data[v][0]).cpu().numpy() for v in vids}


def rows(R, lg, data, fold=None):  # seed-averaged frame logits -> scorer rows
    out = []
    for v, ls in lg.items():
        q, t = score(np.mean(ls, 0), data[v][1])
        r = R[v]
        out.append(dict(video=v, fold=r.fold if fold is None else fold, y=r.y, g=r.g, dur=r.T, q=q, t=t,
                        night=r.night, dim=r.dim, event_type=r.event_type, scene=r.scene))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", choices=list(FEAT), required=True)
    ap.add_argument("--inner", action="store_true", help="epochs per outer fold from inner scene-grouped CV loss")
    a = ap.parse_args()
    inputs = a.inputs
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    run = ROOT / "runs" / f"s5_{inputs}{'_inner' if a.inner else ''}"
    run.mkdir(parents=True, exist_ok=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    cfg = dict(inputs=inputs, inner=a.inner, epochs=EPOCHS, batch=BATCH, lr=LR, wd=WD, clip=CLIP, seeds=SEEDS,
               crop_s=[LMIN, LMAX], sigma=SIGMA, git=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain", "src", "tools")),
               folds_sha256=sha(ROOT / "data" / "folds.csv"), index_sha256=sha(ROOT / "cache" / "index.csv"),
               features_meta_sha256=sha(FEAT[inputs] / "meta.csv"))
    (run / "config.json").write_text(json.dumps(cfg, indent=1))
    f = open(run / "train.log", "w")
    log = lambda s: (print(s, file=f, flush=True), print(s, flush=True))
    R = clips()
    data = load(inputs, R)
    oof, per_seed, train_rows = [], {s: [] for s in SEEDS}, []
    folds = sorted({r.fold for r in R.values()})
    for k in folds:
        tr, te = [v for v in R if R[v].fold != k], [v for v in R if R[v].fold == k]
        E = EPOCHS
        if a.inner:  # inner scene-grouped CV over the 4 training folds; pick epochs by mean inner val loss
            curves = []
            for j in (j for j in folds if j != k):
                for s in SEEDS:
                    curves.append([])
                    fit(data, R, [v for v in tr if R[v].fold != j], s, val=[v for v in tr if R[v].fold == j],
                        log=log, tag=f"{inputs} fold {k} inner {j} seed {s}", curve=curves[-1])
            mean = np.mean(curves, 0)
            E = cfg.setdefault("epochs_chosen", {})[int(k)] = int(mean.argmin()) + 1
            log(f"fold {k}: mean inner val loss by epoch {mean.round(4).tolist()} -> {E} epochs")
        lte, ltr = {v: [] for v in te}, {v: [] for v in tr}
        for s in SEEDS:
            m = fit(data, R, tr, s, val=te, log=log, tag=f"{inputs} fold {k} seed {s}", stop=E)
            torch.save(m.state_dict(), run / f"f{k}_s{s}.pt")
            for lg, vids in ((lte, te), (ltr, tr)):
                for v, l in logits(m, data, vids).items():
                    lg[v].append(l)
            per_seed[s] += rows(R, {v: [lte[v][-1]] for v in te}, data)
        oof += rows(R, lte, data)
        train_rows += rows(R, ltr, data, fold=k)  # fold = held-out fold of the models that made the prediction
    (run / "config.json").write_text(json.dumps(cfg, indent=1))
    pd.DataFrame(oof).sort_values("video").to_csv(run / "oof.csv", index=False)
    for s in SEEDS:
        pd.DataFrame(per_seed[s]).sort_values("video").to_csv(run / f"oof_seed{s}.csv", index=False)
    pd.DataFrame(train_rows).to_csv(run / "train_pred.csv", index=False)
    o = pd.read_csv(run / "oof.csv").sort_values("video").reset_index(drop=True)
    met = {k: report(o, i)[0] for k, i in (("int", True), ("float", False))}
    (run / "metrics.json").write_text(json.dumps(met, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    log(f"{inputs}: xfit F1 (--int) {met['int']['xfit_f1']}, AUROC {met['int']['auroc']}")


if __name__ == "__main__":
    main()
