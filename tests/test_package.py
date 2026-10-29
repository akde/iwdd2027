"""Stage 7 test (wiki/baseline.md; wiki/plan.md section 8; E3r package): python tests/test_package.py

0. src/decision.py holds the shipped run's operating point: tools/operating_point.py --variant VARIANT on runs/RUN/oof.csv
   prints TAU and DELTA; submission/model/MANIFEST.json names RUN and the SHA256 of its 15 checkpoint files.
POLICY "model": parts 1-7. POLICY "constant": parts 4 and 5.
1. On 2 clips (vid0020, vid0426) the package's per-fold logits equal a reference built from cache/frames5 (the training
   frames) with each checkpoint loaded on its own, as E3.logits computed the OOF; its q, t equal oof.csv; full mode =
   sigmoid(mean over folds of the fold's video logit), argmax of the mean logits (wiki/plan.md section 8).
2. Per-fold mode (--folds) on the 500 clips + a corrupt file, --videos without a trailing slash, --results a missing
   nested dir: every TXT equals "" if q < TAU else to_output(t, DELTA, dur) from runs/RUN/oof.csv q, t.
3. Full mode on the same 501 files: the corrupt file's TXT is empty; prints the emission rate on the 500; writes
   reports/stage7/full_txt.csv.
4. POLICY "constant": one pass over the 500 clips + the corrupt file: every TXT is floor(min(13, T)), T from
   cache/index.csv; the corrupt file (no readable duration) gets an empty TXT.
5. No CUDA device (CUDA_VISIBLE_DEVICES=""): policy constant, vid0020 -> floor(min(13, T)), failures 0.
6. A checkpoint that fails its SHA256 (a copy of the package): "model load failed", every clip floor(min(13, T)).
7. Long clips: the chunked tail (W_MAX lowered to 10 windows) gives vid0426's q within 1e-3 of the unchunked one.
Logs of test.py: reports/stage7/test_py_<mode>.log.
E3f (wiki/experiments.md; added parts), when MANIFEST.json holds "full" (the refit; RUN is then the
fixed-epoch CV run, whose fold models parts 0-7 check as before):
0b. "full" lists exactly the 15 g<k>_s<s>.pt; the refit's config is E3f with RUN's recipe (form, augmentation, learning
   rates, weight decay, clip, batch, folds, features, synthetic index) and epochs = RUN's fixed epochs; the SHA256 of
   runs/<refit>/insample_full.csv matches.
8a. On vid0020 and vid0426 each refit group's logits equal a reference from cache/frames5 with each g-file loaded on its
   own; full-mode q, t equal insample_full.csv (written at training time by finetune.full through the same code).
8b. Full mode on the 500 clips: every TXT equals "" if q < TAU else to_output(t, DELTA, dur) from insample_full.csv.
8c. A refit file that fails its SHA256: "model load failed", every clip floor(min(13, T)).
"""
import importlib.util, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "submission" / "test.py"
spec = importlib.util.spec_from_file_location("package", TEST)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)  # puts src/ on sys.path
import finetune as E
from decision import DELTA, RUN, TAU, VARIANT, to_output
from features import load_movinet
from head import Head, score

idx = pd.read_csv(ROOT / "cache" / "index.csv", float_precision="round_trip")
path = {r.video: ROOT / "data" / r.split / "videos" / f"{r.video}.mp4" for r in idx.itertuples()}
const = {f"{r.video}.mp4.txt": str(int(to_output(13.0, 0.0, r.T))) for r in idx.itertuples()}
LOGS = ROOT / "reports" / "stage7"
LOGS.mkdir(parents=True, exist_ok=True)
oof = pd.read_csv(ROOT / "runs" / RUN / "oof.csv", float_precision="round_trip").set_index("video")
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False


def run_test(test, vids, res, *extra, env=None):
    r = subprocess.run([sys.executable, str(test), "--videos", str(vids), "--results", str(res), *extra],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-1000:]
    return r.stdout + r.stderr, {n: (Path(res) / n).read_text() for n in os.listdir(res)}


# 0. the decision layer and the manifest hold the shipped run
op = subprocess.run([sys.executable, "operating_point.py", f"../runs/{RUN}/oof.csv", "--variant", VARIANT],
                    cwd=ROOT / "tools", capture_output=True, text=True, check=True).stdout
assert f"TAU = {TAU:.3f}, DELTA = {DELTA}" in op, (TAU, DELTA, op[-200:])
man = json.loads((P.MODEL / "MANIFEST.json").read_text())
assert man["run"] == RUN and sorted(man["files"]) == sorted(f"f{k}_s{s}.pt" for k in range(5) for s in range(3))
print(f"0. {RUN} ({VARIANT}): tools/operating_point.py gives src/decision.py TAU {TAU}, DELTA {DELTA}; MANIFEST: OK")
FULL = man.get("full")  # E3f refit: parts 0b, 8a-8c
if FULL:
    import hashlib
    assert sorted(FULL["files"]) == sorted(f"g{k}_s{s}.pt" for k in range(5) for s in range(3)), sorted(FULL["files"])
    cf, cr = (json.loads((ROOT / "runs" / r / "config.json").read_text()) for r in (FULL["run"], RUN))
    same = ("form", "aug", "lr_tail", "lr_head", "wd", "clip", "batch", "folds_sha256", "features_meta_sha256")
    syn = lambda c: (c["synthetic"]["index_sha256"], c["synthetic"]["cap"]) if c.get("synthetic") else None
    assert cf["experiment"] == "E3f" and cf["epochs"] == cr["epochs_fixed"] == FULL["epochs"], (cf, cr)
    assert all(cf[k] == cr[k] for k in same) and syn(cf) == syn(cr), [k for k in same if cf[k] != cr[k]]
    ins_csv = ROOT / "runs" / FULL["run"] / "insample_full.csv"
    assert hashlib.sha256(ins_csv.read_bytes()).hexdigest() == FULL["insample_sha256"]
    ins = pd.read_csv(ins_csv, float_precision="round_trip").set_index("video")
    want_full = {f"{v}.mp4.txt": str(int(to_output(r.t, DELTA, r.dur))) if r.q >= TAU else "" for v, r in ins.iterrows()}
    assert len(want_full) == 500
    print(f"0b. refit {FULL['run']}: 15 g-files, E3f with {RUN}'s recipe, {FULL['epochs']} epochs, insample_full.csv: OK")

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    vids = tmp / "videos"
    vids.mkdir()
    for p in path.values():
        (vids / p.name).symlink_to(p)
    (vids / "corrupt.mp4").write_bytes(np.random.default_rng(0).bytes(100_000))
    want = {f"{v}.mp4.txt": str(int(to_output(r.t, DELTA, r.dur))) if r.q >= TAU else "" for v, r in oof.iterrows()}
    assert len(want) == 500

    if P.POLICY == "model":
        # 1. per-fold logits = reference from the training frames, each checkpoint on its own; q, t = oof.csv
        model = P.load_models()
        base = load_movinet()
        trunk = E.split(base)
        for v in ("vid0020", "vid0426"):
            k = int(oof.fold[v])
            lg, t5, T = P.fold_logits(*model, path[v], [k])
            x = np.load(ROOT / "cache" / "frames5" / f"{v}.npy")
            with torch.no_grad():
                h = E.trunk_out(trunk, E.to_x(x), wb=1)
                ref = []
                for s in range(3):
                    ck = torch.load(P.MODEL / f"f{k}_s{s}.pt", map_location="cpu", weights_only=True)
                    tail, head = E.new_tail(base), Head(640, torch.zeros(640), torch.ones(640)).to(P.DEV)
                    tail.load_state_dict(ck["tail"]); head.load_state_dict(ck["head"])
                    ref.append(head.eval()(E.head_in(tail.eval(), h, len(x), "m")).cpu().numpy())
            assert np.array_equal(lg[k], np.mean(ref, 0)), v
            assert score(lg[k], t5) == (oof.q[v], oof.t[v]) and T == oof.dur[v], (v, score(lg[k], t5), oof.loc[v])
            full, _, _ = P.fold_logits(*model, path[v])
            vl = [float(torch.as_tensor(l).topk(min(4, len(l))).values.mean()) for l in full.values()]
            q, t = P.full_score(full, t5)
            assert np.isclose(q, 1 / (1 + np.exp(-np.mean(vl)))) and t == t5[np.argmax(np.mean(list(full.values()), 0))]
            print(f"1. {v}: per-fold logits == reference from cache/frames5 ({len(x)} rows), q, t == oof.csv; "
                  f"full mode q {q:.4f}, t {t}: OK")
        # 7. long clips: chunked tail
        lg0, t5, _ = P.fold_logits(*model, path["vid0426"])
        P.W_MAX = 10
        lg1, _, _ = P.fold_logits(*model, path["vid0426"])
        P.W_MAX = 240
        dq = abs(P.full_score(lg0, t5)[0] - P.full_score(lg1, t5)[0])
        assert dq < 1e-3, dq
        print(f"7. chunked tail on vid0426 ({len(lg0[0])} rows): |dq| {dq:.1e} < 1e-3: OK")
        if FULL:  # 8a. refit groups: logits = reference from cache/frames5, each g-file on its own; q, t = insample
            fm = P.load_models(full=True)
            for v in ("vid0020", "vid0426"):
                lg, t5, T = P.fold_logits(*fm, path[v])
                x = np.load(ROOT / "cache" / "frames5" / f"{v}.npy")
                with torch.no_grad():
                    h = E.trunk_out(trunk, E.to_x(x), wb=1)
                    for k in range(5):
                        ref = []
                        for s in range(3):
                            ck = torch.load(P.MODEL / f"g{k}_s{s}.pt", map_location="cpu", weights_only=True)
                            tail, head = E.new_tail(base), Head(640, torch.zeros(640), torch.ones(640)).to(P.DEV)
                            tail.load_state_dict(ck["tail"]); head.load_state_dict(ck["head"])
                            ref.append(head.eval()(E.head_in(tail.eval(), h, len(x), "m")).cpu().numpy())
                        assert np.array_equal(lg[k], np.mean(ref, 0)), (v, k)
                q, t = P.full_score(lg, t5)
                assert (q, t) == (ins.q[v], ins.t[v]) and T == ins.dur[v], (v, q, t, ins.loc[v])
                print(f"8a. {v}: refit group logits == reference from cache/frames5; full-mode q {q:.4f}, t {t} == "
                      f"insample_full.csv: OK")
            del fm
        del model, base, trunk
        torch.cuda.empty_cache()

        # 2, 3. test.py end to end
        for mode, extra in (("per_fold", ["--folds", str(ROOT / "data" / "folds.csv")]), ("full", [])):
            res = tmp / mode / "nested"
            log, got = run_test(TEST, vids, res, *extra)
            (LOGS / f"test_py_{mode}.log").write_text(log)
            assert set(got) == set(want) | {"corrupt.mp4.txt"} and got["corrupt.mp4.txt"] == "", (mode, len(got))
            assert "policy model; failures 1" in log, log[-300:]  # the corrupt file
            emit = {n: got[n] != "" for n in want}
            if mode == "per_fold":
                bad = [n for n in want if got[n] != want[n]]
                assert not bad, (len(bad), [(n, got[n], want[n]) for n in bad[:10]])
                print(f"2. per-fold mode: all 500 TXTs equal to_output from {RUN}/oof.csv (emits on "
                      f"{sum(emit.values())}); corrupt file -> empty TXT: OK")
            else:
                y = dict(zip(idx.video + ".mp4.txt", idx.y))
                rate = lambda c: np.mean([emit[n] for n in want if y[n] in c])
                print(f"3. full mode: emits on {sum(emit.values())}/500 = {rate((0, 1)):.3f} (positives "
                      f"{rate((1,)):.3f}, negatives {rate((0,)):.3f}); corrupt file -> empty TXT")
                pd.DataFrame({"file": sorted(got), "txt": [got[n] for n in sorted(got)]}).to_csv(
                    LOGS / "full_txt.csv", index=False)
                if FULL:  # 8b. full mode = the refit: every TXT from insample_full.csv
                    bad = [n for n in want_full if got[n] != want_full[n]]
                    assert "models: g<k>_s<s>.pt of " + FULL["run"] in log and not bad, (len(bad), bad[:10])
                    print(f"8b. full mode: all 500 TXTs equal to_output from {FULL['run']}/insample_full.csv: OK")

        # 6. a checkpoint that fails its SHA256 -> the constant for every clip
        pkg = tmp / "pkg"
        for d in ("submission", "src", "vendor/movinet-c2d1edf"):
            shutil.copytree(ROOT / d, pkg / d, ignore=shutil.ignore_patterns("__pycache__"))
        (pkg / "tools").mkdir()
        shutil.copy(ROOT / "tools" / "iwdd_eval.py", pkg / "tools")
        with open(pkg / "submission" / "model" / "f2_s1.pt", "ab") as f:
            f.write(b"\0")
        three = tmp / "three"
        three.mkdir()
        for v in ("vid0020", "vid0426", "vid0064"):
            (three / f"{v}.mp4").symlink_to(path[v])
        log, got = run_test(pkg / "submission" / "test.py", three, tmp / "bad_sha")
        assert "model load failed" in log and "SHA256 mismatch" in log and "policy constant; failures 0" in log, log[-500:]
        assert all(got[n] == const[n] for n in got) and len(got) == 3, got
        print(f"6. a checkpoint with a wrong SHA256: model load failed, constant on 3 clips {got}: OK")
        if FULL:  # 8c. a refit file with a wrong SHA256 (f2_s1.pt restored)
            shutil.copyfile(P.MODEL / "f2_s1.pt", pkg / "submission" / "model" / "f2_s1.pt")
            with open(pkg / "submission" / "model" / "g2_s1.pt", "ab") as f:
                f.write(b"\0")
            log, got = run_test(pkg / "submission" / "test.py", three, tmp / "bad_sha_g")
            assert "model load failed" in log and "g2_s1.pt: SHA256 mismatch" in log and "policy constant; failures 0" in log, log[-500:]
            assert all(got[n] == const[n] for n in got) and len(got) == 3, got
            print(f"8c. a refit file with a wrong SHA256: model load failed, constant on 3 clips: OK")
    else:
        # 4. constant policy
        log, got = run_test(TEST, vids, tmp / "constant" / "nested")
        (LOGS / "test_py_constant.log").write_text(log)
        assert set(got) == set(const) | {"corrupt.mp4.txt"} and got["corrupt.mp4.txt"] == ""
        bad = [n for n in const if got[n] != const[n]]
        assert not bad, bad[:10]
        print("4. constant mode: all 500 TXTs = floor(min(13, T)); corrupt file -> empty TXT: OK")

    # 5. no CUDA device: the constant, no failure
    cpu = tmp / "cpu"
    (cpu / "v").mkdir(parents=True)
    (cpu / "v" / "vid0020.mp4").symlink_to(path["vid0020"])
    log, got = run_test(TEST, cpu / "v", cpu / "r", env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert got == {"vid0020.mp4.txt": const["vid0020.mp4.txt"]} and "policy constant; failures 0" in log, log[-300:]
    print(f"5. no CUDA: vid0020 -> '{const['vid0020.mp4.txt']}' (the constant), failures 0: OK")
print("stage 7 test passed")
