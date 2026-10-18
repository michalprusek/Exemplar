#!/usr/bin/env python3
"""Golden comparison: the ORIGINAL AutoSeg benchmark against this package, on the same GPU, seed by seed.

    python tools/golden_compare.py --autoseg_tree /path/to/AutoSeg --datasets dsb2018,drive \
        --seeds 2 --gpu 0 --cache /path/to/golden_cache --out /path/to/golden_out

For the authors: AutoSeg, the research code that produced the paper's records, is not public. The result of this
comparison on a GPU is in docs/verification/.

Run it on a CUDA machine, in the Python environment of the reference runs (the same torch, transformers, numpy, scipy
and scikit-image), with the dataset environment variables of the reference runs (ASG_DATA_ROOT, SPHEROID_ROOT,
DSB_ROOT, PANEL_DL_ROOT). It runs, one after the other, each in its own process on the same GPU:

  (a) the original bench, `<autoseg_tree>/scripts/closed_form_bench.py`, with the paper's scored command for v4
      (`SCORED_FLAGS` below; the arms final10/ridge4_*). It runs from a copy written to <out>/orig/bench/ in which two
      lines capture what the bench never writes: `blocks` is wrapped so every image's inputs are fingerprinted, and
      `ridge_fit` is handed an empty refit `state` dict (its only effect: the per-image sums are kept until the seed
      ends) so the Gram sums and the column shift can be read; then the fitted readouts are saved. No arithmetic line
      is changed; both anchors must occur exactly once or the tool refuses.
  (b) this package's runner, `benchmark/run.py`, with the same datasets, K, seeds and GPU.

Each run gets its own coarse-view cache (<cache>/golden_orig_bf16, <cache>/golden_new_bf16; emptied first unless
--keep_cache), so both compute every input themselves. --shared_cache DIR makes both read one existing cache instead
(the coarse views are then equal by construction and everything downstream is compared).

Compared per (dataset, seed), in pipeline order, so the first mismatch locates a divergence:
  support indices -> magnification z -> inputs of every support image (coarse view, fine view, priors: sha1 of the
  float32 bytes) -> column shift -> per-image sums (G, Gf, bb, counts) -> lambda of each weighting -> blend weight a,
  variant, cut -> weights w (max abs diff) -> held-out per-image values; then the test inputs (every image at every
  decided z) and the records (per_image, alt plain_mse / bal_mse / blend_ex / blend1_ex, split fingerprint, zooms,
  lambdas). The environments (library versions, GPU, TF32 switches) must agree.

Ablation arms of Table 2: --blocks, --zoom_fixed and --last_layer_only go to both runs (the bench gets --blocks and
--zoom_fixed as given and the feature string without `_lf12` for --last_layer_only, as scripts/c117/ridge4_paper.sh
runs the arms). The bench's kept sums are then its column-selected ones, so the per-image sum fingerprints of a
--blocks arm are reported as WARN; everything from lambda on is compared as for the reported method.

PASS requires: identical choices (support, z, both lambdas, a, variant, cut), per-image test scores within --score_tol
(default 1e-6) for the primary readout and every common alt readout, and max|w_new - w_orig| / max|w_orig| within
--w_tol (default 1e-6). Input or sum fingerprints that differ are reported as WARN with their magnitudes (they
explain a failure; they cannot cause a PASS to be wrong).

Intermediates go to <out>/{orig,new}/capture/: capture.json (fingerprints, stats, choices), <ds>_s<seed>.npz (w of
every readout, column shift pc/ps); with --dump full also the arrays of seed 0 (support inputs, per-image sums) for
bisecting with numpy. Logs: <out>/{orig,new}/log.txt; the report: <out>/report.json. Exit 0 PASS, 1 FAIL, 2 error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
READOUTS = ("plain_mse", "bal_mse", "blend_ex", "blend1_ex")
CHOICE_FIELDS = ("lam", "lam_bal", "a", "variant", "tau")
FEAT = "head_fusion_best_nobank_noloss_nocolor_flat_es_ep500_wd4_do5_mix_noaux_fzu_fam_lf12_natres"
SCORED_FLAGS = ["--readout", "ridge_all", "--gram_mode", "grid", "--gram_dtype", "bfloat16", "--enc_amp", "bf16",
                "--solve_dtype", "float32", "--fast_fine", "--fast_zoom", "--ridge_variants", "plain,bal", "--ex_rule",
                "--primary", "blend1_ex", "--only_primary", "--score_rules", "mse,ex", "--lam_search", "newton",
                "--ex_metric", "dataset", "--loo_std", "fold", "--precision", "single", "--method_name", "ridge",
                "--blend", "--blend_rule", "both", "--feat_method", FEAT]
# (anchor line, replacement lines) in scripts/closed_form_bench.py; each anchor must occur exactly once
PATCHES = [
    ("    def precompute(be, images):",
     ["    blocks = _GOLDEN.wrap_blocks(blocks)", "", "    def precompute(be, images):"]),
    ("                fr, tms = ridge_fit(be, sup, rng, metric, check=a.gram_check and s == 0)",
     ["                _gst = {} if len(sup) >= 2 else None",
      "                fr, tms = ridge_fit(be, sup, rng, metric, check=a.gram_check and s == 0, state=_gst)",
      "                _GOLDEN.fit_orig(ds, s, sel, be, fr, _gst, sup)",
      "                del _gst"]),
]


# ---------------------------------------------------------------------------------------------------------------------
# capture (runs inside a worker process)

def _stats(t):
    """Fingerprint of a tensor or array: shape, sha1 of the float32 bytes, sum, max |x|."""
    if t is None:
        return None
    if hasattr(t, "detach"):
        a = t.detach().float().cpu().contiguous().numpy()
    else:
        a = np.ascontiguousarray(np.asarray(t, np.float32))
    h = hashlib.sha1(a.tobytes())
    h.update(str(a.shape).encode())
    return dict(shape=list(a.shape), sha1=h.hexdigest(), sum=float(a.astype(np.float64).sum()),
                absmax=float(np.abs(a).max()) if a.size else 0.0)


def _array(t):
    return None if t is None else (t.detach().float().cpu().numpy() if hasattr(t, "detach") else np.asarray(t))


def _image_key(image):
    a = np.ascontiguousarray(np.asarray(image))
    h = hashlib.sha1(a.tobytes())
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    return h.hexdigest()[:16]


def _environment():
    import platform

    import scipy
    import skimage
    import torch
    import transformers
    env = dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
               numpy=np.__version__, scipy=scipy.__version__, skimage=skimage.__version__,
               cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32), cudnn_tf32=bool(torch.backends.cudnn.allow_tf32),
               float32_matmul_precision=torch.get_float32_matmul_precision(),
               cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
               cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
               CUBLAS_WORKSPACE_CONFIG=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
               CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"))
    return env


class Capture:
    """Collects fingerprints of inputs, sums and readouts from either implementation."""

    def __init__(self, impl, out_dir, dump, k=8):
        self.impl, self.out_dir, self.dump, self.k = impl, out_dir, dump, k
        os.makedirs(out_dir, exist_ok=True)
        self.features_seen, self.fits, self.env, self.errors = {}, {}, None, []
        # dump full: the last K inputs read, as arrays; in both implementations the K reads just before a seed's fit
        # are its support images, so seed 0's are saved when its fit is recorded
        self.pending = {}

    # -- inputs ------------------------------------------------------------------------------------------------------
    def features(self, image, zoom, xc, fine, bank):
        try:
            key = f"{_image_key(image)}|{int(zoom)}"
            st = dict(xc=_stats(xc), fine=_stats(fine), bank=_stats(bank))
            prev = self.features_seen.get(key)
            if prev is None:
                self.features_seen[key] = dict(st, n=1, consistent=True)
            else:                                        # the same image read again at the same z: must not change
                prev["n"] += 1
                if any((prev[k] or {}).get("sha1") != (st[k] or {}).get("sha1") for k in ("xc", "fine", "bank")):
                    prev["consistent"] = False
            if self.dump == "full":
                self.pending.pop(key, None)
                self.pending[key] = dict(xc=_array(xc), fine=_array(fine), bank=_array(bank))
                while len(self.pending) > self.k:
                    self.pending.pop(next(iter(self.pending)))
        except Exception as e:                           # never let the capture change the run
            self.errors.append(f"features: {e!r}")

    def wrap_blocks(self, fn):
        """The bench's blocks(be, image) -> (xc, fine, bank, hw), fingerprinting its outputs."""
        def wrapped(be, image):
            out = fn(be, image)
            self.features(image, int(getattr(be, "fine_zoom", 1)), out[0], out[1], out[2])
            return out
        return wrapped

    def wrap_inputs(self, inputs):
        """This package's FrozenInputs(image, zoom), fingerprinting its outputs."""
        cap = self

        class _Hooked:
            def __getattr__(self, name):
                return getattr(inputs, name)

            def __call__(self, image, zoom):
                out = inputs(image, zoom)
                cap.features(image, zoom, out[0], out[1], out[2])
                return out
        return _Hooked()

    # -- fits --------------------------------------------------------------------------------------------------------
    def _record_fit(self, ds, s, sel, zoom, fr, sums, pc, ps, support_images):
        if self.env is None:
            self.env = _environment()
        tag = f"{ds}/{s}"
        rd = {}
        arrays = {}
        for k in READOUTS:
            r = fr[k]
            rd[k] = {f: (r.get(f) if not isinstance(r.get(f), np.generic) else r.get(f).item())
                     for f in CHOICE_FIELDS + ("heldout", "gain", "se") if f in r}
            if r.get("per_img") is not None:
                rd[k]["per_img"] = [float(x) for x in np.asarray(r["per_img"])]
            arrays[f"w_{k}"] = np.asarray(r["w"], np.float32)
        if pc is not None:
            arrays["pc"], arrays["ps"] = _array(pc), _array(ps)
        keys = [f"{_image_key(im)}|{int(zoom)}" for im in support_images]
        rec = dict(sel=[int(x) for x in sel], zoom=int(zoom), readouts=rd, support_keys=keys,
                   pc=_stats(pc), ps=_stats(ps),
                   sums=None if sums is None else [dict(G=_stats(g["G"]), Gf=_stats(g["Gf"]), bb=_stats(g["bb"]),
                                                        nn=float(g["nn"]), nf=float(g["nf"])) for g in sums])
        np.savez(os.path.join(self.out_dir, f"{ds}_s{s}.npz"), **arrays)
        if self.dump == "full" and s == 0:
            full = {}
            for i, k in enumerate(keys):
                for name, arr in (self.pending.get(k) or {}).items():
                    if arr is not None:
                        full[f"img{i}_{name}"] = arr
            for i, g in enumerate(sums or []):
                for name in ("G", "Gf", "bb"):
                    full[f"img{i}_{name}"] = _array(g[name])
            np.savez(os.path.join(self.out_dir, f"{ds}_s0_full.npz"), **full)
        self.pending.clear()
        self.fits[tag] = rec
        self.save()

    def fit_orig(self, ds, s, sel, be, fr, st, sup):
        """Called by the patched bench after each seed's ridge_fit."""
        try:
            import torch
            sums, pc, ps = None, None, None
            if st:
                sums = [dict(G=o["G"], Gf=o["Gf"] if o["Gf"] is not None else torch.zeros_like(o["G"]),
                             bb=o["bb"], nn=o["nn"], nf=o["nf"]) for o in st["imgs"]]
                pc, ps = st["pc"], st["ps"]
            self._record_fit(ds, s, sel, be.fine_zoom, fr, sums, pc, ps, [ex.image for ex in sup])
        except Exception as e:
            self.errors.append(f"fit_orig {ds}/{s}: {e!r}")
            self.save()

    def fit(self, ds, s, sel, model):
        """Called by benchmark/run.py after each seed's fit."""
        try:
            import torch
            st = model._state
            sums, pc, ps = None, None, None
            if st and st["incremental"]:
                sums = []
                for o in st["imgs"]:
                    g = o["groups"][0]
                    sums.append(dict(G=g["G"], Gf=g["Gf"] if g["Gf"] is not None else torch.zeros_like(g["G"]),
                                     bb=g["bb"] if g["bb"] is not None else torch.zeros(g["G"].shape[0], device=g["G"].device),
                                     nn=g["nn"], nf=g["nf"]))
                pc, ps = st["pc"], st["ps"]
            self._record_fit(ds, s, sel, model.zoom, model.result, sums, pc, ps, model.images)
        except Exception as e:
            self.errors.append(f"fit {ds}/{s}: {e!r}")
            self.save()

    def save(self):
        with open(os.path.join(self.out_dir, "capture.json"), "w") as f:
            json.dump(dict(impl=self.impl, env=self.env, fits=self.fits, features=self.features_seen,
                           errors=self.errors), f, default=float)


# ---------------------------------------------------------------------------------------------------------------------
# workers

def patch_bench(src):
    lines = src.split("\n")
    for anchor, repl in PATCHES:
        hits = [i for i, ln in enumerate(lines) if ln == anchor]
        if len(hits) != 1:
            raise SystemExit(f"golden_compare: the anchor {anchor.strip()!r} occurs {len(hits)} times in "
                             f"closed_form_bench.py (expected exactly once); this AutoSeg tree is not supported")
        lines[hits[0]:hits[0] + 1] = repl
    return "\n".join(lines)


def ablation_flags(a, bench: bool):
    """The ablation options of one run: the bench's flags (and feature string), or benchmark/run.py's."""
    out = []
    if a.blocks != "coarse,fine,bank":
        out += ["--blocks", a.blocks]
    if a.zoom_fixed:
        out += ["--zoom_fixed", str(a.zoom_fixed)]
    if a.last_layer_only and not bench:
        out += ["--last_layer_only"]
    return out


def bench_flags(a):
    """SCORED_FLAGS with this run's feature string (without `_lf12` for --last_layer_only) and ablation flags."""
    flags = list(SCORED_FLAGS)
    if a.last_layer_only:
        flags[flags.index("--feat_method") + 1] = FEAT.replace("_lf12", "")
    return flags + ablation_flags(a, bench=True)


def worker_orig(a):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)        # before torch is imported (the bench sets it too)
    tree = os.path.abspath(a.autoseg_tree)
    src = open(os.path.join(tree, "scripts", "closed_form_bench.py")).read()
    bdir = os.path.join(a.out, "orig", "bench", "scripts")
    os.makedirs(bdir, exist_ok=True)
    path = os.path.join(bdir, "closed_form_bench_golden.py")
    with open(path, "w") as f:
        f.write(patch_bench(src))
    sys.path[:0] = [os.path.join(tree, "scripts"), tree]
    cap = Capture("orig", os.path.join(a.out, "orig", "capture"), a.dump, a.k)
    sys.argv = [path] + bench_flags(a) + ["--seeds", str(a.seeds), "--k", str(a.k), "--datasets", a.datasets,
                                        "--gpu", str(a.gpu), "--cache", a.orig_cache,
                                        "--score_dir", os.path.join(a.out, "orig", "records")]
    print("[orig] " + " ".join(sys.argv), flush=True)
    try:
        runpy.run_path(path, init_globals={"_GOLDEN": cap}, run_name="__main__")
    finally:
        cap.save()


def worker_new(a):
    sys.path.insert(0, REPO)
    from benchmark import run as R                          # imports no torch; run.main sets CUDA_VISIBLE_DEVICES

    cap = Capture("new", os.path.join(a.out, "new", "capture"), a.dump, a.k)
    argv = ["--datasets", a.datasets, "--k", str(a.k), "--seeds", str(a.seeds), "--gpu", str(a.gpu),
            "--cache", a.new_cache, "--score_dir", os.path.join(a.out, "new", "records")] + ablation_flags(a, bench=False)
    print("[new] benchmark/run.py " + " ".join(argv), flush=True)
    try:
        R.main(argv, hook=cap)
    finally:
        cap.save()


def run_worker(which, a):
    log = os.path.join(a.out, which, "log.txt")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    cmd = [sys.executable, os.path.abspath(__file__), "--_worker", which] + a.forward
    t0 = time.time()
    with open(log, "w") as lf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             env=dict(os.environ, PYTHONUNBUFFERED="1"))
        for line in p.stdout:
            lf.write(line)
            sys.stdout.write(f"[{which}] {line}")
        p.wait()
    print(f"[{which}] exit {p.returncode} after {time.time() - t0:.0f} s (log {log})", flush=True)
    if p.returncode != 0:
        raise SystemExit(f"golden_compare: the {which} run failed (exit {p.returncode}); see {log}")


# ---------------------------------------------------------------------------------------------------------------------
# comparison

def _load(path):
    with open(path) as f:
        return json.load(f)


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return float("inf"), float("inf")
    d = float(np.abs(a - b).max()) if a.size else 0.0
    return d, d / max(float(np.abs(a).max()) if a.size else 0.0, 1e-30)


def compare(a):
    oc = _load(os.path.join(a.out, "orig", "capture", "capture.json"))
    nc = _load(os.path.join(a.out, "new", "capture", "capture.json"))
    fails, warns, rows = [], [], []

    def fail(msg):
        fails.append(msg)
        print("FAIL " + msg)

    def warn(msg):
        warns.append(msg)
        print("WARN " + msg)

    for c in (oc, nc):
        for e in c.get("errors", []):
            fail(f"capture error ({c['impl']}): {e}")
    print("\n== environment")
    for k in sorted(set(oc.get("env") or {}) | set(nc.get("env") or {})):
        vo, vn = (oc.get("env") or {}).get(k), (nc.get("env") or {}).get(k)
        print(f"  {k:28s} {vo!s:32s} {vn!s}")
        if vo != vn:
            fail(f"environment differs: {k} = {vo} (orig) vs {vn} (new)")

    datasets = a.datasets.split(",")
    for ds in datasets:
        print(f"\n== {ds}")
        for s in range(a.seeds):
            tag = f"{ds}/{s}"
            fo, fn = oc["fits"].get(tag), nc["fits"].get(tag)
            if fo is None or fn is None:
                fail(f"{tag}: fit missing ({'orig' if fo is None else 'new'})")
                continue
            row = dict(tag=tag)
            if fo["sel"] != fn["sel"]:
                fail(f"{tag}: support indices {fo['sel']} vs {fn['sel']}")
            if fo["zoom"] != fn["zoom"]:
                fail(f"{tag}: magnification {fo['zoom']} vs {fn['zoom']}")
            row["zoom"] = f"{fo['zoom']}/{fn['zoom']}"
            nfeat_same = 0
            for k in fo["support_keys"]:
                so, sn = oc["features"].get(k), nc["features"].get(k)
                if so is None or sn is None:
                    continue
                same = all((so[x] or {}).get("sha1") == (sn[x] or {}).get("sha1") for x in ("xc", "fine", "bank"))
                nfeat_same += same
                if not same:
                    for x in ("xc", "fine", "bank"):
                        if (so[x] or {}).get("sha1") != (sn[x] or {}).get("sha1"):
                            warn(f"{tag}: support input {x} differs at {k}: shape {so[x]['shape']} vs {sn[x]['shape']}, "
                                 f"sum {so[x]['sum']:.9g} vs {sn[x]['sum']:.9g}, max|x| {so[x]['absmax']:.6g} vs "
                                 f"{sn[x]['absmax']:.6g}")
            row["inputs"] = f"{nfeat_same}/{len(fo['support_keys'])}" if fo["support_keys"] else "-"
            if fo.get("pc") and fn.get("pc"):
                for x in ("pc", "ps"):
                    if fo[x]["sha1"] != fn[x]["sha1"]:
                        warn(f"{tag}: column shift {x} differs (sum {fo[x]['sum']:.9g} vs {fn[x]['sum']:.9g})")
            if fo.get("sums") and fn.get("sums"):
                bad = 0
                for i, (go, gn) in enumerate(zip(fo["sums"], fn["sums"])):
                    for x in ("G", "Gf", "bb"):
                        if go[x]["sha1"] != gn[x]["sha1"]:
                            bad += 1
                            warn(f"{tag}: image {i} sum {x} differs: sum {go[x]['sum']:.9g} vs {gn[x]['sum']:.9g}, "
                                 f"max {go[x]['absmax']:.6g} vs {gn[x]['absmax']:.6g}")
                    if go["nn"] != gn["nn"] or go["nf"] != gn["nf"]:
                        fail(f"{tag}: image {i} counts {go['nn']}/{go['nf']} vs {gn['nn']}/{gn['nf']}")
                row["sums"] = "identical" if not bad else f"{bad} differ"
            else:
                row["sums"] = "-"
            zo = np.load(os.path.join(a.out, "orig", "capture", f"{ds}_s{s}.npz"))
            zn = np.load(os.path.join(a.out, "new", "capture", f"{ds}_s{s}.npz"))
            if "pc" in zo.files and "pc" in zn.files:
                for x in ("pc", "ps"):
                    d, r = _rel(zo[x], zn[x])
                    if d:
                        warn(f"{tag}: column shift {x} max |diff| {d:.3g} (relative {r:.3g})")
            wmax = 0.0
            for k in READOUTS:
                ro, rn = fo["readouts"][k], fn["readouts"][k]
                for f in CHOICE_FIELDS:
                    if f in ro or f in rn:
                        if ro.get(f) != rn.get(f):
                            fail(f"{tag}: {k}.{f} = {ro.get(f)!r} (orig) vs {rn.get(f)!r} (new)")
                if "per_img" in ro and "per_img" in rn:
                    d, _ = _rel(ro["per_img"], rn["per_img"])
                    if d > a.score_tol:
                        fail(f"{tag}: {k} held-out per-image values differ by {d:.3g}")
                d, r = _rel(zo[f"w_{k}"], zn[f"w_{k}"])
                wmax = max(wmax, r)
                if r > a.w_tol:
                    fail(f"{tag}: {k} weights differ: max |dw| {d:.3g}, relative {r:.3g} > {a.w_tol:g}")
                elif d:
                    warn(f"{tag}: {k} weights not bit-identical: max |dw| {d:.3g} (relative {r:.3g})")
            b1o, b1n = fo["readouts"]["blend1_ex"], fn["readouts"]["blend1_ex"]
            row.update(lam=f"{b1o['lam']:.6g}/{b1n['lam']:.6g}",
                       lam_bal=f"{fo['readouts']['bal_mse']['lam']:.6g}/{fn['readouts']['bal_mse']['lam']:.6g}",
                       a=f"{b1o['a']:g}/{b1n['a']:g}", w_rel=f"{wmax:.2g}")
            rows.append(row)

        ro_path = os.path.join(a.out, "orig", "records", f"ridge__{ds}.json")
        rn_path = os.path.join(a.out, "new", "records", f"ridge__{ds}.json")
        if not (os.path.exists(ro_path) and os.path.exists(rn_path)):
            fail(f"{ds}: record missing ({ro_path if not os.path.exists(ro_path) else rn_path})")
            continue
        ro, rn = _load(ro_path), _load(rn_path)
        for f in ("metric", "test_per_seed", "seeds", "split_fp", "method"):
            if ro.get(f) != rn.get(f):
                fail(f"{ds}: record field {f}: {ro.get(f)!r} vs {rn.get(f)!r}")
        for f in ("zooms_decided", "lam_chosen", "tau_chosen", "support", "pool", "feat_method", "primary", "blocks",
                  "zoom_fixed"):
            if ro["protocol"].get(f) != rn["protocol"].get(f):
                fail(f"{ds}: protocol {f}: {ro['protocol'].get(f)!r} vs {rn['protocol'].get(f)!r}")
        d, _ = _rel(ro["per_image"], rn["per_image"])
        print(f"  per_image: {len(ro['per_image'])} scores, max |diff| {d:.3g}; mean {np.mean(ro['per_image']):.6f} "
              f"(orig) {np.mean(rn['per_image']):.6f} (new)")
        if d > a.score_tol:
            fail(f"{ds}: per-image test scores differ by {d:.3g} > {a.score_tol:g}")
        for k in READOUTS:
            if k in ro["alt"] and k in rn["alt"]:
                d, _ = _rel(ro["alt"][k], rn["alt"][k])
                if d > a.score_tol:
                    fail(f"{ds}: alt[{k}] test scores differ by {d:.3g} > {a.score_tol:g}")
            else:
                fail(f"{ds}: alt[{k}] missing in {'orig' if k not in ro['alt'] else 'new'}")
        for k, v in nc["features"].items():
            if not v.get("consistent", True):
                warn(f"new: inputs of image {k} changed between two reads (nondeterministic)")
        for k, v in oc["features"].items():
            if not v.get("consistent", True):
                warn(f"orig: inputs of image {k} changed between two reads (nondeterministic)")
    common = sorted(set(oc["features"]) & set(nc["features"]))
    same = [k for k in common if all((oc["features"][k][x] or {}).get("sha1") == (nc["features"][k][x] or {}).get("sha1")
                                     for x in ("xc", "fine", "bank"))]
    print(f"\n== inputs: {len(same)} of {len(common)} images (at their magnifications) bit-identical "
          f"({len(set(oc['features']) ^ set(nc['features']))} read by only one implementation)")
    if len(same) != len(common):
        warn(f"{len(common) - len(same)} images have inputs that are not bit-identical")
    print("\n== per seed (orig/new)")
    cols = ("tag", "zoom", "inputs", "sums", "lam", "lam_bal", "a", "w_rel")
    wid = {c: max([len(c)] + [len(str(r.get(c, "-"))) for r in rows]) for c in cols}
    print("  " + " | ".join(f"{c:>{wid[c]}s}" for c in cols))
    for r in rows:
        print("  " + " | ".join(f"{str(r.get(c, '-')):>{wid[c]}s}" for c in cols))
    verdict = "PASS" if not fails else "FAIL"
    report = dict(verdict=verdict, tolerances=dict(score_tol=a.score_tol, w_tol=a.w_tol), fails=fails, warns=warns,
                  rows=rows, datasets=datasets, seeds=a.seeds, k=a.k)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"\n{verdict}: {len(fails)} failure(s), {len(warns)} warning(s); tolerances: identical choices "
          f"(support, z, lambda, lambda_bal, a, variant, cut), per-image scores |diff| <= {a.score_tol:g}, "
          f"weights max|dw|/max|w| <= {a.w_tol:g}. Report: {os.path.join(a.out, 'report.json')}")
    return 0 if not fails else 1


# ---------------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--autoseg_tree", required=True, help="AutoSeg tree whose scripts/closed_form_bench.py is the reference")
    ap.add_argument("--datasets", required=True, help="comma list of panel datasets")
    ap.add_argument("--seeds", type=int, default=2, help="seeds 0 .. N-1 (as the bench's --seeds)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--gpu", default="0", help="the one GPU both runs use (CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--cache", required=True, help="base directory of the two coarse-view caches")
    ap.add_argument("--out", required=True, help="output directory (records, captures, logs, report)")
    ap.add_argument("--shared_cache", default="", help="read this existing coarse-view cache in both runs instead")
    ap.add_argument("--keep_cache", action="store_true", help="do not empty the two golden caches before the runs")
    ap.add_argument("--dump", default="summary", choices=("summary", "full"),
                    help="full: also save seed 0's support inputs and per-image sums as arrays (large)")
    ap.add_argument("--score_tol", type=float, default=1e-6)
    ap.add_argument("--w_tol", type=float, default=1e-6)
    ap.add_argument("--only", default="", choices=("", "orig", "new"), help="run only one side (then compare)")
    ap.add_argument("--compare_only", action="store_true", help="compare existing captures, run nothing")
    ap.add_argument("--blocks", default="coarse,fine,bank", help="ablation arm: the readout's column blocks (both runs)")
    ap.add_argument("--zoom_fixed", type=int, default=0, help="ablation arm: fixed magnification (both runs)")
    ap.add_argument("--last_layer_only", action="store_true", help="ablation arm: the last block only (both runs)")
    ap.add_argument("--_worker", default="", help=argparse.SUPPRESS)
    ap.add_argument("--orig_cache", default="", help=argparse.SUPPRESS)
    ap.add_argument("--new_cache", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    if a._worker == "orig":
        return worker_orig(a)
    if a._worker == "new":
        return worker_new(a)

    if not os.path.isfile(os.path.join(a.autoseg_tree, "scripts", "closed_form_bench.py")):
        sys.exit(f"golden_compare: no scripts/closed_form_bench.py under {a.autoseg_tree}")
    sys.path.insert(0, REPO)
    from benchmark.registry import ALL_DATASETS
    bad = [d for d in a.datasets.split(",") if d not in ALL_DATASETS]
    if bad:
        sys.exit(f"golden_compare: unknown dataset(s) {bad}; the panel is {ALL_DATASETS}")
    patch_bench(open(os.path.join(a.autoseg_tree, "scripts", "closed_form_bench.py")).read())   # refuse early
    if a.shared_cache:
        a.orig_cache = a.new_cache = os.path.abspath(a.shared_cache)
    else:
        a.orig_cache = os.path.join(os.path.abspath(a.cache), "golden_orig")
        a.new_cache = os.path.join(os.path.abspath(a.cache), "golden_new")
    a.forward = ["--autoseg_tree", os.path.abspath(a.autoseg_tree), "--datasets", a.datasets, "--seeds", str(a.seeds),
                 "--k", str(a.k), "--gpu", str(a.gpu), "--cache", a.cache, "--out", a.out, "--dump", a.dump,
                 "--orig_cache", a.orig_cache, "--new_cache", a.new_cache, "--blocks", a.blocks,
                 "--zoom_fixed", str(a.zoom_fixed)] + (["--last_layer_only"] if a.last_layer_only else [])
    if not a.compare_only:
        for which in (("orig", "new") if not a.only else (a.only,)):
            for ds in a.datasets.split(","):              # a record left from an earlier run would be skipped
                rp = os.path.join(a.out, which, "records", f"ridge__{ds}.json")
                if os.path.exists(rp):
                    os.remove(rp)
            cdir = os.path.join(a.out, which, "capture")
            if os.path.isdir(cdir):
                shutil.rmtree(cdir)
            if not a.shared_cache and not a.keep_cache:
                cache = (a.orig_cache if which == "orig" else a.new_cache) + "_bf16"
                if os.path.isdir(cache):
                    shutil.rmtree(cache)
            run_worker(which, a)
    return compare(a)


if __name__ == "__main__":
    sys.exit(main())
