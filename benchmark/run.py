#!/usr/bin/env python3
"""The paper's protocol for Exemplar: K support masks drawn from a fixed pool per seed, the full test split scored.

    python -m benchmark.run --datasets drive,hrf --k 8 --seeds 10 --gpu 0 --cache /data/cache --score_dir out/ridge_k8

Per dataset: the pool and the test split are loaded once (seed 0). Per seed s: rng = default_rng(s) draws the K support
indices from the pool, and the SAME generator then draws the column-shift sample of the readout; the magnification is
decided from the K masks and the readout fitted. The test images are then read once per decided magnification and
every seed with that magnification is scored on them (IoU, or centreline Dice on DRIVE, HRF, ISBI2012-EM and FISBe).

Records: <score_dir>/ridge__<dataset>.json in the AutoSeg benchmark's format (per_image seed-major, the primary readout
blend1_ex; alt = blend1_ex, blend_ex, plain_mse and bal_mse per test image; split fingerprint; protocol), so the
paper's table generators read them unchanged. An existing record is skipped. Table 2's readout rows "plain readout,
threshold 0.5" and "balanced readout, threshold 0.5" are alt["plain_mse"] and alt["bal_mse"] of the reported arm.

Table 2's other rows are separate runs with one ablation option each (scripts/c117/ridge4_paper.sh in AutoSeg; the
records carry it in protocol.blocks, protocol.zoom_fixed and protocol.feat_method, which the ablation table checks):

    --blocks coarse,fine       coarse + fine view only        (ridge4_nocls_k8)
    --blocks bank              priors only                     (ridge4_bankonly_k8)
    --blocks coarse,bank       coarse view + priors            (ridge4_coarseonly_k8)
    --blocks fine,bank         fine view + priors              (ridge4_fineonly_k8)
    --last_layer_only          last layer only (no block 12)   (ridge4_nolf12_k8)
    --zoom_fixed 1             fine view not magnified         (ridge4_nozoom_k8)

Source: scripts/closed_form_bench.py (AutoSeg) main(), the --readout ridge_all path (2349-2483, 2764-2907).
Line references point to the authors' research code, which is not public.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.registry import ALL_DATASETS, CLDICE, POOL, load_split   # noqa: E402  (no torch import here)

METHOD_NAME = "ridge"
PRIMARY = "blend1_ex"
COMBOS = ("plain_mse", "bal_mse", "blend_ex", "blend1_ex")   # readouts scored on the test images (alt)
# The research code's name of the reported inputs, a constant field of every record: the paper's table generators check
# it (Table 2's "last layer only" row is the same string without `_lf12`), so the records carry it unchanged.
FEAT_METHOD = "head_fusion_best_nobank_noloss_nocolor_flat_es_ep500_wd4_do5_mix_noaux_fzu_fam_lf12_natres"
LAM_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]                    # recorded only, as the bench does (closed_form_bench.py:58)
BLOCKS_ALL = "coarse,fine,bank"                              # --blocks default: the reported method


def feat_method(last_layer_only: bool) -> str:
    """The AutoSeg feature string the record carries: without `_lf12` when only the last block is read."""
    return FEAT_METHOD.replace("_lf12", "") if last_layer_only else FEAT_METHOD


def split_fingerprint(pairs) -> str:
    """Content digest of a loaded split; byte-compatible with the AutoSeg records, whose table generators refuse to
    pair two records with different fingerprints. (closed_form_bench.py:61-71)"""
    h = hashlib.sha256()
    for image, label in pairs:
        for a in (np.ascontiguousarray(image), np.ascontiguousarray(label)):
            h.update(str(a.shape).encode())
            h.update(str(a.dtype).encode())
            h.update(a.tobytes())
    return h.hexdigest()[:16]


def run_dataset(ds, a, inputs, device, device_name, hook=None):
    """Fit every seed on one dataset, score the test split, write the record. hook (tools/golden_compare.py): an
    object with fit(ds, seed, sel, model), called after each fit. a.blocks, a.zoom_fixed: the ablation options
    (the reported method: coarse,fine,bank and 0); a.last_layer_only only names the inputs' layers in the record."""
    import torch
    from concurrent.futures import ThreadPoolExecutor

    from benchmark.metrics import cldice as cldice_fn, foreground_iou as iou_fn
    from exemplar import Exemplar
    from exemplar.readout import logit_map

    metric = "cldice" if ds in CLDICE else "fg_iou"
    blocks = getattr(a, "blocks", BLOCKS_ALL)            # an argument namespace built by hand: the reported method
    zoom_fixed = int(getattr(a, "zoom_fixed", 0) or 0)
    use_fine = "fine" in blocks.split(",")
    out_path = os.path.join(a.score_dir, f"{a.method_name}__{ds}.json")
    if os.path.exists(out_path):
        print(f"[skip] {ds}: record exists", flush=True)
        return None
    if a.k > POOL.get(ds, 20):                           # a pool smaller than K
        print(f"[skip] {ds}: the pool of {POOL.get(ds, 20)} cannot supply K={a.k}", flush=True)
        return None
    pool, test = load_split(ds)                          # raises on a short pool or an empty test split
    if a.k > len(pool):
        raise RuntimeError(f"{ds}: pool of {len(pool)} images cannot supply K={a.k}")
    fp = split_fingerprint(test)
    counters = {}
    per_image, seeds_done, lams, taus_used, zooms, stage1_ts, fit_detail = [], [], [], [], [], [], []
    pend, by_zoom = [], {}
    t_fit_total = 0.0
    for s in range(a.seeds):
        rng = np.random.default_rng(s)
        sel = rng.choice(len(pool), a.k, replace=False)
        images = [pool[i][0] for i in sel]
        masks = [np.asarray(pool[i][1]) for i in sel]
        for im in images:                                # the coarse views first, in the bench's order
            inputs.coarse(im)
        model = Exemplar(device, metric="cldice" if metric == "cldice" else "iou", inputs=inputs,
                         blocks=blocks, zoom_fixed=zoom_fixed or None)
        model.counters = counters                        # single-precision counters are per dataset
        t0 = time.time()
        model.fit(images, masks, rng=rng)
        t_fit = time.time() - t0
        t_fit_total += t_fit
        if hook is not None:
            hook.fit(ds, s, sel, model)
        model.release_support()
        fr = model.result
        pend.append(dict(s=s, fr=fr, zoom=int(model.zoom)))
        by_zoom.setdefault(int(model.zoom), True)
        seeds_done.append(s)
        zooms.append(int(model.zoom))
        lams.append(fr[PRIMARY]["lam"])
        taus_used.append(fr[PRIMARY]["tau"])
        stage1_ts.append(round(t_fit, 1))
        fit_detail.append(dict(total_s=round(t_fit, 1), gate=dict(value=None, metric=metric),
                               chosen={k: dict(lam=fr[k]["lam"], tau=round(fr[k]["tau"], 4), heldout=fr[k]["heldout"],
                                               **{k2: fr[k][k2] for k2 in ("variant", "gain", "se", "a", "lam_bal")
                                                  if k2 in fr[k]})
                                       for k in COMBOS},
                               newton={v: fr[f"{v}_mse"]["newton"] for v in ("plain", "bal")}))
        view = f"fine view at {model.zoom}x" if use_fine else "no fine view"
        print(f"  [{ds}] seed {s}: {view}, fit {t_fit:.1f} s, a={fr[PRIMARY]['a']:g} "
              f"lam={fr[PRIMARY]['lam']:.4g} lam_bal={fr['bal_mse']['lam']:.4g} held-out {fr[PRIMARY]['heldout']:.4f}",
              flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # the test images are read once per magnification for all the seeds that decided it (closed_form_bench.py:2764-2864)
    t_ev = time.time()
    sc = {p["s"]: {k: [None] * len(test) for k in COMBOS} for p in pend}
    workers = ThreadPoolExecutor(max_workers=int(os.environ.get("OMP_NUM_THREADS", "8")))
    dump = set(x for x in a.dump_qual.split(",") if x)
    for zf in by_zoom:
        grp = [p for p in pend if p["zoom"] == zf]
        for ti, (im, lb) in enumerate(test):
            im = np.asarray(im)
            gt = np.asarray(lb) > 0
            xc, fine, bank, hw = inputs(im, zf)
            if not use_fine:                             # a readout without the fine view never reads it (487)
                fine = None
            jobs = []
            for p in grp:
                fr, maps = p["fr"], {}
                for k in COMBOS:
                    wkey = (fr[k].get("variant", k.split("_")[0]), fr[k]["lam"], fr[k].get("a"))
                    if wkey not in maps:
                        maps[wkey] = logit_map(xc, fine, bank, hw, fr[k]["w"])
                    lg = maps[wkey]
                    if lg.shape != im.shape[:2]:
                        raise RuntimeError(f"{ds} test {ti}: score map {lg.shape} vs image {im.shape[:2]}")
                    jobs.append((p["s"], k, lg > fr[k]["tau"]))
            if f"{ds}_{ti}" in dump:                     # the qualitative figure's panels: seed 0, primary readout
                for s_, k_, fg_ in jobs:
                    if s_ == 0 and k_ == PRIMARY:
                        os.makedirs(a.dump_qual_dir, exist_ok=True)
                        np.savez_compressed(os.path.join(a.dump_qual_dir, f"{ds}_{ti}.npz"), image=im,
                                            gt=np.asarray(lb).astype(np.int32), pred_fg=np.asarray(fg_).astype(np.uint8),
                                            ds=ds, metric=metric, method=a.method_name, seed=0, k=a.k)
            vals = workers.map(lambda j: float(cldice_fn(j[2], gt) if metric == "cldice" else iou_fn(j[2], gt)), jobs)
            for (s_, k, _), v in zip(jobs, vals):
                sc[s_][k][ti] = v
            del xc, fine, bank
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    workers.shutdown()
    alt = {}
    for p in pend:
        for k in sorted(COMBOS):
            alt.setdefault(k, []).extend(sc[p["s"]][k])
        per_image.extend(sc[p["s"]][PRIMARY])
        view = f"fine view {p['zoom']}x" if use_fine else "no fine view"
        print(f"  [{ds}] seed {p['s']} ({view}): "
              + " ".join(f"{k}={np.mean(sc[p['s']][k]):.4f}" for k in COMBOS), flush=True)
    print(f"  [{ds}] test pass {time.time() - t_ev:.0f} s for {len(pend)} seeds, {len(by_zoom)} magnification(s)",
          flush=True)

    from exemplar import __version__
    lam_rule = ("lambda (and, per rule, the threshold) by held-out quality of each support image"
                if a.k >= 2 else "lambda by held-out quality of each QUADRANT of the single support image")
    note = ("fully closed-form ridge over EVERY support pixel (per-image Gram matrices exact on the patch grids): coarse + "
            "fine DINOv3 and the 35 native priors. per_image = blend1_ex (a w_plain + (1 - a) w_bal cut at 1/2, a by the "
            "exact held-out metric under the one-SE rule); alt = blend1_ex, blend_ex (the best a without the one-SE rule), "
            "plain_mse and bal_mse (each weighting at 1/2, lambda by the exact held-out squared error). "
            "No gradient step anywhere.")
    # The protocol keeps the research benchmark's full field set. px_per_image, lam_grid, lam_rule, tau_bins, rf_dim,
    # skel_weight, stage2 and the other fixed values below are constant fields (options of that benchmark which the
    # reported configuration does not use, or fixed descriptions); they are kept so that the paper's table generators
    # read these records exactly as they read the paper's.
    rec = dict(method=a.method_name, dataset=ds, metric=metric, test_per_seed=len(test),
               seeds=seeds_done, per_image=per_image, split_fp=fp,
               protocol=dict(support=a.k, pool=POOL.get(ds, 20), test=10000, res=672,
                             blend=dict(rule="grid", nogrid=True, test_grid=False),
                             px_per_image=20000, lam_grid=LAM_GRID, lam_rule=lam_rule,
                             lam_chosen=lams, tau_rule=PRIMARY, tau_chosen=taus_used,
                             gradient_steps=0, feat_method=feat_method(getattr(a, "last_layer_only", False)),
                             loo_rows=-1,
                             zooms_decided=zooms, fit_s_mean=t_fit_total / max(1, len(seeds_done)),
                             lam_select="loo", lam_check=[], stage2="none",
                             lam2_chosen=[], stage1_s=stage1_ts, features_s=[],
                             skel_weight=0.0, gram_dtype="bfloat16", tau_search="grid",
                             balance=False, rf_dim=0,
                             tau_s=[], readout="ridge_all", enc_amp="bf16", solve_dtype="float32",
                             primary=PRIMARY, tau_bins=512, only_primary=True, device=device_name,
                             ridge_lams="1e-4,1e-3,1e-2,1e-1,1,10,100", gram_mode="grid",
                             lam_search="newton", ex_metric="dataset", blocks=blocks,
                             zoom_fixed=zoom_fixed, loo_std="fold", precision="single", weighting="select",
                             single_fallbacks=counters.get("single_fallbacks", 0),
                             single_rejected_evals=counters.get("single_rejected_evals", 0),
                             implementation=f"exemplar {__version__}", fit_detail=fit_detail),
               alt=alt, note=note)
    os.makedirs(a.score_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rec, f)
    print(f"[{ds}] {np.mean(per_image):.4f} over {len(seeds_done)} seeds "
          f"({t_fit_total / max(1, len(seeds_done)):.1f} s/fit) -> {out_path}", flush=True)
    return rec


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", default=",".join(ALL_DATASETS), help="comma list of panel datasets")
    ap.add_argument("--k", type=int, default=8, help="support masks per seed")
    ap.add_argument("--seeds", type=int, default=10, help="seeds 0 .. N-1")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES for this run")
    ap.add_argument("--cache", default="", help="coarse-view cache directory (the run uses <cache>_bf16); "
                                                "empty: no disk cache")
    ap.add_argument("--score_dir", required=True)
    ap.add_argument("--method_name", default=METHOD_NAME, help="the name the records carry")
    ap.add_argument("--dump_qual", default="", help="comma list of <dataset>_<test index>: seed 0's prediction is "
                                                     "saved as <dump_qual_dir>/<key>.npz")
    ap.add_argument("--dump_qual_dir", default="")
    ap.add_argument("--blocks", default=BLOCKS_ALL,
                    help="ablation: the column blocks the readout may use, a comma list of coarse, fine, bank; the "
                         "others are removed from the sums before lambda and the blend are chosen (a list without "
                         "fine never computes the fine view)")
    ap.add_argument("--zoom_fixed", type=int, default=0,
                    help="ablation: the fine view at this magnification on every support set instead of the one "
                         "decided from the masks (0: decided; 1: 'fine view not magnified')")
    ap.add_argument("--last_layer_only", action="store_true",
                    help="ablation 'last layer only': both views read the last block of the backbone alone (1024 "
                         "channels each), not block 12 beside it; the record's feature string loses _lf12")
    a = ap.parse_args(argv)
    if a.dump_qual and not a.dump_qual_dir:
        ap.error("--dump_qual needs --dump_qual_dir")
    want = {b.strip() for b in a.blocks.split(",") if b.strip()}
    if not want or not want <= set(BLOCKS_ALL.split(",")):
        ap.error(f"--blocks {a.blocks}: a comma list of coarse, fine, bank")
    a.blocks = ",".join(b for b in BLOCKS_ALL.split(",") if b in want)   # canonical order: the table compares it
    if a.zoom_fixed < 0:
        ap.error("--zoom_fixed must be 0 (decided) or a positive magnification")
    unknown = [d for d in a.datasets.split(",") if d not in ALL_DATASETS]
    if unknown:
        ap.error(f"unknown dataset(s) {unknown}; the panel is {ALL_DATASETS}")
    return a


def main(argv=None, hook=None, inputs=None):
    a = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)        # before torch is imported
    import torch
    from exemplar import FrozenInputs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"device: {device_name} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})", flush=True)
    if inputs is None:
        inputs = FrozenInputs(device, cache_dir=a.cache or None, bf16=True,
                              layers=(-1,) if a.last_layer_only else (-1, 12), fine="fine" in a.blocks.split(","))
    if hook is not None and hasattr(hook, "wrap_inputs"):
        inputs = hook.wrap_inputs(inputs)
    os.makedirs(a.score_dir, exist_ok=True)
    for ds in a.datasets.split(","):
        run_dataset(ds, a, inputs, device, device_name, hook=hook)


if __name__ == "__main__":
    main()
