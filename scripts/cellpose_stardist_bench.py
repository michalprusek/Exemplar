#!/usr/bin/env python3
"""Off-the-shelf PRETRAINED microscopy segmentation references on the AutoSeg panel.

Cellpose (``cyto3``, ``cpsam``) and StarDist (``2D_versatile_fluo`` / ``2D_versatile_he``)
are PRETRAINED GENERALISTS -- **NOT few-shot learners**. They ignore the 8-image support
set entirely and predict the 24 test images directly. They belong in the paper as an
*off-the-shelf reference*, and every number produced here is clearly labelled
``pretrained-not-few-shot`` so it is never confused with our few-shot method.

Why this file shells out per backend
-------------------------------------
Cellpose needs a torch env; StarDist needs a tensorflow env. They cannot coexist cleanly,
so each ``--backend`` run happens inside its own venv and dumps a small JSON of per-dataset
primary-metric numbers. ``--aggregate`` then merges those JSONs into the final table --
no model import needed for the merge.

Protocol (exactly as specified)
-------------------------------
* Datasets: keys in ``active_segmenter.eval.registry.PANEL`` --
  spheroid, spheroidj, dsb2018, rozpad, kvasir, hrf, microtubules (7 total).
* ``_, test = load_dataset(spec, pool=20, test=24, seed=0)`` -- the SAME fixed split
  ``scripts/sota_final.py`` gives our method, so the numbers sit beside ours honestly.
  Pretrained models ignore ``support``; we predict the ``test`` split. Passing ``seed=seed``
  here would silently move the test slice on download-kind datasets (the flat-dir loader
  slices ``permutation(seed)[support : support+test]``) and score different images than ours.
* NOTE on seed variance: the split is now pinned at ``seed=0`` for EVERY loader kind, and
  these models are deterministic and ignore the support, so the score is identical across
  seeds -> std=0 throughout. That is the honest report for a pretrained generalist: the
  seeds vary our method's support draw, and there is nothing for them to vary here. We
  memoize by a cheap split signature so the split is scored once, not six times.
* Score every image with ``active_segmenter.eval.scoring.score_prediction`` using the
  dataset's designated metric (``spec.metric``): dsb2018 -> ``instance_ap`` (per-instance
  masks are passed through, which Cellpose/StarDist produce natively); spheroid /
  spheroidj / rozpad / kvasir -> fg-IoU; hrf -> clDice. The union of predicted instances
  forms the foreground for the semantic metrics.

Reproduction gate (fairness)
----------------------------
Before trusting our-data numbers, each model must reproduce a published number so we know
it is set up OPTIMALLY. dsb2018 is in our panel and StarDist reports it directly. The
comparison metric is the DSB/Kaggle **mean AP over IoU thresholds [.5:.95]** (== our
primary ``instance_ap`` value, ``ap``); we additionally record the IoU=0.5 detection rate
(``ap50``, always higher). Published mean-AP ballparks (shown in the aggregate report):
  * StarDist 2D_versatile_fluo on DSB2018:  mean AP ~ 0.55-0.65  (Schmidt et al., MICCAI'18 / stardist repo)
  * Cellpose-SAM (cpsam) on DSB2018 nuclei:  mean AP ~ 0.55-0.75  (Pachitariu et al. 2025)
  * Cellpose3 cyto3 generalist on nuclei:    mean AP ~ 0.40-0.70  (Stringer/Pachitariu)
A number far below its band means a preprocessing/normalization bug -- fix before reporting.

Usage
-----
  # in cellpose_env:
  PYTHONPATH=/disk1/prusek/active-segmenter python cellpose_stardist_bench.py \
      --backend cellpose_cpsam --out results/csb_cpsam.json
  PYTHONPATH=... python cellpose_stardist_bench.py --backend cellpose_cyto3 --out results/csb_cyto3.json
  # in stardist_env (CPU is fine):
  CUDA_VISIBLE_DEVICES= PYTHONPATH=... python cellpose_stardist_bench.py \
      --backend stardist_fluo --out results/csb_stardist_fluo.json
  # merge + final table:
  python cellpose_stardist_bench.py --aggregate results/csb_cyto3.json results/csb_cpsam.json \
      results/csb_stardist_fluo.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

import numpy as np

PANEL_ORDER = ["spheroid", "spheroidj", "dsb2018", "rozpad", "kvasir", "hrf", "microtubules"]

# Preferred column order + display label for the aggregate table.
BACKEND_COLS = [
    ("cellpose_cyto3", "cellpose_cyto3"),
    ("cellpose_cpsam", "cellpose_cpsam"),
    ("stardist_fluo", "stardist"),      # the canonical "stardist" column (fluorescence/nuclei)
    ("stardist_he", "stardist_he"),     # extra column if run (H&E histology model)
]

# Published DSB2018 mean-AP@[.5:.95] bands, for the reproduction gate (compared vs the
# primary instance_ap value). ap50 is reported separately as supplementary.
REPRO_BANDS = {
    "stardist_fluo": (0.55, 0.65, "StarDist 2D_versatile_fluo, Schmidt et al. MICCAI'18 / stardist repo"),
    "stardist_he":   (0.00, 1.00, "H&E model on fluorescence is OUT of domain; no band"),
    "cellpose_cpsam": (0.55, 0.75, "Cellpose-SAM, Pachitariu et al. 2025 (nuclei)"),
    "cellpose_cyto3": (0.40, 0.70, "Cellpose3 cyto3 generalist, Stringer/Pachitariu (nuclei)"),
}


# --------------------------------------------------------------------------- #
# image helpers
# --------------------------------------------------------------------------- #
def to_gray(img: np.ndarray) -> np.ndarray:
    """Return a 2-D single-channel view (luminosity of RGB), dtype preserved where sane."""
    a = np.asarray(img)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=-1)
        a = a.astype(np.uint16 if np.asarray(img).dtype == np.uint16 else np.float32)
    return a


def to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """Return an HxWx3 uint8 image (for the H&E StarDist model / RGB-friendly cpsam)."""
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    elif a.ndim == 3 and a.shape[2] >= 3:
        a = a[..., :3]
    elif a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=-1)
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        lo, hi = float(a.min()), float(a.max())
        a = (255.0 * (a - lo) / (hi - lo)) if hi > lo else np.zeros_like(a)
        a = a.astype(np.uint8)
    return a


def cpsam_input(img: np.ndarray) -> np.ndarray:
    """cpsam is channel-agnostic: keep RGB when available (colour helps kvasir/hrf),
    else pass the 2-D grayscale image."""
    a = np.asarray(img)
    return a[..., :3] if a.ndim == 3 else a


def label_to_pred(labels: np.ndarray):
    """Model instance-label map -> (fg_bool, list[per-instance bool mask])."""
    labels = np.asarray(labels)
    fg = labels > 0
    ids = [int(i) for i in np.unique(labels) if i != 0]
    instances = [labels == i for i in ids]
    return fg, instances


def _n_tiles(shape, block: int = 1024):
    return tuple(max(1, int(np.ceil(s / block))) for s in shape)


def split_signature(pairs) -> str:
    """Cheap content hash of a test split, so identical splits (fewshot/dsb loaders take
    the same first-N regardless of seed) are scored once instead of once per seed."""
    import hashlib

    h = hashlib.md5()
    for img, gt in pairs:
        a = np.asarray(img)
        g = np.asarray(gt)
        h.update(str(a.shape).encode())
        h.update(str(int(a.astype(np.int64).sum())).encode())
        h.update(str(int((g > 0).sum())).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# backends (imports are lazy so the module loads in either env)
# --------------------------------------------------------------------------- #
def load_backend(name: str):
    """Return ``(kind, variant, model, predict_fn)`` or raise on unavailability."""
    if name in ("cellpose_cyto3", "cellpose_cpsam"):
        import cellpose
        from cellpose import models as cpmodels

        ver = str(getattr(cellpose, "version", getattr(cellpose, "__version__", "0")))
        major = int(ver.split(".")[0]) if ver[:1].isdigit() else 0

        if name == "cellpose_cpsam":
            # cpsam is the Cellpose-SAM (v4) transformer generalist; only exists in v4+.
            if major < 4:
                raise RuntimeError(f"cpsam requires cellpose>=4 (found {ver}); run in cellpose_env")
            model = cpmodels.CellposeModel(gpu=True)  # cpsam is the v4 default
            variant = "cpsam"

            def predict(img):
                res = model.eval(cpsam_input(img), diameter=None)
                return np.asarray(res[0])

        else:  # cellpose_cyto3
            # CRITICAL: cellpose v4 SILENTLY ignores pretrained_model='cyto3' and loads
            # cpsam (verified: identical CPSAM net / cpsam_v2 weights). A cyto3 column
            # from v4 would be a fake duplicate of cpsam -- a handicapped/invalid baseline.
            # The genuine cyto3 (CPnet CNN) lives only in cellpose 3.x, run in cellpose3_env.
            if major >= 4:
                raise RuntimeError(
                    f"cyto3 requires cellpose 3.x (found {ver}); cellpose>=4 silently "
                    "substitutes cpsam. Run this backend in /disk2/prusek/cellpose3_env")
            # v3 Cellpose bundles a SizeModel -> diameter=None auto-estimates (optimal
            # off-the-shelf use, no manual tuning). channels=[0,0] = grayscale cytoplasm.
            model = cpmodels.Cellpose(gpu=True, model_type="cyto3")
            variant = "cyto3"

            def predict(img):
                res = model.eval(to_gray(img), diameter=None, channels=[0, 0])
                return np.asarray(res[0])

        return "cellpose", variant, model, predict

    if name in ("stardist_fluo", "stardist_he"):
        from csbdeep.utils import normalize
        from stardist.models import StarDist2D

        key = "2D_versatile_fluo" if name == "stardist_fluo" else "2D_versatile_he"
        model = StarDist2D.from_pretrained(key)
        he = name == "stardist_he"

        def predict(img):
            if he:
                x = to_rgb_uint8(img).astype(np.float32)
                x = normalize(x, 1, 99.8, axis=(0, 1))
                nt = _n_tiles(x.shape[:2]) + (1,)
                labels, _ = model.predict_instances(x, n_tiles=nt)
            else:
                g = to_gray(img).astype(np.float32)
                x = normalize(g, 1, 99.8)
                labels, _ = model.predict_instances(x, n_tiles=_n_tiles(g.shape))
            return np.asarray(labels)

        return "stardist", ("he" if he else "fluo"), model, predict

    raise ValueError(f"unknown backend: {name}")


# --------------------------------------------------------------------------- #
# run one backend over the panel
# --------------------------------------------------------------------------- #
def _score_split(eff, pk, predict, test_pairs, dname):
    """Score one test split with metric ``eff``; return per-split means + timing.

    ``per_image`` carries the individual primary-metric scores alongside the aggregates, because
    a paired Wilcoxon needs them and averaging them away here is what left these baselines with
    no significance test. The aggregate keys are untouched — ``--out``'s schema depends on them.

    ``eff`` is the EFFECTIVE metric (the dataset's, or ``fg_iou`` under --fg-scoring), not the
    DatasetSpec tag, so the foreground path needs no separate code: the foreground of the model's
    instance label map is already what ``label_to_pred`` hands over, and ``score_prediction``
    returns ``fg_iou`` for every metric — passing ``fg_iou`` selects that number and skips an
    instance-AP that a foreground run does not report.
    """
    from active_segmenter.eval import metrics
    from active_segmenter.eval.scoring import score_prediction

    prim, fgi, bf, ap50 = [], [], [], []
    t0 = time.time()
    for i, (img, gt) in enumerate(test_pairs):
        try:
            labels = predict(img)
            if labels.shape != np.asarray(gt).shape[:2]:
                labels = labels[: gt.shape[0], : gt.shape[1]]
        except Exception as e:      # scoring a crash as an empty mask = 0.0 DEFLATES this baseline
            raise RuntimeError(f"{dname}[{i}] predict failed; refusing to score it as an empty "
                               f"mask (that would silently understate this baseline)") from e
        fg, inst = label_to_pred(labels)
        sc = score_prediction(eff, fg, gt, inst if eff == "instance_ap" else None)
        prim.append(float(sc[pk]))
        fgi.append(float(sc.get("fg_iou", 0.0)))
        bf.append(float(sc.get("bf", 0.0)))
        if eff == "instance_ap":
            d = metrics.instance_ap(inst, gt_labels=gt) if inst else {"ap50": 0.0}
            ap50.append(float(d["ap50"]))
    return {"primary": float(np.mean(prim)), "per_image": prim, "fg_iou": float(np.mean(fgi)),
            "bf": float(np.mean(bf)),
            "ap50": float(np.mean(ap50)) if ap50 else None,
            "n": len(prim), "sec_per_img": (time.time() - t0) / max(1, len(prim))}


def write_record(score_dir, backend, dname, pk, per_seed, seeds, split_fp, protocol):
    """Emit the sota_final score record for one (backend, dataset); return the path written.

    This is ADDITIONAL to the ``--out`` aggregate, which keeps its own one-file-per-backend schema
    that other tooling reads. The two differ in what they are for: ``--out`` reports the numbers,
    this makes them testable.
    """
    from active_segmenter.eval.score_record import write_score_record

    return write_score_record(
        score_dir, method=backend, dataset=dname,
        # `pk`, NOT the row's `metric`: the row records the DatasetSpec tag (`instance_ap`) while
        # a score record must carry the primary key (`ap`) our method's records use. Disagreeing
        # here does not raise — stats() just skips every comparison, silently.
        metric=pk,
        # Seed-major tiling of a DETERMINISTIC result. These pretrained models ignore the support
        # draw and the split is pinned at seed 0, so every seed really did score these same images
        # identically — which is why `per_seed` already holds the memoized entry once per seed.
        # std=0 across seeds is therefore correct rather than a bug.
        per_seed_images=[d["per_image"] for d in per_seed], seeds=seeds, split_fp=split_fp,
        protocol=protocol,
        note="OFF-THE-SHELF pretrained generalist; support ignored (NOT few-shot). Deterministic: "
             "the split is scored ONCE and the per-image vector is replicated across seeds, so "
             "std=0 by construction.")


def run_backend(name, datasets, support, pool, test, seeds, limit_images, out_path, score_dir="",
                fg_scoring=False):
    from active_segmenter.eval.registry import PANEL, load_dataset
    from active_segmenter.eval.score_record import split_fingerprint
    from active_segmenter.eval.scoring import effective_metric, primary_key

    print(f"[bench] backend={name}  seeds={seeds}  fg_scoring={fg_scoring}  "
          f"(PRETRAINED, not few-shot)", flush=True)
    try:
        kind, variant, _model, predict = load_backend(name)
    except Exception as e:  # model unavailable -> whole backend skipped, never crash
        traceback.print_exc()
        result = {"backend": name, "error": f"backend load failed: {e}",
                  "pretrained_not_fewshot": True, "results": {}, "skipped": {}}
        _dump(out_path, result)
        print(f"[bench] SKIP backend {name}: {e}", flush=True)
        return result

    results, skipped = {}, {}
    for dname in datasets:
        if dname not in PANEL:
            skipped[dname] = "not in PANEL"
            continue
        spec = PANEL[dname]
        # ONE expression decides the score, the `--out` row's label and the record's `metric`
        # field, so they cannot drift apart into a silent METRIC MISMATCH skip in stats().
        eff = effective_metric(spec.metric, fg_scoring)
        pk = primary_key(eff)
        per_seed, sig_cache = [], {}
        try:
            # SAME fixed test split as scripts/sota_final.py: loaded ONCE at seed 0 with the same POOL
            # size. Passing `support=8, seed=seed` (the previous behaviour) scored a DIFFERENT set of
            # images than our method on every download-kind dataset, because the flat-directory loader
            # slices `permutation(seed)[support : support+test]` — both arguments move the test slice,
            # while the image COUNT stays the same, so nothing downstream could detect it.
            _, test_pairs = load_dataset(spec, pool, test, seed=0)
            if limit_images:
                test_pairs = test_pairs[:limit_images]
            sig = split_signature(test_pairs)
            for seed in seeds:
                if sig not in sig_cache:  # deterministic pretrained model + seed-invariant split
                    sig_cache[sig] = _score_split(eff, pk, predict, test_pairs, dname)
                per_seed.append(sig_cache[sig])
        except Exception as e:
            skipped[dname] = f"load/score failed: {e}"
            print(f"[bench]   SKIP {dname}: {e}", flush=True)
            continue

        prims = [d["primary"] for d in per_seed]
        ap50s = [d["ap50"] for d in per_seed if d["ap50"] is not None]
        # `eff`, not spec.metric: under --fg-scoring the row's numbers ARE foreground IoU, and a row
        # that labels them `instance_ap` would be read as an AP by every consumer of --out.
        row = {"metric": eff, "primary_key": pk,
               "primary_mean": float(np.mean(prims)), "primary_std": float(np.std(prims)),
               "per_seed_primary": prims,
               "ap50_mean": float(np.mean(ap50s)) if ap50s else None,
               "ap50_std": float(np.std(ap50s)) if ap50s else None,
               "fg_iou_mean": float(np.mean([d["fg_iou"] for d in per_seed])),
               "n": per_seed[0]["n"], "n_seeds": len(seeds),
               "n_unique_splits": len(sig_cache),
               "sec_per_img": per_seed[0]["sec_per_img"]}
        results[dname] = row
        # Deliberately outside the try above: a dataset that fails to load or predict is a
        # legitimate SKIP, but a record this script builds wrongly is a bug, and routing it into
        # the same handler would file it under "load/score failed" and leave the paper a column
        # with no test behind it — the very defect these records exist to close.
        if score_dir:
            # NOT `split_signature` — that is the cheap md5 over shapes/sums used only to memoize
            # repeated seeds. Pairing needs sota_final's byte-exact sha256 over the pixel data, or
            # stats() refuses the comparison as DIFFERENT TEST IMAGES. After --limit-images.
            path = write_record(score_dir, name, dname, pk, per_seed, seeds,
                                split_fingerprint(test_pairs),
                                # fg_scoring is recorded even though `metric` usually implies it:
                                # on a clDice dataset the convention keeps clDice either way, so
                                # the record would otherwise carry no trace of which scoring
                                # convention produced it.
                                dict(pool=pool, test=test, support=support, split_seed=0,
                                     backend=name, fg_scoring=bool(fg_scoring)))
            print(f"[bench]   wrote {path}", flush=True)
        extra = (f"  AP@0.5={row['ap50_mean']:.4f}+/-{row['ap50_std']:.4f}"
                 if row["ap50_mean"] is not None else "")
        print(f"[bench]   {dname:12s} {pk:8s}={row['primary_mean']:.4f}+/-{row['primary_std']:.4f}"
              f"  n={row['n']} splits={row['n_unique_splits']}/{len(seeds)}"
              f"  ({row['sec_per_img']:.2f}s/img){extra}", flush=True)

    result = {"backend": name, "kind": kind, "variant": variant,
              "pretrained_not_fewshot": True, "support": support, "test": test,
              "seeds": list(seeds), "fg_scoring": bool(fg_scoring),
              "note": "OFF-THE-SHELF pretrained generalist; support ignored; mean+/-std over seeds",
              "results": results, "skipped": skipped}
    _dump(out_path, result)
    print(f"[bench] wrote {out_path}", flush=True)
    return result


def _dump(path, obj):
    if not path:
        return
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# --------------------------------------------------------------------------- #
# aggregate -> final table
# --------------------------------------------------------------------------- #
def aggregate(paths):
    from active_segmenter.eval.scoring import primary_key
    from active_segmenter.eval.registry import PANEL

    loaded = {}
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        loaded[d["backend"]] = d

    cols = [(b, lbl) for b, lbl in BACKEND_COLS if b in loaded]
    # any backend not in the preferred list still gets a column
    for b in loaded:
        if b not in [c[0] for c in cols]:
            cols.append((b, b))

    dsets = [d for d in PANEL_ORDER if any(d in loaded[b].get("results", {}) for b, _ in cols)]
    seeds = next((loaded[b].get("seeds") for b, _ in cols if loaded[b].get("seeds")), None)

    print("=" * 92)
    print("OFF-THE-SHELF PRETRAINED REFERENCES  (NOT few-shot; support set ignored)")
    print("Cellpose (cyto3, cpsam) + StarDist (2D_versatile_fluo).  "
          f"support=8, test=24, seeds={seeds}.")
    print("Cells = mean +/- std over seeds. fewshot/dsb splits are seed-invariant (std=0);")
    print("only kvasir/hrf (download loaders) resample per seed.")
    print("=" * 92)

    labels = [lbl for _, lbl in cols]
    header = f"{'dataset':13s} {'metric':10s} " + " ".join(f"{l:>17s}" for l in labels)
    print(header)
    print("-" * len(header))
    for dname in dsets:
        spec = PANEL[dname]
        cells, seen_pk = [], {}
        for b, lbl in cols:
            r = loaded[b].get("results", {}).get(dname)
            if loaded[b].get("error"):
                cells.append("SKIP(model)")
            elif r is None or r.get("primary_mean") is None:
                cells.append("SKIP")
            else:
                # The ROW's own primary_key, not primary_key(spec.metric): an fg-scored backend
                # reports fg_iou for an instance_ap dataset, and re-deriving the label from the
                # registry would head the column `ap` over foreground numbers.
                seen_pk[lbl] = r.get("primary_key", primary_key(spec.metric))
                cells.append(f"{r['primary_mean']:.3f}+/-{r['primary_std']:.3f}")
        pk = next(iter(seen_pk.values()), primary_key(spec.metric))
        print(f"{dname:13s} {pk:10s} " + " ".join(f"{c:>17s}" for c in cells))
        if len(set(seen_pk.values())) > 1:
            # One header cannot name two metrics; mixing an fg-scored file with a natively scored
            # one puts incomparable numbers side by side under a single label.
            print(f"{'':13s} !! MIXED METRICS in this row -- NOT comparable: {seen_pk}")

    # ---- reproduction gate (DSB2018 mean AP@[.5:.95] vs published band) ----
    print()
    print("REPRODUCTION GATE -- DSB2018 mean AP@[.5:.95] vs each model's published band")
    print("(ap50 = IoU=0.5 detection rate, supplementary):")
    for b, lbl in cols:
        r = loaded[b].get("results", {}).get("dsb2018")
        band = REPRO_BANDS.get(b)
        if loaded[b].get("error"):
            print(f"  {lbl:15s} SKIP (model unavailable)")
            continue
        if not r or r.get("primary_mean") is None:
            print(f"  {lbl:15s} n/a (dsb2018 not run)")
            continue
        if r.get("primary_key") not in (None, "ap"):
            # The published bands are mean AP@[.5:.95]. Under --fg-scoring `primary_mean` is
            # foreground IoU, and a PASS/LOW-CHECK verdict on it would be a fairness claim about a
            # quantity no paper reported. Re-run without --fg-scoring to exercise this gate.
            print(f"  {lbl:15s} n/a (primary is {r['primary_key']}, the bands are mean AP -- "
                  f"re-run without --fg-scoring)")
            continue
        meanap = r["primary_mean"]
        ap50 = r.get("ap50_mean")
        ap50s = f"  ap50={ap50:.4f}" if ap50 is not None else ""
        if band:
            lo, hi, ref = band
            ok = "PASS" if lo <= meanap <= hi else ("HIGH-OK" if meanap > hi else "LOW-CHECK")
            print(f"  {lbl:15s} meanAP={meanap:.4f}  band=[{lo:.2f},{hi:.2f}]  {ok}{ap50s}")
            print(f"  {'':15s}   ref: {ref}")
        else:
            print(f"  {lbl:15s} meanAP={meanap:.4f}{ap50s}")

    # ---- skips ----
    anyskip = False
    for b, lbl in cols:
        sk = loaded[b].get("skipped", {})
        err = loaded[b].get("error")
        if err:
            anyskip = True
            print(f"[skip] {lbl}: {err}")
        for dn, why in sk.items():
            anyskip = True
            print(f"[skip] {lbl}/{dn}: {why}")
    if not anyskip:
        print("[skip] none")

    print("CELLPOSE_STARDIST_BENCH_DONE")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend",
                    choices=["cellpose_cyto3", "cellpose_cpsam", "stardist_fluo", "stardist_he"])
    ap.add_argument("--datasets", nargs="+", default=PANEL_ORDER)
    ap.add_argument("--support", type=int, default=8,
                    help="recorded for provenance only; these pretrained models ignore the support")
    ap.add_argument("--pool", type=int, default=20,
                    help="support POOL size; must match sota_final's --pool so the test split (sliced "
                         "AFTER the pool by the flat-dir loader) is identical to our method's")
    ap.add_argument("--test", type=int, default=24)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--limit-images", type=int, default=None,
                    help="smoke: only first N test images per dataset")
    ap.add_argument("--out", default=None, help="write per-backend JSON here")
    ap.add_argument("--score_dir", default="",
                    help="ADDITIONALLY write one per-image score record per dataset here "
                         "(results/scores), so this backend enters `sota_final.py stats`'s PAIRED "
                         "significance. Separate from --out, whose one-file-per-backend aggregate "
                         "schema other tooling reads and which is unchanged. Empty = no records.")
    ap.add_argument("--fg-scoring", action="store_true",
                    help="score the FOREGROUND (fg_iou) on every non-clDice dataset instead of its "
                         "native metric, the campaign-wide convention that lets one table row name "
                         "one metric. Same semantics and flag name as specialist_finetune_bench.py; "
                         "the --out row's `metric` and the score record's `metric` field both "
                         "follow, so the record still pairs.")
    ap.add_argument("--aggregate", nargs="+", default=None,
                    help="merge these per-backend JSONs into the final table")
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args.aggregate)
        return
    if not args.backend:
        ap.error("either --backend or --aggregate is required")
    if args.score_dir and args.limit_images:
        # --limit-images truncates the split, so its records would carry a different split_fp and
        # test_per_seed than the full-panel ones and could only ever be SKIPPED — after sitting in
        # the score dir looking real. Refuse at the single gate the writes are behind.
        print("[bench] REFUSING to write score records with --limit-images: a truncated split must "
              "never land beside full-split records. Re-run without it for pairable records.",
              flush=True)
        args.score_dir = ""
    run_backend(args.backend, args.datasets, args.support, args.pool, args.test,
                args.seeds, args.limit_images, args.out, args.score_dir, args.fg_scoring)


if __name__ == "__main__":
    sys.exit(main())
