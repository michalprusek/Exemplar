#!/usr/bin/env python
"""micro-SAM baseline on the AutoSeg panel — SELF-CONTAINED (does not import al_testbed).

Benchmarks **micro-SAM** (Archit et al., "Segment Anything for Microscopy", Nature Methods
2025) on our few-shot panel at the MATCHED protocol, so its per-dataset designated-metric
numbers sit directly beside ours.

MODE — be honest about what this measures
-----------------------------------------
We run micro-SAM's **Automatic Instance Segmentation (AIS)** with the **light-microscopy
finetuned** model (``vit_b_lm`` by default; ``vit_l_lm`` if you have it cached), which uses the
extra UNETR-style *decoder* (predicts foreground + center/boundary distances -> seeded
watershed). ``get_predictor_and_segmenter(..., segmentation_mode="ais")`` returns an
``InstanceSegmentationWithDecoder`` — the proper, fast micro-SAM microscopy path, NOT the slow
grid-prompt AMG fallback.

This is the realistic *"off-the-shelf microscopy tool"* usage. It is **PRETRAINED-GENERALIST,
NOT few-shot**: the 8 support masks are IGNORED (micro-SAM AIS has no train-on-support hook;
its generalist was already trained on ~17k LM images). We still consume the exact same
``load_dataset(spec, pool=20, test=24, seed=0)`` split and score the SAME test images with the SAME
metric so the comparison to our few-shot method is protocol-matched. Label it honestly in any
write-up: **micro-SAM AIS = off-the-shelf generalist, our method = 8-shot.**

REPRODUCTION CHECK (fairness — is our setup OPTIMAL, not handicapped?)
---------------------------------------------------------------------
``--mode reproduce`` runs DSB2018 (one of the datasets the paper reports) with ``vit_b_lm`` AIS
and prints our mSA next to the paper's reported LM-generalist AIS number (0.654). ``mSA`` (mean
segmentation accuracy = TP/(TP+FP+FN) averaged over IoU 0.5:0.05:0.95) is *exactly* our
``instance_ap`` metric. All of DSB/LIVECell/DeepBacs were in the LM-generalist training corpus,
so the paper's DSB 0.654 IS the in-distribution generalist number — the correct target for our
cached model. A match (we get ~0.64) confirms the integration is not handicapped.

Datasets: spheroid, spheroidj, dsb2018, rozpad, kvasir, hrf (keys in
``active_segmenter.eval.registry.PANEL``). dsb2018 -> instance_ap (per-instance masks); the
others -> fg-IoU / clDice on the binary foreground (= union of AIS instances). Per-dataset the
mean±std is over seeds. A dataset that fails to load/segment is SKIPped with a reason, never
crashes the run.

Run on tulen (env + weights already provisioned):
  export MAMBA_ROOT_PREFIX=/disk2/prusek/mm
  export PYTHONPATH=/disk1/prusek/active-segmenter
  # sanity that the setup is optimal (DSB mSA ~ paper 0.654):
  /disk2/prusek/microsam_env/bin/python scripts/microsam_bench.py --mode reproduce
  # quick smoke (seed 0, test idx 4, dsb2018 + one blob):
  /disk2/prusek/microsam_env/bin/python scripts/microsam_bench.py --mode smoke --datasets dsb2018,spheroid
  # full panel (6 seeds):
  /disk2/prusek/microsam_env/bin/python scripts/microsam_bench.py --mode panel --datasets all --seeds 6
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# --- make active_segmenter.eval.{registry,scoring} importable (numpy/skimage only) ----------
_ASG_ROOT = os.environ.get("ASG_ROOT", "/disk1/prusek/active-segmenter")
if _ASG_ROOT not in sys.path:
    sys.path.insert(0, _ASG_ROOT)

from active_segmenter.eval.registry import PANEL, load_dataset  # noqa: E402
from active_segmenter.eval.score_record import split_fingerprint, write_score_record  # noqa: E402
from active_segmenter.eval.scoring import effective_metric, primary_key, score_prediction  # noqa: E402

# The panel keys we benchmark. The hand-written prefix fixes the REPORT ORDER of the datasets that
# were here first; everything else in the registry is appended so a newly registered dataset is
# benchmarked by default instead of being silently absent from the baseline column. The previous
# fixed list of six had gone stale against a registry of fifteen, so `--datasets all` quietly
# skipped drive, monuseg, ctc_u373, isbi2012em, fisbe, bbbc010 and bacteria.
_REPORT_FIRST = ["spheroid", "spheroidj", "dsb2018", "rozpad", "kvasir", "hrf"]
PANEL_KEYS = ([k for k in _REPORT_FIRST if k in PANEL]
              + [k for k in PANEL if k not in _REPORT_FIRST])

# Archit et al. 2025 — reported LM-GENERALIST *AIS* mSA (mean segmentation accuracy) for the
# reproduction sanity check. mSA == our instance_ap. Only dsb2018 overlaps our panel.
PAPER_AIS_mSA = {
    "dsb2018": 0.654, "livecell": 0.415, "deepbacs": 0.497,
    "tissuenet": 0.329, "covid_if": 0.317, "dynamicnuclearnet": 0.592,
}


# ------------------------------------------------------------------------------------------
def prep_image(img) -> np.ndarray:
    """Return an image micro-SAM's SAM encoder accepts: uint8, 2-D grayscale or (H,W,3) RGB.

    Non-uint8 inputs (e.g. 16-bit microscopy) are min-max scaled per image so SAM's fixed
    pixel normalization does not clip them. RGBA -> RGB; singleton channel -> 2-D."""
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim == 3 and a.shape[-1] == 4:
        a = a[..., :3]
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        lo, hi = float(a.min()), float(a.max())
        a = np.zeros_like(a, np.uint8) if hi <= lo else ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return a


def build_segmenter(model_type: str, device: str, tile: int):
    """Load the micro-SAM predictor + AIS decoder segmenter (weights read from
    ~/.cache/micro_sam/models; no download when cached)."""
    from micro_sam.automatic_segmentation import get_predictor_and_segmenter

    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type, device=device, segmentation_mode="ais",
        is_tiled=bool(tile),
    )
    return predictor, segmenter


def run_ais(predictor, segmenter, img, tile: int) -> np.ndarray:
    """One image -> instance label map (2-D int)."""
    from micro_sam.automatic_segmentation import automatic_instance_segmentation

    kw = {}
    if tile:
        kw["tile_shape"] = (tile, tile)
        kw["halo"] = (tile // 4, tile // 4)
    seg = automatic_instance_segmentation(
        predictor=predictor, segmenter=segmenter, input_path=prep_image(img),
        ndim=2, verbose=False, **kw,
    )
    return np.asarray(seg)


def score_image(metric: str, seg: np.ndarray, gt) -> dict:
    """Score one AIS label map against GT with ``metric`` (the dataset's, or the --fg-scoring one).

    Foreground scoring needs no separate code path: the foreground of an instance label map is just
    ``seg > 0``, which is what this already hands ``score_prediction``, and ``score_prediction``
    returns ``fg_iou`` for every metric. So passing ``fg_iou`` here selects an already-computed
    number and additionally skips enumerating instances for an AP nobody asked for."""
    fg = seg > 0
    instances = None
    if metric == "instance_ap":
        instances = [seg == i for i in np.unique(seg) if i != 0]
    return score_prediction(metric, fg, gt, instances)


# ------------------------------------------------------------------------------------------
def eval_dataset(name, predictor, segmenter, seeds, pool, test, tile, max_test=None,
                 fg_scoring=False):
    """Return ``(mean, std, pk, per_seed_list, per_image_vals, split_fp)`` for one dataset.

    ``per_image_vals`` and ``split_fp`` exist so the caller can emit a sota_final score record:
    the paired Wilcoxon needs the individual image scores, which this function used to average
    away. Raises on load/segment failure so the caller can SKIP with a reason."""
    spec = PANEL[name]
    # ONE expression decides both the score and the `pk` the record is labelled with, so the two
    # cannot drift apart into a silent METRIC MISMATCH skip in stats().
    eff = effective_metric(spec.metric, fg_scoring)
    pk = primary_key(eff)
    # SAME fixed test split as scripts/sota_final.py, loaded ONCE at seed 0 with the same POOL size.
    # Previously this passed `support=8, seed=seed`, which on every download-kind dataset scored a
    # different set of images than our method: the flat-directory loader slices
    # `permutation(seed)[support : support+test]`, so both arguments move the test slice. The image
    # COUNT still matched, so nothing downstream could notice.
    _support_pool, test_pairs = load_dataset(spec, pool, test, seed=0)
    if max_test is not None:
        test_pairs = test_pairs[:max_test]
    if len(test_pairs) == 0:
        raise RuntimeError("no test images")
    # AFTER the max_test truncation, so the digest describes the images actually scored.
    split_fp = split_fingerprint(test_pairs)
    # micro-SAM ignores the support masks and is deterministic, and the split above no longer depends
    # on the seed — so every seed would re-segment the identical images for the identical score. Score
    # once and replicate, instead of burning N passes to reproduce one number.
    vals = []
    for img, gt in test_pairs:
        seg = run_ais(predictor, segmenter, img, tile)
        vals.append(score_image(eff, seg, gt)[pk])
        sys.stderr.write("."); sys.stderr.flush()
    score = float(np.mean(vals))
    per_seed = [score] * max(1, len(list(seeds)))
    sys.stderr.write(f" [{name} {pk}={score:.3f} (deterministic; replicated over {len(per_seed)} seeds)]\n")
    sys.stderr.flush()
    return score, 0.0, pk, per_seed, vals, split_fp


def write_record(args, name, pk, vals, seeds, split_fp):
    """Emit the sota_final score record for one dataset; return the path written.

    The method is ``microsam_<model>`` rather than a bare ``microsam`` because ``stats`` keys on
    (dataset, method) and REFUSES a duplicate: a ``vit_l_lm`` run into the same score dir as a
    ``vit_b_lm`` one would otherwise be a silent overwrite of a different model's numbers.
    """
    return write_score_record(
        args.score_dir, method=f"microsam_{args.model}", dataset=name, metric=pk,
        # Seed-major tiling of a DETERMINISTIC result: micro-SAM ignores the support draw and the
        # split is pinned at seed 0, so every seed really did score these same images identically —
        # which is exactly what the per-seed mean list already asserted, only now per image. The
        # resulting across-seed std of 0 is correct by construction, not a bug to explain away.
        per_seed_images=[list(vals)] * len(seeds), seeds=seeds, split_fp=split_fp,
        # fg_scoring is recorded even though `metric` usually implies it: on a clDice dataset the
        # convention keeps clDice either way, so the record would otherwise carry no trace of which
        # scoring convention produced it.
        protocol=dict(pool=args.pool, test=args.test, support=args.support, split_seed=0,
                      model=args.model, tile=args.tile, fg_scoring=bool(args.fg_scoring)),
        note="OFF-THE-SHELF generalist, support masks IGNORED (not few-shot). Deterministic: the "
             "split is scored ONCE and the per-image vector is replicated across seeds, so the "
             "across-seed std is 0 by construction.")


def resolve_datasets(arg: str):
    if arg in ("all", ""):
        return list(PANEL_KEYS)
    names = [n.strip() for n in arg.split(",") if n.strip()]
    bad = [n for n in names if n not in PANEL]
    if bad:
        raise SystemExit(f"unknown dataset(s): {bad}; valid = {list(PANEL)}")
    return names


# ------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="micro-SAM AIS baseline on the AutoSeg panel")
    ap.add_argument("--mode", choices=["panel", "smoke", "reproduce"], default="panel")
    ap.add_argument("--datasets", default="all", help="comma keys or 'all'")
    ap.add_argument("--model", default="vit_b_lm",
                    help="micro-SAM LM model (vit_b_lm cached; vit_l_lm if you fetch it)")
    ap.add_argument("--seeds", type=int, default=6, help="panel: seeds 0..N-1")
    ap.add_argument("--support", type=int, default=8,
                    help="recorded for provenance only; micro-SAM ignores the support masks")
    ap.add_argument("--pool", type=int, default=20,
                    help="support POOL size; must match sota_final's --pool so the test split (sliced "
                         "AFTER the pool by the flat-dir loader) is identical to our method's")
    ap.add_argument("--test", type=int, default=24)
    ap.add_argument("--device", default=("cuda" if _cuda() else "cpu"))
    ap.add_argument("--tile", type=int, default=0,
                    help="tile size for large images (0 = off; SAM resizes longest side to 1024)")
    ap.add_argument("--smoke-idx", type=int, default=4, help="smoke: single test image index")
    ap.add_argument("--fg-scoring", action="store_true",
                    help="score the FOREGROUND (fg_iou) on every non-clDice dataset instead of its "
                         "native metric, the campaign-wide convention that lets one table row name "
                         "one metric. Same semantics and flag name as specialist_finetune_bench.py; "
                         "the record's `metric` field follows, so the record still pairs. Ignored "
                         "by --mode reproduce, whose published target is an instance-AP number.")
    ap.add_argument("--score_dir", default="",
                    help="panel mode: write per-image score records here (results/scores) so this "
                         "baseline enters `sota_final.py stats`'s PAIRED significance instead of "
                         "being read off stdout. The method is named microsam_<model> so vit_b_lm / "
                         "vit_l_lm / histopathology runs cannot overwrite one another. Empty = "
                         "table only.")
    args = ap.parse_args()

    if args.score_dir and args.mode != "panel":
        # `reproduce` checks ONE dataset against a published number and `smoke` scores ONE image;
        # neither is a panel measurement, but a record from either is indistinguishable from one
        # once it is sitting in the score dir. Disable at the single gate the writes are behind.
        print(f"[microsam_bench] REFUSING to write score records in --mode {args.mode}: only the "
              f"panel run scores the full split that stats() pairs against (reproduce = fairness "
              f"check, smoke = a single image).", flush=True)
        args.score_dir = ""

    print(f"# micro-SAM AIS baseline | model={args.model} device={args.device} "
          f"mode={args.mode} | OFF-THE-SHELF generalist (support masks IGNORED, NOT few-shot)",
          flush=True)
    t0 = time.time()
    predictor, segmenter = build_segmenter(args.model, args.device, args.tile)
    print(f"# loaded {type(segmenter).__name__} in {time.time()-t0:.1f}s "
          f"({'decoder-AIS' if type(segmenter).__name__=='InstanceSegmentationWithDecoder' else 'AMG-FALLBACK!'})",
          flush=True)

    if args.mode == "reproduce":
        # DSB2018 (a dataset the paper reports) vs paper LM-generalist AIS mSA = 0.654.
        # fg_scoring is deliberately NOT forwarded: mSA IS instance-AP, so a foreground-scored
        # number compared against that band would print a confident PASS/LOW-CHECK verdict about a
        # quantity the paper never reported. Say so rather than let the flag quietly not apply.
        if args.fg_scoring:
            print("[microsam_bench] --fg-scoring does NOT apply to --mode reproduce: the published "
                  "target (mSA=0.654) is instance-AP, so this gate keeps the native metric.",
                  flush=True)
        seeds = list(range(max(1, args.seeds)))
        mean, std, pk, ps, _vals, _fp = eval_dataset("dsb2018", predictor, segmenter, seeds,
                                                     args.pool, args.test, args.tile)
        paper = PAPER_AIS_mSA["dsb2018"]
        print(f"\nREPRODUCTION  dsb2018  {pk}(=mSA) = {mean:.3f}±{std:.3f}  "
              f"vs paper {paper:.3f}  (Δ={mean-paper:+.3f}) over seeds {seeds}", flush=True)
        verdict = "MATCH — setup optimal" if abs(mean - paper) <= 0.10 else "MISMATCH — INVESTIGATE"
        print(f"REPRODUCTION VERDICT: {verdict}", flush=True)
        print("MICROSAM_BENCH_DONE", flush=True)
        return

    if args.mode == "smoke":
        seeds = [0]
        max_test = None
        names = resolve_datasets(args.datasets)
        idx = args.smoke_idx
        print(f"# SMOKE: seed 0, single test image idx={idx}, datasets={names}", flush=True)
        for name in names:
            try:
                spec = PANEL[name]
                eff = effective_metric(spec.metric, args.fg_scoring)
                pk = primary_key(eff)
                _s, test_pairs = load_dataset(spec, args.pool, args.test, seed=0)
                if idx >= len(test_pairs):
                    print(f"SKIP {name}: test idx {idx} >= n_test {len(test_pairs)}", flush=True)
                    continue
                img, gt = test_pairs[idx]
                seg = run_ais(predictor, segmenter, img, args.tile)
                sc = score_image(eff, seg, gt)
                print(f"{name:10s} {pk:11s} {sc[pk]:.3f}   (img {np.asarray(img).shape}, "
                      f"n_pred_inst={len(np.unique(seg))-1})", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"SKIP {name}: {type(e).__name__}: {e}", flush=True)
        print("MICROSAM_BENCH_DONE", flush=True)
        return

    # ---- panel ----
    seeds = list(range(max(1, args.seeds)))
    names = resolve_datasets(args.datasets)
    print(f"# PANEL: seeds={seeds} support={args.support} test={args.test} datasets={names} "
          f"fg_scoring={args.fg_scoring}", flush=True)
    rows = []
    for name in names:
        try:
            mean, std, pk, ps, vals, split_fp = eval_dataset(name, predictor, segmenter, seeds,
                                                             args.pool, args.test, args.tile,
                                                             fg_scoring=args.fg_scoring)
        except Exception as e:  # noqa: BLE001
            print(f"SKIP {name}: {type(e).__name__}: {e}", flush=True)
            continue
        note = PANEL[name].note
        rows.append((name, pk, mean, std, note))
        print(f"{name:10s} {pk:11s} {mean:.3f}±{std:.3f}   ({note})", flush=True)
        # Outside the SKIP handler on purpose: a dataset that fails to segment is a legitimate
        # skip, but a record this script builds wrongly is a bug, and folding it into the same
        # handler would print it as if micro-SAM had merely failed on that dataset.
        if args.score_dir:
            print(f"  wrote {write_record(args, name, pk, vals, seeds, split_fp)}", flush=True)

    print("\n===== micro-SAM AIS PANEL SUMMARY (off-the-shelf generalist; mean±std over seeds) =====",
          flush=True)
    for name, pk, mean, std, note in rows:
        extra = ""
        if name in PAPER_AIS_mSA and pk == "ap":
            extra = f"   [paper LM-gen AIS mSA={PAPER_AIS_mSA[name]:.3f}]"
        print(f"  {name:10s} {pk:11s} {mean:.3f}±{std:.3f}{extra}", flush=True)
    print("MICROSAM_BENCH_DONE", flush=True)


def _cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    main()
