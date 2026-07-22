#!/usr/bin/env python
"""Broad-panel benchmark — always test acquisition/segmenter methods across a WIDE range of
datasets, not one. Reuses the al_testbed machinery (Ctx / run_curve / SCORERS) per dataset and
prints a dataset x method table of final fg-IoU and AUC-under-budget (label efficiency). A
dataset that fails to load or download is SKIPPED with a logged reason, never crashing the run.

Run on tulen:
  ~/dinov3_env/bin/python scripts/panel_benchmark.py --datasets all \
    --methods random,typiclust,smec,geoloop --support 20 --test 24 --rounds 6 --seeds 3 \
    --cache /disk1/prusek/asg_cache_panel
"""
import argparse

import numpy as np

from active_segmenter.config import EncoderConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval.registry import PANEL, load_dataset
from active_segmenter.acquire import coldstart


def _auc(curve):
    """Mean fg-IoU over the budget = area under the budget-IoU curve (label efficiency)."""
    return float(np.mean([v for _, v in curve]))


def run_quality(spec, args, dev):
    """Segmenter-quality snapshot (no AL loop): fit each backend on the full support set and
    score test with the metric DESIGNATED for this dataset (spec.metric): instance-AP for
    per-instance GT, clDice for tubular structures, fg-IoU for semantic blobs. Used to rank
    SEGMENTERS across the panel, independent of acquisition."""
    from active_segmenter.eval.registry import load_dataset
    from active_segmenter.eval.scoring import primary_key, score_prediction
    from active_segmenter.segment.base import LabeledExample
    from scripts.al_testbed import make_backend

    seeds = max(1, args.seeds)
    # MULTI-SEED: draw a support POOL and subsample a different support set per seed. Use a FIXED
    # --pool (independent of K) so a K-curve is CLEAN: same support pool AND same test for every K.
    # (The old pool=2·K shifted both the support composition AND — for load_flat_fewshot datasets
    # like hrf/kvasir, where test = chosen[support:support+test] — the TEST SET itself with K, making
    # K-curves non-comparable.) --pool 0 keeps the legacy 2·K for back-compat.
    pool_n = args.pool if args.pool > 0 else (args.support * 2 if seeds > 1 else args.support)
    support_pool, test = load_dataset(spec, pool_n, args.test, seed=0)
    if len(support_pool) < args.support or len(test) < 2:
        raise RuntimeError(f"too few images (pool={len(support_pool)} test={len(test)})")
    cfg = RunConfig(device="auto", cache_dir=args.cache,
                    encoder=EncoderConfig(model_id=args.model, resolution=args.res,
                                          backbone=args.backbone, convnext_stage=args.stage,
                                          tile=args.tile, layer=args.layer,
                                          gram_refine=args.gram,
                                          superres_factor=args.superres, jbu=args.jbu))
    enc = CachedEncoder(cfg, dev, args.cache)
    pool = [LabeledExample(im, enc.extract(im), np.asarray(l)) for im, l in support_pool]
    tst = [LabeledExample(im, enc.extract(im), np.asarray(l)) for im, l in test]
    pk = primary_key(spec.metric)

    from active_segmenter.segment.base import reset_backend_for_new_support as _reset_head

    out = {}
    for be_name in args.qbackends.split(","):
        be = make_backend(be_name, cfg, dev, refine=("none" if args.crop else args.refine), enc=enc)
        if args.crop:                                  # native-res crop pipeline (fg per crop -> stitch)
            from active_segmenter.segment.crop_segmenter import CropSegmenter
            be = CropSegmenter(be, enc, crop=args.crop, overlap=args.crop_overlap,
                               max_crops_fit=args.crop_maxfit)
        prims, bfs = [], []
        for s in range(seeds):
            rng = np.random.default_rng(s)
            idx = (list(rng.choice(len(pool), args.support, replace=False)) if seeds > 1
                   else list(range(min(args.support, len(pool)))))
            _reset_head(be)                            # forget the previous draw's self-configuration;
            #                                            _fcache (encoder-only) survives, _ccache cannot
            be.fit([pool[i] for i in idx])
            rows = []
            for ex in tst:
                fg = be.foreground(ex.image, ex.feat_grid)
                instances = None
                if spec.metric == "instance_ap":
                    # NO try/except: a crash here used to be scored as AP 0.0, which silently deflates
                    # whichever backend crashed. A legitimate "no instances" is an empty list from the
                    # normal path, so the only thing catching would hide is a real failure.
                    instances = [m.mask for m in be.predict(ex.image, ex.feat_grid)]
                rows.append(score_prediction(spec.metric, fg, ex.label_map, instances))
            prims.append(float(np.mean([r[pk] for r in rows])))
            bfs.append(float(np.mean([r["bf"] for r in rows])))
        out[be_name] = (float(np.mean(prims)), float(np.mean(bfs)), float(np.std(prims)))
    return out, args.support, len(tst), pk


def run_one(spec, args, dev):
    from scripts.al_testbed import Ctx, SCORERS, make_backend, run_curve

    support, test = load_dataset(spec, args.support, args.test, seed=0)
    if len(support) < 4 or len(test) < 2:
        raise RuntimeError(f"too few images (support={len(support)} test={len(test)})")
    cfg = RunConfig(device="auto", cache_dir=args.cache,
                    encoder=EncoderConfig(model_id=args.model, resolution=args.res,
                                          backbone=args.backbone, convnext_stage=args.stage,
                                          tile=args.tile, layer=args.layer,
                                          gram_refine=args.gram,
                                          superres_factor=args.superres, jbu=args.jbu))
    enc = CachedEncoder(cfg, dev, args.cache)
    trf = [enc.extract(im) for im, _ in support]
    tef = [enc.extract(im) for im, _ in test]
    cls = np.stack([enc.extract_cls(im) for im, _ in support])
    ctx = Ctx(trf, support, tef, test, cls, dev, metric=spec.metric)
    if args.segmenter:
        ctx.set_backend(make_backend(args.segmenter, cfg, dev, refine=args.refine))
    pool0 = list(range(len(support)))
    methods = [m for m in args.methods.split(",") if m in SCORERS]
    out = {}
    for m in methods:
        finals, aucs = [], []
        for seed in range(args.seeds):
            cold = coldstart.typiclust(ctx.cls, 3, seed=seed)
            curve = run_curve(SCORERS[m], cold, pool0, ctx, args.rounds,
                              np.random.default_rng(seed), batch=args.batch, method=m)
            finals.append(curve[-1][1])
            aucs.append(_auc(curve))
        out[m] = (float(np.mean(finals)), float(np.std(finals)), float(np.mean(aucs)))
    return out, len(support), len(test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all", help="'all' or comma list of panel names")
    ap.add_argument("--methods", default="random,typiclust,smec,geoloop")
    ap.add_argument("--segmenter", default=None,
                    choices=[None, "correspondence", "head", "insid3", "sam3",
                             "head_fusion", "head_fusion_al", "head_fusion_v2"])
    ap.add_argument("--support", type=int, default=20)
    ap.add_argument("--pool", type=int, default=0,
                    help="FIXED support-pool size to subsample K from (0=legacy 2·support); set a "
                         "constant across K for a clean, comparable K-curve (fixes the test-shift bug)")
    ap.add_argument("--test", type=int, default=24)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--backbone", default="auto", choices=["auto", "vit", "convnext"])
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--tile", action="store_true",
                    help="tile a DINOv3-ViT at native scale (feather-blended) -> native-res "
                         "DINOv3 dense features without the single-pass quadratic-attention OOM")
    ap.add_argument("--layer", type=int, default=-1,
                    help="DINOv3 ViT block to read (-1=last; intermediate helps fine structures)")
    ap.add_argument("--gram", action="store_true", help="Gram self-similarity feature refinement")
    ap.add_argument("--superres", type=int, default=1,
                    help="sub-patch shift-merge densification factor (1=off, 2 or 4)")
    ap.add_argument("--jbu", action="store_true",
                    help="edge-guided JBU feature snapping (parameter-free feature-level CRF)")
    ap.add_argument("--refine", default="none",
                    choices=["none", "point", "mask", "mask_box", "amodal"],
                    help="SAM refine of instance proposals (mask=shape-prompt, amodal=keep overlaps)")
    ap.add_argument("--cache", default="/tmp/asg_cache_panel")
    ap.add_argument("--crop", type=int, default=0,
                    help="native-res CROP pipeline: tile into crops of this size (0=off, e.g. 672)")
    ap.add_argument("--crop_overlap", type=float, default=0.25, help="fractional crop overlap")
    ap.add_argument("--crop_maxfit", type=int, default=12, help="max labeled crops per support image")
    ap.add_argument("--quality", action="store_true",
                    help="rank SEGMENTERS (fit on full support -> fg-IoU + boundary-F per dataset), "
                         "no AL loop; use for native-res ConvNeXt (--res 0)")
    ap.add_argument("--qbackends", default="correspondence,head")
    args = ap.parse_args()

    names = list(PANEL) if args.datasets == "all" else [n for n in args.datasets.split(",") if n in PANEL]
    dev = RunConfig(device="auto").device_resolved()
    methods = [m for m in args.methods.split(",")]
    res_label = "native" if args.res <= 0 else args.res
    print(f"device={dev} datasets={names} "
          f"{'quality qbackends='+args.qbackends if args.quality else 'methods='+str(methods)} "
          f"support={args.support} test={args.test} rounds={args.rounds} seeds={args.seeds} "
          f"segmenter={args.segmenter} backbone={args.backbone} res={res_label} stage={args.stage}",
          flush=True)

    if args.quality:
        qresults, qmetric = {}, {}
        for name in names:
            try:
                res, ns, nt, pk = run_quality(PANEL[name], args, dev)
                qresults[name] = res
                qmetric[name] = pk
                cells = "  ".join(f"{b}: {pk} {v[0]:.3f}±{v[2]:.3f} bF {v[1]:.3f}" for b, v in res.items())
                print(f"\n[{name}] ({PANEL[name].note}; metric={pk}; support={ns} test={nt})"
                      f"\n  {cells}", flush=True)
            except Exception as e:
                print(f"\n[{name}] SKIP — {repr(e)[:120]}", flush=True)
        if qresults:
            qbs = args.qbackends.split(",")
            print("\n===== SEGMENTER-QUALITY SUMMARY (each dataset's DESIGNATED metric) =====",
                  flush=True)
            print(f"{'dataset':>12} {'metric':>12} " + " ".join(f"{b:>14}" for b in qbs))
            for name, res in qresults.items():
                print(f"{name:>12} {qmetric[name]:>12} " + " ".join(
                    f"{res[b][0]:>8.3f}±{res[b][2]:<5.3f}" if b in res else f"{'-':>14}" for b in qbs))
        print("PANEL_DONE")
        return

    results = {}
    for name in names:
        spec = PANEL[name]
        try:
            res, ns, nt = run_one(spec, args, dev)
            results[name] = res
            mtag = spec.metric if args.segmenter else "iou(frozen)"
            print(f"\n[{name}] ({spec.note}; metric={mtag}; support={ns} test={nt})", flush=True)
            print(f"  {'method':>12} {'final':>16} {'AUC':>8}")
            for m, (mu, sd, auc) in res.items():
                print(f"  {m:>12} {mu:>10.3f} +/-{sd:>5.3f} {auc:>8.3f}", flush=True)
        except Exception as e:
            print(f"\n[{name}] SKIP — {repr(e)[:120]}", flush=True)

    # summary table: dataset x method, each dataset's DESIGNATED metric (with --segmenter)
    if results:
        mnote = "designated metric" if args.segmenter else "frozen fg-IoU"
        print(f"\n===== PANEL SUMMARY (final, {mnote}, mean over seeds) =====", flush=True)
        hdr = f"{'dataset':>12} " + " ".join(f"{m:>10}" for m in methods)
        print(hdr)
        for name, res in results.items():
            row = f"{name:>12} " + " ".join(
                f"{res[m][0]:>10.3f}" if m in res else f"{'-':>10}" for m in methods)
            print(row)
        # win counts vs random
        if "random" in methods:
            print("\n# beats-random count (final IoU) per method, across datasets:")
            for m in methods:
                if m == "random":
                    continue
                wins = sum(1 for r in results.values()
                           if m in r and "random" in r and r[m][0] > r["random"][0])
                print(f"  {m:>12}: {wins}/{len(results)}")
    print("PANEL_DONE")


if __name__ == "__main__":
    main()
