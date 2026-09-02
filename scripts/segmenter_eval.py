#!/usr/bin/env python
"""Fixed-budget segmenter-quality table. For a given labeled support set, evaluate each
backend on the held-out test set with fg-IoU (continuity with prior findings), instance
AP@[.5:.95], and boundary-F (small objects). Complements ``al_testbed.py --segmenter``,
which gives the fg-IoU curve vs #labels; this gives the multi-metric snapshot the findings
table reports.

Run on tulen (detached):
  ~/dinov3_env/bin/python scripts/segmenter_eval.py --dataset fewshot \
    --data <DECAY_ROOT> --cache /disk1/prusek/asg_cache_rozpad \
    --pool 10 --test 24 --res 1024 --backends correspondence,head,insid3
"""
import argparse

import numpy as np

from active_segmenter.config import EncoderConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval import metrics
from active_segmenter.eval.datasets import load_dsb2018, load_fewshot
from active_segmenter.segment.base import LabeledExample


def evaluate_backend(be, support_examples, test_examples) -> dict:
    """Fit the backend on the support set, then mean fg-IoU / instance-AP / boundary-F
    over the test set. AP is NaN-tolerant so a backend that can't emit instances (e.g. an
    unavailable SAM 3) still yields fg-IoU/BF."""
    be.fit(support_examples)
    fg_ious, aps, bfs = [], [], []
    for ex in test_examples:
        fg = be.foreground(ex.image, ex.feat_grid)
        fg_ious.append(metrics.foreground_iou(fg, ex.label_map))
        bfs.append(metrics.boundary_f1(fg, np.asarray(ex.label_map) > 0))
        try:
            insts = [m.mask for m in be.predict(ex.image, ex.feat_grid)]
            aps.append(metrics.instance_ap(insts, gt_labels=ex.label_map)["ap"] if insts else 0.0)
        except Exception:
            aps.append(float("nan"))
    return {
        "fg_iou": float(np.mean(fg_ious)),
        "ap": float(np.nanmean(aps)) if not np.all(np.isnan(aps)) else float("nan"),
        "bf": float(np.mean(bfs)),
    }


def _to_examples(pairs, feats):
    # keep the ORIGINAL (possibly per-instance) label map so instance-AP has true GT
    # instances; the backends binarize to fg internally in fit(), and foreground_iou uses
    # ``> 0``, so a per-instance map is safe for every metric (DSB's AP was 0 when binarized).
    return [LabeledExample(im, f, np.asarray(l)) for (im, l), f in zip(pairs, feats)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["dsb2018", "fewshot"], default="fewshot")
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", default="/tmp/asg_cache")
    ap.add_argument("--pool", type=int, default=10)
    ap.add_argument("--test", type=int, default=24)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--backends", default="correspondence,head,insid3")
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--backbone", default="auto", choices=["auto", "vit", "convnext"])
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--superres", type=int, default=1,
                    help="sub-patch shift-merge densification factor (1=off, 2 or 4)")
    ap.add_argument("--jbu", action="store_true",
                    help="edge-guided JBU feature snapping (parameter-free feature-level CRF)")
    args = ap.parse_args()

    cfg = RunConfig(device="auto", cache_dir=args.cache,
                    encoder=EncoderConfig(model_id=args.model, resolution=args.res,
                                          backbone=args.backbone, convnext_stage=args.stage,
                                          superres_factor=args.superres, jbu=args.jbu))
    dev = cfg.device_resolved()
    load = load_fewshot if args.dataset == "fewshot" else load_dsb2018
    train = load(args.data, "support" if args.dataset == "fewshot" else "train", args.pool)
    test = load(args.data, "test", args.test)
    enc = CachedEncoder(cfg, dev, args.cache)
    sup = _to_examples(train, [enc.extract(im) for im, _ in train])
    tst = _to_examples(test, [enc.extract(im) for im, _ in test])

    from scripts.al_testbed import make_backend

    print(f"device={dev} res={args.res} pool={len(sup)} test={len(tst)}", flush=True)
    print(f"{'backend':>14} {'fg_iou':>8} {'ap':>8} {'bf':>8}", flush=True)
    for name in args.backends.split(","):
        try:
            be = make_backend(name, cfg, dev)
            m = evaluate_backend(be, sup, tst)
            print(f"{name:>14} {m['fg_iou']:>8.3f} {m['ap']:>8.3f} {m['bf']:>8.3f}", flush=True)
        except Exception as e:
            print(f"{name:>14} UNAVAILABLE: {repr(e)[:80]}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
