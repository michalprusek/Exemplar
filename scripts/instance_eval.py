#!/usr/bin/env python
"""M3 gate — instance decomposition quality on DSB2018.

Builds a bank from N annotated images, prelabels each test image with kNN
correspondence, decomposes the score map into per-instance masks via (a) the
library's clustering and (b) a connected-components baseline, and reports instance
AP (DSB metric) for each so the clustering's added value is measured, not assumed.

Run on tulen:
  ~/dinov3_env/bin/python scripts/instance_eval.py --cache /disk1/prusek/asg_cache \
      --data /disk1/prusek/dsb2018 --pool 60 --test 50 --bank 10 --res 672
"""
import argparse

import numpy as np

from active_segmenter.config import ClusterConfig, EncoderConfig, MatchConfig, RunConfig
from active_segmenter.encoder.cache import EmbeddingCache
from active_segmenter.encoder.dinov3 import Dinov3Encoder
from active_segmenter.eval import metrics
from active_segmenter.eval.datasets import load_dsb2018
from active_segmenter.membank.bank import MemoryBank
from active_segmenter.propose import correspondence as corr
from active_segmenter.propose import instances


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/asg_cache")
    ap.add_argument("--data", default="/tmp/dsb2018")
    ap.add_argument("--pool", type=int, default=60)
    ap.add_argument("--test", type=int, default=50)
    ap.add_argument("--bank", type=int, default=10)
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--use-features", action="store_true")
    ap.add_argument("--sam", action="store_true", help="also SAM-refine the clustering masks")
    args = ap.parse_args()

    cfg = RunConfig(device="auto", cache_dir=args.cache, encoder=EncoderConfig(resolution=args.res))
    dev = cfg.device_resolved()
    train = load_dsb2018(args.data, "train", args.pool)
    test = load_dsb2018(args.data, "test", args.test)
    enc = Dinov3Encoder(cfg.encoder, dev)
    cache = EmbeddingCache(args.cache)
    extra = f"{cfg.encoder.model_id.split('/')[-1]}-res{args.res}"
    tr = [cache.get_or_compute(im, extra, lambda im=im: enc.extract(im)) for im, _ in train]
    tf = [cache.get_or_compute(im, extra, lambda im=im: enc.extract(im)) for im, _ in test]

    bank = MemoryBank()
    for i in range(args.bank):
        binlabel = (np.asarray(train[i][1]) > 0).astype(int)
        bank.add_from_annotation(tr[i], binlabel, {1: 1}, round=0)

    ccfg = ClusterConfig(score_thresh=0.0, min_patches=1, distance_threshold=1.5,
                         use_features=args.use_features, xy_weight=0.3)
    mc = MatchConfig(topk=5, bidirectional=False)
    refiner = None
    if args.sam:
        from active_segmenter.config import RefineConfig
        from active_segmenter.refine import build_refiner
        refiner = build_refiner(RefineConfig(kind="sam"), dev)

    ap_clu, ap_cc, ap50_clu, ap50_cc, ap_sam, ap50_sam = [], [], [], [], [], []
    for feat, (im, lbl) in zip(tf, test):
        shape = np.asarray(lbl).shape
        score = corr.score_map(feat, bank, 1, mc)
        clu = instances.upsample_masks(
            instances.decompose(score, ccfg, 1, feat_grid=feat if args.use_features else None), shape)
        cc = instances.upsample_masks(instances.connected_components(score, ccfg, 1), shape)
        clu_m = [m.mask for m in clu] or [np.zeros(shape, bool)]
        cc_m = [m.mask for m in cc] or [np.zeros(shape, bool)]
        ac = metrics.instance_ap(clu_m, gt_labels=lbl)
        ax = metrics.instance_ap(cc_m, gt_labels=lbl)
        ap_clu.append(ac["ap"]); ap50_clu.append(ac["ap50"])
        ap_cc.append(ax["ap"]); ap50_cc.append(ax["ap50"])
        if refiner is not None:
            ref = refiner.refine(im, clu, feat_grid=feat)
            ref_m = [m.mask for m in ref] or [np.zeros(shape, bool)]
            asam = metrics.instance_ap(ref_m, gt_labels=lbl)
            ap_sam.append(asam["ap"]); ap50_sam.append(asam["ap50"])

    print(f"device={dev} res={args.res} bank={args.bank} test={len(test)} use_features={args.use_features} sam={args.sam}")
    print(f"clustering  AP={np.mean(ap_clu):.3f}  AP50={np.mean(ap50_clu):.3f}")
    print(f"conn-comp   AP={np.mean(ap_cc):.3f}  AP50={np.mean(ap50_cc):.3f}")
    if refiner is not None:
        print(f"clu+SAM     AP={np.mean(ap_sam):.3f}  AP50={np.mean(ap50_sam):.3f}")
    print("DONE")


if __name__ == "__main__":
    main()
