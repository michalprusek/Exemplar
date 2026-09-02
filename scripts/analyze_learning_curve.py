#!/usr/bin/env python
"""Experiment A+B — learning curve to saturation, ceiling, and trivial baselines.

Answers: is accuracy growing with #annotated images, where is the ceiling, and how
does the in-context model compare to no-learning baselines (Otsu, Otsu+watershed)?

- Learning curve: bank size 3..60, per-patch kNN correspondence fg IoU on a held-out
  test set, averaged over several random bank orders (mean +/- std).
- Ceiling: the whole-pool bank (largest size) is the in-context training-free ceiling.
- Instance AP: correspondence -> clustering -> SAM refine at a few bank sizes.
- Baselines: Otsu threshold (best polarity) fg IoU; Otsu + distance-watershed instance AP.

Run on tulen:
  ~/dinov3_env/bin/python scripts/analyze_learning_curve.py --cache /disk1/prusek/asg_cache \
      --data /disk1/prusek/dsb2018 --pool 60 --test 40 --res 672
"""
import argparse
import time

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.transform import resize

from active_segmenter.config import ClusterConfig, EncoderConfig, MatchConfig, RefineConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval import metrics
from active_segmenter.eval.datasets import load_dsb2018
from active_segmenter.membank.bank import MemoryBank
from active_segmenter.propose import correspondence as corr
from active_segmenter.propose import instances
from active_segmenter.refine import build_refiner

MC = MatchConfig(topk=5, bidirectional=False)


def build_bank(feats, data, idxs):
    bank = MemoryBank()
    for i in idxs:
        binlabel = (np.asarray(data[i][1]) > 0).astype(int)
        bank.add_from_annotation(feats[i], binlabel, {1: 1}, round=0)
    return bank


def corr_fg_iou(feat, lbl, bank, dev):
    s = corr.score_map(feat, bank, 1, MC, device=dev)
    pf = resize((s > 0).astype(np.float32), np.asarray(lbl).shape, order=0,
                mode="edge", anti_aliasing=False) > 0.5
    return metrics.foreground_iou(pf, lbl)


def mean_test_iou(tef, test, bank, dev):
    return float(np.mean([corr_fg_iou(f, l, bank, dev) for f, (im, l) in zip(tef, test)]))


# ---- baselines --------------------------------------------------------------
def otsu_iou(image, lbl):
    g = np.asarray(image, np.float32)
    if g.ndim == 3:
        g = g.mean(2)
    try:
        t = threshold_otsu(g)
    except Exception:
        return 0.0
    a = metrics.foreground_iou(g > t, lbl)
    b = metrics.foreground_iou(g < t, lbl)  # best polarity (favours the baseline)
    return max(a, b)


def otsu_watershed_ap(image, lbl):
    g = np.asarray(image, np.float32)
    if g.ndim == 3:
        g = g.mean(2)
    try:
        t = threshold_otsu(g)
    except Exception:
        return 0.0
    fg = g > t if (g > threshold_otsu(g)).mean() < 0.5 else g < t  # nuclei = minority class
    dist = ndimage.distance_transform_edt(fg)
    coords = peak_local_max(dist, min_distance=5, labels=fg)
    markers = np.zeros(fg.shape, int)
    for k, (y, x) in enumerate(coords, 1):
        markers[y, x] = k
    if markers.max() == 0:
        return 0.0
    lab = watershed(-dist, markers, mask=fg)
    preds = [lab == i for i in range(1, lab.max() + 1)]
    return metrics.instance_ap(preds or [np.zeros(fg.shape, bool)], gt_labels=lbl)["ap"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/asg_cache")
    ap.add_argument("--data", default="/tmp/dsb2018")
    ap.add_argument("--pool", type=int, default=60)
    ap.add_argument("--test", type=int, default=40)
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    cfg = RunConfig(device="auto", cache_dir=args.cache, encoder=EncoderConfig(resolution=args.res))
    dev = cfg.device_resolved()
    train = load_dsb2018(args.data, "train", args.pool)
    test = load_dsb2018(args.data, "test", args.test)
    enc = CachedEncoder(cfg, dev, args.cache)
    trf = [enc.extract(im) for im, _ in train]
    tef = [enc.extract(im) for im, _ in test]

    sizes = [s for s in [3, 5, 10, 15, 20, 30, 45, 60] if s <= args.pool]
    print(f"device={dev} res={args.res} pool={len(train)} test={len(test)} seeds={args.seeds}")

    # --- learning curve (fg IoU, kNN correspondence) ---
    print("\n# learning curve (kNN correspondence fg IoU)")
    print(f"{'bank':>5} {'IoU_mean':>9} {'IoU_std':>8}")
    curve = {}
    for k in sizes:
        vals = []
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            idxs = list(rng.choice(len(train), size=k, replace=False))
            vals.append(mean_test_iou(tef, test, build_bank(trf, train, idxs), dev))
        curve[k] = (float(np.mean(vals)), float(np.std(vals)))
        print(f"{k:>5} {curve[k][0]:>9.3f} {curve[k][1]:>8.3f}")

    # --- instance AP with SAM at a few sizes ---
    print("\n# instance AP (kNN -> clustering -> SAM refine)")
    refiner = build_refiner(RefineConfig(kind="sam"), dev)
    ccfg = ClusterConfig(score_thresh=0.0, min_patches=1, distance_threshold=1.5)
    for k in [x for x in [5, 20, 60] if x <= args.pool]:
        rng = np.random.default_rng(0)
        idxs = list(rng.choice(len(train), size=k, replace=False))
        bank = build_bank(trf, train, idxs)
        aps = []
        for feat, (im, lbl) in zip(tef, test):
            sh = np.asarray(lbl).shape
            grid = instances.decompose(corr.score_map(feat, bank, 1, MC, device=dev), ccfg, 1)
            up = instances.upsample_masks(grid, sh)
            ref = refiner.refine(im, up, feat_grid=feat)
            aps.append(metrics.instance_ap([m.mask for m in ref] or [np.zeros(sh, bool)],
                                           gt_labels=lbl)["ap"])
        print(f"bank={k:>3}  instance_AP(SAM)={np.mean(aps):.3f}")

    # --- trivial baselines ---
    print("\n# trivial baselines (no learning)")
    otsu = np.mean([otsu_iou(im, l) for im, l in test])
    ws = np.mean([otsu_watershed_ap(im, l) for im, l in test])
    print(f"Otsu (best polarity) fg IoU = {otsu:.3f}")
    print(f"Otsu + watershed instance AP = {ws:.3f}")
    print("DONE")


if __name__ == "__main__":
    main()
