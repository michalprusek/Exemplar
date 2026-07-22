#!/usr/bin/env python
"""Overlap-honesty benchmark (what DSB2018 cannot test).

Synthetic translucent discs with known overlap. We show that the per-instance
representation + SAM refine PRESERVE overlap (each disc its own mask, masks share
pixels), while a connected-components label map DESTROYS it (overlapping discs merge
into one component). Per-instance seeds are the GT distance-transform peaks, standing
in for the corrected exemplars the AL loop accumulates (separating overlapping
same-class instances from correspondence alone needs per-exemplar seeds — a known
design limitation the correction loop resolves).

Run on tulen:
  ~/dinov3_env/bin/python scripts/overlap_eval.py --n 20 --instances 3 --overlap 0.35
"""
import argparse

import numpy as np
from scipy import ndimage

from active_segmenter.config import ClusterConfig, RefineConfig, RunConfig
from active_segmenter.eval import metrics
from active_segmenter.eval.datasets import make_synthetic_overlap
from active_segmenter.propose import instances
from active_segmenter.refine import build_refiner
from active_segmenter.types import InstanceMask


def peak_seed_masks(gt_masks):
    """One tiny seed mask at each GT instance's distance-transform peak."""
    seeds = []
    for k, m in enumerate(gt_masks):
        dt = ndimage.distance_transform_edt(m)
        yy, xx = np.unravel_index(int(np.argmax(dt)), dt.shape)
        s = np.zeros_like(m)
        s[yy, xx] = True
        seeds.append(InstanceMask(mask=s, points=None, class_id=1, instance_id=k))
    return seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument("--overlap", type=float, default=0.35)
    args = ap.parse_args()

    cfg = RunConfig(device="auto")
    dev = cfg.device_resolved()
    data = make_synthetic_overlap(args.n, args.instances, args.overlap, seed=0)
    # race prompt strength (point vs mask-prompt) × overlap policy (separate vs amodal)
    configs = {
        "point-separate": RefineConfig(kind="sam", prompt_mode="point", sam_negatives=True),
        "point-amodal": RefineConfig(kind="sam", prompt_mode="point", amodal=True, sam_negatives=False),
        "mask-separate": RefineConfig(kind="sam", prompt_mode="mask", sam_negatives=True),
        "mask-amodal": RefineConfig(kind="sam", prompt_mode="mask", amodal=True, sam_negatives=False),
    }

    def eval_refiner(refiner):
        aps, kept = [], 0
        for image, gt_masks in data:
            refined = refiner.refine(image, peak_seed_masks(gt_masks), feat_grid=None)
            pred = [m.mask for m in refined]
            aps.append(metrics.instance_ap(pred, gt_masks=gt_masks)["ap"])
            if any(np.logical_and(pred[i], pred[j]).any()
                   for i in range(len(pred)) for j in range(i + 1, len(pred))):
                kept += 1
        return float(np.mean(aps)), kept

    results = {name: eval_refiner(build_refiner(rc, dev)) for name, rc in configs.items()}

    ap_cc, cc_merged = [], 0
    for image, gt_masks in data:
        union = np.any(gt_masks, axis=0)
        lab, n = ndimage.label(union)
        cc_masks = [lab == i for i in range(1, n + 1)]
        ap_cc.append(metrics.instance_ap(cc_masks, gt_masks=gt_masks)["ap"])
        if n < len(gt_masks):
            cc_merged += 1

    print(f"device={dev} n={args.n} instances/img={args.instances} overlap_frac={args.overlap}", flush=True)
    print(f"{'refiner':>16} {'AP':>7} {'overlap-preserved':>18}", flush=True)
    for name, (ap, kept) in results.items():
        print(f"{name:>16} {ap:>7.3f} {f'{kept}/{args.n} imgs':>18}", flush=True)
    print(f"{'conn-components':>16} {np.mean(ap_cc):>7.3f} "
          f"{f'DESTROYED {cc_merged}/{args.n}':>18}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
