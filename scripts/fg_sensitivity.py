"""fg-quality -> instance-AP SENSITIVITY curve, and a test of WHICH KIND of fg error kills AP.

Why this exists. The project has exactly TWO points on the fg->AP map: our real foreground, and a
perfect oracle foreground. Everything in between is uncharacterised, so "how much better must the
foreground get to beat a specialist?" has never been answerable, and "foreground is the whole
bottleneck" rests on a single oracle point.

The fixed-harness campaign makes that gap urgent, because the two datasets disagree:

    dsb2018   ours fg 0.846 -> AP 0.502 | cellpose_ft fg 0.872 -> AP 0.647   (+0.026 fg = +0.146 AP)
    monuseg   ours fg 0.622 -> AP 0.220 | cellpose_ft fg 0.716 -> AP 0.419   (+0.093 fg = +0.199 AP)

On dsb2018 a 0.026 foreground difference cannot linearly explain a 0.146 AP difference, so either AP
is violently nonlinear there, or OUR foreground errors are structurally worse than theirs at equal
fg-IoU. This script separates those two possibilities.

TWO degradations of the GT foreground, swept to the SAME fg-IoU levels:

  DILATE  -- grows every object outward. Models the measured failure mode (C16: precision 0.683,
             recall 0.885, detection 0.97, boundary-band err 0.52 => diffuse bleed into stroma).
             Crucially it BRIDGES neighbouring objects, which merges instances.
  SPECKLE -- adds the SAME amount of false-positive area, but at random background locations away
             from the objects. Same fg-IoU, same over-prediction magnitude, but it does NOT bridge.

If AP collapses under DILATE and survives under SPECKLE at equal fg-IoU, then the enemy is not the
AMOUNT of false foreground but its PLACEMENT between touching objects -- which means the lever is a
bridge-suppressing objective, not a generically better foreground.

Everything downstream (decoder, r*, merge_cos, features) is best_v2's, unchanged.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from active_segmenter.config import EncoderConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval.registry import PANEL, load_dataset
from active_segmenter.eval.scoring import primary_key, score_prediction
from active_segmenter.segment.base import LabeledExample
from active_segmenter.segment.head_fusion_backend import _affinity_watershed_instances, _ridge_map
from scripts.al_testbed import make_backend

MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
CACHE = os.environ.get("ASG_SENS_CACHE", "/disk1/prusek/asg_cache_oracle")
DATASETS = ["monuseg", "dsb2018", "ctc_u373"]
RADII = [0, 1, 2, 3, 4, 6, 8]
SEEDS = int(os.environ.get("ASG_SENS_SEEDS", "2"))


def disk(r):
    if r <= 0:
        return None
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (y * y + x * x) <= r * r


def speckle_like(fg, added_px, r, rng):
    """Add ~`added_px` false-positive pixels as disks of radius r at random BACKGROUND locations,
    kept away from the true objects so they cannot bridge them."""
    if added_px <= 0 or r <= 0:
        return fg.copy()
    se = disk(r)
    far = ~ndi.binary_dilation(fg, structure=disk(max(2 * r, 3)))   # only well outside the objects
    ys, xs = np.nonzero(far)
    if len(ys) == 0:
        return fg.copy()
    per_blob = max(1, int(se.sum()))
    n = max(1, int(round(added_px / per_blob)))
    n = min(n, len(ys))
    pick = rng.choice(len(ys), size=n, replace=False)
    seeds = np.zeros_like(fg, bool)
    seeds[ys[pick], xs[pick]] = True
    return fg | ndi.binary_dilation(seeds, structure=se)


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def main():
    dev = RunConfig(device="auto").device_resolved()
    cfg = RunConfig(device="auto", cache_dir=CACHE,
                    encoder=EncoderConfig(model_id=MODEL, resolution=672))
    enc = CachedEncoder(cfg, dev, CACHE)
    pk = primary_key("instance_ap")

    print(f"# fg-quality -> instance-AP sensitivity | {SEEDS} seed(s), K=8, pool 20 / test 24, res 672")
    print("# DILATE = bleeds outward, BRIDGES neighbours. SPECKLE = same false-positive area, placed apart.\n")

    for name in DATASETS:
        try:
            pool, test = load_dataset(PANEL[name], 20, 24, seed=0)
        except Exception as e:
            print(f"{name}: LOAD FAILED {type(e).__name__}: {e}", flush=True)
            continue

        print(f"=== {name} ===")
        print(f"{'r':>3} {'dil_fgIoU':>10} {'dil_AP':>8} {'spk_fgIoU':>10} {'spk_AP':>8}")
        acc = {r: {"di": [], "da": [], "si": [], "sa": []} for r in RADII}

        for seed in range(SEEDS):
            be = make_backend("head_fusion_best_cgate_film_nobank", cfg, dev, enc=enc)
            sub = list(np.random.default_rng(seed).choice(len(pool), 8, replace=False))
            be.fit([LabeledExample(pool[i][0], enc.extract(pool[i][0]), np.asarray(pool[i][1]))
                    for i in sub])
            rng = np.random.default_rng(1000 + seed)

            for im, gt in test:
                gt = np.asarray(gt)
                fg = gt > 0
                fgrid = enc.extract(im)
                ridge = _ridge_map(be._channel(im))

                def ap_of(mask):
                    inst = [m.mask for m in _affinity_watershed_instances(
                        mask, ridge, fgrid, 1, r_star=be._inst_r, merge_cos=be._inst_merge_cos)]
                    return float(score_prediction("instance_ap", mask, gt, inst)[pk])

                for r in RADII:
                    dil = fg.copy() if r == 0 else ndi.binary_dilation(fg, structure=disk(r))
                    spk = speckle_like(fg, int(dil.sum() - fg.sum()), r, rng)
                    acc[r]["di"].append(iou(dil, fg)); acc[r]["da"].append(ap_of(dil))
                    acc[r]["si"].append(iou(spk, fg)); acc[r]["sa"].append(ap_of(spk))

        for r in RADII:
            a = acc[r]
            print(f"{r:>3} {np.mean(a['di']):10.3f} {np.mean(a['da']):8.3f} "
                  f"{np.mean(a['si']):10.3f} {np.mean(a['sa']):8.3f}", flush=True)
        print()

    print("READ IT LIKE THIS")
    print("  Find the row whose dil_fgIoU matches our REAL fg-IoU (monuseg 0.622, dsb2018 0.846,")
    print("  ctc_u373 0.739) and compare dil_AP there with our REAL AP (0.220 / 0.502 / 0.428).")
    print("  dil_AP ~= our AP  => our foreground error IS bleed; fixing bleed is the whole game.")
    print("  dil_AP >> our AP  => our error is structurally worse than bleed; something else is wrong.")
    print("  spk_AP >> dil_AP at equal fg-IoU => the AMOUNT of false foreground is not what kills AP;")
    print("                                      its PLACEMENT between touching objects is.")


if __name__ == "__main__":
    main()
