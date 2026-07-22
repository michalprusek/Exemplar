#!/usr/bin/env python
"""Measure the per-dataset morphology descriptors that will drive the adaptive gates, so the gate
calibration is DATA-DRIVEN not guessed. For each dataset's support pool prints:
  - thinness  = mean skeleton-to-area ratio of the GT (drives the clDice topology gate; GT-only, train)
  - contrast  = mean global RMS contrast std(gray01) of the image (drives the CLAHE gate; image-only,
                works at inference — CLAHE helps FAINT/low-contrast structures)
  - fg_faint  = mean |mean(fg)-mean(bg)| foreground/background separation (secondary faintness proxy)

Expected from the ablations: clDice wants HIGH thinness (hrf only); CLAHE wants LOW contrast (hrf, kvasir),
NOT high-contrast crumbs/spheroids. The printed values calibrate the gate thresholds.

Run: PYTHONPATH=. ~/dinov3_env/bin/python scripts/morph_descriptors.py --cache /disk1/prusek/asg_cache_panel
"""
import argparse

import numpy as np

from active_segmenter.eval.registry import PANEL, load_dataset
from active_segmenter.segment.head_fusion_backend import _gray01, _thinness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="spheroid,spheroidj,dsb2018,rozpad,kvasir,hrf")
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--cache", default="/tmp/asg_cache_panel")
    args = ap.parse_args()
    from skimage.morphology import skeletonize, convex_hull_image
    from scipy import ndimage

    def descriptors(lab):
        """Candidate tubularity descriptors on a GT mask (global). A TUBE of length L width w has
        skeleton≈L, area≈Lw → these separate elongated (tubular) from compact/small."""
        m = np.asarray(lab) > 0
        area = float(m.sum())
        if area < 4:
            return dict(skel_area=0, elong=0, skel_sqrt=0, solidity=1.0)
        skel = float(skeletonize(m).sum())
        try:
            hull = float(convex_hull_image(m).sum())
        except Exception:
            hull = area
        return dict(
            skel_area=skel / area,                 # OLD: 1/thickness — conflates thin with small/jagged
            elong=skel * skel / area,              # skeleton²/area ≈ aspect ratio L/w (scale-invariant)
            skel_sqrt=skel / np.sqrt(area),        # skeleton/√area ≈ √(aspect ratio)
            solidity=area / max(hull, 1.0),        # area/convex-hull: LOW for sparse/branchy tubular
        )

    names = [n for n in args.datasets.split(",") if n in PANEL]
    print(f"{'dataset':>10} {'skel/area':>9} {'elong':>8} {'skl/√A':>8} {'solidity':>8} "
          f"{'contrast':>8}  note", flush=True)
    for name in names:
        try:
            pool, _ = load_dataset(PANEL[name], args.pool, 2, seed=0)
        except Exception as e:
            print(f"{name:>10}  SKIP {repr(e)[:60]}", flush=True)
            continue
        acc = {k: [] for k in ("skel_area", "elong", "skel_sqrt", "solidity")}
        ct = []
        for im, lab in pool:
            ct.append(float(_gray01(im).std()))
            d = descriptors(lab)
            for k in acc:
                acc[k].append(d[k])
        print(f"{name:>10} {np.mean(acc['skel_area']):>9.3f} {np.mean(acc['elong']):>8.1f} "
              f"{np.mean(acc['skel_sqrt']):>8.2f} {np.mean(acc['solidity']):>8.3f} "
              f"{np.mean(ct):>8.3f}  {PANEL[name].note[:28]}", flush=True)
    print("MORPH_DONE")


if __name__ == "__main__":
    main()
