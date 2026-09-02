#!/usr/bin/env python
"""Structure width per dataset, in units of the backbone's effective patch stride.

The encoder sees a 672-px input, so one 16-px patch covers 16 * (native / 672) NATIVE pixels. A
dataset asks for sub-patch structure when its foreground half-width falls below that.

Reports three statistics, because the choice matters and the obvious one is wrong:
  * mean DT radius   -- what the method's own descriptor uses (head_fusion_backend.py:653), and a
                        poor estimator on bimodal masks: rozpad (a thick spheroid core plus fine
                        debris) reads 29.5 px against a median of 5.7.
  * MEDIAN DT radius -- what registry C79 found tracks the prior bank's value.
  * fraction of foreground thinner than one patch.

Needs the datasets, so it runs where they live (tulen).
  PYTHONPATH=<repo> python scripts/measure_structure_width.py
"""
import sys, numpy as np
from scipy import ndimage as ndi
from active_segmenter.eval.registry import PANEL, load_dataset

DS = ["spheroidj", "rozpad", "dsb2018", "monuseg", "ctc_u373", "bbbc010", "bacteria",
      "drive", "hrf", "isbi2012em", "fisbe"]

print("dataset,stride_px,mean_over_stride,median_over_stride,frac_subpatch")
for name in DS:
    pool, _ = load_dataset(PANEL[name], 20, 10000, seed=0)
    mu, md, fr, nat = [], [], [], []
    for _img, _lb in pool:
        lb = np.asarray(_lb) > 0
        if lb.sum() == 0:
            continue
        r = ndi.distance_transform_edt(lb)[lb]
        stride = 16.0 * max(lb.shape[:2]) / 672.0
        mu.append(r.mean() / stride); md.append(np.median(r) / stride)
        fr.append((r < stride).mean()); nat.append(max(lb.shape[:2]))
    print(f"{name},{16.0 * np.median(nat) / 672.0:.2f},{np.median(mu):.4f},"
          f"{np.median(md):.4f},{np.median(fr):.4f}")
