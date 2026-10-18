"""The two test metrics: foreground IoU (objects) and centreline Dice (vessels, membranes, filaments).

Source: active_segmenter/eval/metrics.py (AutoSeg), unchanged. clDice uses skimage.morphology.skeletonize.
"""
from __future__ import annotations

import numpy as np


def foreground_iou(pred_mask: np.ndarray, gt_labels: np.ndarray) -> float:
    pred = np.asarray(pred_mask, bool)
    gt = np.asarray(gt_labels) > 0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union else 1.0


def cldice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Centreline Dice (Shit et al., CVPR 2021): harmonic mean of the topology precision (predicted skeleton inside
    the mask) and the topology sensitivity (mask skeleton inside the prediction). Both empty -> 1; one empty -> 0."""
    from skimage.morphology import skeletonize

    p = np.asarray(pred_mask, bool)
    g = np.asarray(gt_mask, bool)
    if not p.any() and not g.any():
        return 1.0
    if not p.any() or not g.any():
        return 0.0
    sp = skeletonize(p)
    sg = skeletonize(g)
    tprec = float((sp & g).sum() / sp.sum()) if sp.sum() else 0.0
    tsens = float((sg & p).sum() / sg.sum()) if sg.sum() else 0.0
    if tprec + tsens == 0:
        return 0.0
    return float(2 * tprec * tsens / (tprec + tsens))
