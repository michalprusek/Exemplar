"""Segmentation metrics.

- ``foreground_iou`` — the spike's semantic metric (mean fg IoU), our regression anchor.
- ``instance_ap`` — the DSB2018 / Kaggle metric: AP@t = TP / (TP + FP + FN) with greedy
  IoU matching, averaged over t in {0.5, 0.55, ..., 0.95}. This is the honest
  instance-quality number the semantic IoU cannot capture.
- ``panoptic_sq_rq`` — panoptic SQ (mean matched IoU) and RQ (F1 of matches).

GT may be a label map (``gt_labels``) or a list of per-instance boolean masks
(``gt_masks``). The mask-list form is overlap-capable — instances may share pixels,
which a single label map physically cannot represent.
"""
from __future__ import annotations

import numpy as np

DEFAULT_THRESHOLDS = np.arange(0.5, 1.0, 0.05)


def foreground_iou(pred_mask: np.ndarray, gt_labels: np.ndarray) -> float:
    pred = np.asarray(pred_mask, bool)
    gt = np.asarray(gt_labels) > 0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union else 1.0


def cldice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Centerline Dice (Shit et al. CVPR'21) — the metric designed for TUBULAR structures
    (vessels, microtubules). Harmonic mean of topology precision (pred skeleton inside GT)
    and topology sensitivity (GT skeleton inside pred). Unlike region IoU, a broken/
    disconnected thin structure is heavily penalised because its skeleton leaves the mask.
    Both empty -> 1.0; one empty -> 0.0."""
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


def boundary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, tol: int = 2) -> float:
    """Boundary F1 (BF score): precision/recall of boundary pixels matched within ``tol``
    pixels, harmonic mean. Rewards small-object / thin-edge accuracy that region IoU
    washes out. Both empty -> 1.0; one empty -> 0.0."""
    from scipy.ndimage import binary_erosion, distance_transform_edt

    p = np.asarray(pred_mask, bool)
    g = np.asarray(gt_mask, bool)
    if not p.any() and not g.any():
        return 1.0
    if not p.any() or not g.any():
        return 0.0
    pb = p ^ binary_erosion(p)
    gb = g ^ binary_erosion(g)
    g_dt = distance_transform_edt(~gb)
    p_dt = distance_transform_edt(~pb)
    prec = float((g_dt[pb] <= tol).mean()) if pb.any() else 0.0
    rec = float((p_dt[gb] <= tol).mean()) if gb.any() else 0.0
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def _labels_to_masks(gt_labels: np.ndarray) -> list[np.ndarray]:
    gt = np.asarray(gt_labels)
    return [gt == i for i in np.unique(gt) if i != 0]


def _resolve_gt(gt_labels, gt_masks) -> list[np.ndarray]:
    if gt_masks is not None:
        return [np.asarray(m, bool) for m in gt_masks]
    if gt_labels is not None:
        return _labels_to_masks(gt_labels)
    raise ValueError("provide gt_labels or gt_masks")


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def _bbox(m: np.ndarray):
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    return ys.min(), ys.max(), xs.min(), xs.max()


def _iou_matrix(preds: list[np.ndarray], gts: list[np.ndarray]) -> np.ndarray:
    """Pairwise IoU ``[P, G]``, computed once. Bounding-box disjoint pairs skip the
    pixel AND/OR (they have IoU 0), which is the common case for many instances."""
    P, G = len(preds), len(gts)
    iou = np.zeros((P, G), np.float32)
    p_area = [int(p.sum()) for p in preds]
    g_area = [int(g.sum()) for g in gts]
    p_box = [_bbox(p) for p in preds]
    g_box = [_bbox(g) for g in gts]
    for pi in range(P):
        pb = p_box[pi]
        if pb is None:
            continue
        for gi in range(G):
            gb = g_box[gi]
            if gb is None:
                continue
            if pb[1] < gb[0] or gb[1] < pb[0] or pb[3] < gb[2] or gb[3] < pb[2]:
                continue  # bounding boxes disjoint -> IoU 0
            inter = int(np.logical_and(preds[pi], gts[gi]).sum())
            if inter == 0:
                continue
            iou[pi, gi] = inter / (p_area[pi] + g_area[gi] - inter)
    return iou


def _match_from_iou(iou: np.ndarray, n_pred: int, n_gt: int, thr: float):
    """Greedy IoU matching at threshold ``thr`` from a precomputed matrix."""
    pi_idx, gi_idx = np.where(iou >= thr)
    vals = iou[pi_idx, gi_idx]
    order = np.argsort(-vals)
    used_p: set[int] = set()
    used_g: set[int] = set()
    matched_ious = []
    for o in order:
        pi, gi = int(pi_idx[o]), int(gi_idx[o])
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matched_ious.append(float(vals[o]))
    tp = len(matched_ious)
    return tp, n_pred - tp, n_gt - tp, matched_ious


def _match(preds: list[np.ndarray], gts: list[np.ndarray], thr: float):
    iou = _iou_matrix(preds, gts)
    return _match_from_iou(iou, len(preds), len(gts), thr)


def instance_ap(
    pred_instances: list[np.ndarray],
    gt_labels: np.ndarray | None = None,
    *,
    gt_masks: list[np.ndarray] | None = None,
    thresholds: np.ndarray = DEFAULT_THRESHOLDS,
) -> dict:
    preds = [np.asarray(p, bool) for p in pred_instances]
    gts = _resolve_gt(gt_labels, gt_masks)
    iou = _iou_matrix(preds, gts)
    per_thresh = {}
    for t in thresholds:
        tp, fp, fn, _ = _match_from_iou(iou, len(preds), len(gts), float(t))
        denom = tp + fp + fn
        per_thresh[round(float(t), 2)] = tp / denom if denom else 1.0
    ap = float(np.mean(list(per_thresh.values())))
    return {"ap": ap, "ap50": per_thresh[0.5], "per_thresh": per_thresh}


def panoptic_sq_rq(
    pred_instances: list[np.ndarray],
    gt_labels: np.ndarray | None = None,
    *,
    gt_masks: list[np.ndarray] | None = None,
    match_thr: float = 0.5,
) -> dict:
    preds = [np.asarray(p, bool) for p in pred_instances]
    gts = _resolve_gt(gt_labels, gt_masks)
    tp, fp, fn, matched_ious = _match(preds, gts, match_thr)
    sq = float(np.mean(matched_ious)) if matched_ious else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) else 1.0
    return {"pq": sq * rq, "sq": sq, "rq": rq, "tp": tp, "fp": fp, "fn": fn}
