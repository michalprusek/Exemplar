"""Support leave-one-out grounded uncertainty — a GT-grounded error field.

The AL curse (uncertainty barely beats random) is that *confident errors have low
uncertainty*: a variance/disagreement signal can't see a prediction that is stable but
wrong. This module sidesteps it by grounding uncertainty in the ONE thing that is labelled —
the support set itself. For each labelled support image, segment it from the REST of the bank
(leave-one-out) and score the prediction against its KNOWN ground-truth mask. ``error = 1 -
IoU`` is then a *measured* proxy error (not a variance), marking where the frozen segmenter
fails on its own manifold. Pool candidates near those high-error support regions on the
sphere inherit the error and are worth annotating.

Cleaner than pure prompt-ensemble variance (which measures sensitivity to support choice, not
error) and than CCV cycle-consistency (which is circular through the query's own prediction);
here the cycle closes on the support's real GT. Pure, model-free (predict/gt/sim injected).
See [[frozen-loop-al-direction]].
"""
from __future__ import annotations

import numpy as np


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def support_loo_errors(support_idxs, predict_fn, gt_fn) -> dict:
    """``{support_idx: 1 - IoU(pred_from_others, known_gt)}``.

    ``predict_fn(others, s) -> bool mask`` segments support image ``s`` conditioned on the
    OTHER support images (``s`` left out); ``gt_fn(s) -> bool mask`` is ``s``'s known GT."""
    support_idxs = list(support_idxs)
    errs = {}
    for s in support_idxs:
        others = [t for t in support_idxs if t != s]
        errs[s] = 1.0 - _iou(predict_fn(others, s), gt_fn(s))
    return errs


def propagate_errors_to_pool(pool, support_errs: dict, cls) -> dict:
    """Each pool candidate inherits nearby support error:
    ``score[x] = sum_s max(0, cos(x, s)) * error_s`` over labelled support ``s``. High = the
    candidate sits near a region where the frozen segmenter demonstrably fails."""
    cls = np.asarray(cls, np.float32)
    norm = np.linalg.norm(cls, axis=1, keepdims=True)
    unit = cls / np.maximum(norm, 1e-12)
    s_idx = list(support_errs.keys())
    if not s_idx:
        return {x: 0.0 for x in pool}
    s_mat = unit[s_idx]                                  # [S, D]
    e_vec = np.array([support_errs[s] for s in s_idx], np.float32)  # [S]
    out = {}
    for x in pool:
        sim = np.maximum(0.0, unit[x] @ s_mat.T)         # [S]
        out[x] = float(np.dot(sim, e_vec))
    return out
