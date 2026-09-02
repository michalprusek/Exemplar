"""Out-of-distribution confidence gate (W3).

The kvasir failure analysis showed catastrophic per-image collapses (fg-IoU 0.11 vs 0.93) that
correlate with NO simple image statistic — they are *semantic/appearance* OOD: the test image is
unlike anything in the (few) labeled support. A model-aware segmenter has no way to know it is
extrapolating. This module gives a **model-free** confidence signal from the frozen DINOv3-CLS
geometry alone: how far is a test image from the labeled support manifold, measured in units of the
support's own spread.

Two uses, same score:
- **Confidence gate (deployment):** high OOD distance → flag the auto-segmentation as low-confidence
  (don't silently trust it); degrade gracefully.
- **Active-learning signal:** the most-OOD unlabeled image is exactly the one worth labeling next —
  it complements the low-K coverage cold-start (labels the manifold's uncovered frontier).
"""
from __future__ import annotations

import numpy as np


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = x[None]
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def support_spread(cls_support: np.ndarray) -> float:
    """Mean nearest-neighbour cosine distance WITHIN the support = the natural in-distribution
    scale. A test image farther than a few multiples of this is genuinely OOD, not just normal
    within-support variation."""
    S = _unit(cls_support)
    if len(S) < 2:
        return 1.0
    sim = S @ S.T
    np.fill_diagonal(sim, -1.0)
    return float((1.0 - sim.max(1)).mean()) + 1e-6


def ood_scores(cls_support: np.ndarray, cls_test: np.ndarray):
    """Per-test OOD score = nearest-support cosine distance, normalised by the support spread.

    Returns ``(raw, z)``: ``raw[i] = 1 - max_j cos(test_i, support_j)`` (0 = identical to a labeled
    image, higher = more novel); ``z[i] = raw[i] / support_spread`` (in units of in-distribution
    scale — z≳2–3 flags genuine OOD). Model-free (CLS only), so it works from the very first label."""
    S, T = _unit(cls_support), _unit(cls_test)
    raw = 1.0 - (T @ S.T).max(1)
    z = raw / support_spread(cls_support)
    return raw.astype(np.float32), z.astype(np.float32)
