"""Non-parametric uncertainty acquisition.

Uncertainty is the fraction of ambiguous patches in a pre-label (|margin| below a
threshold) — computed through the frozen features + memory bank, never MC-dropout
over the backbone. Ranks the pool by that per-image uncertainty, descending.
"""
from __future__ import annotations

import numpy as np

from active_segmenter.acquire.base import AcqContext


def ambiguous_fraction(score_map: np.ndarray, eps: float = 0.03) -> float:
    s = np.asarray(score_map, np.float32)
    return float(np.mean(np.abs(s) < eps))


class UncertaintyAcq:
    def rank(self, pool: list[int], ctx: AcqContext) -> list[int]:
        return sorted(pool, key=lambda i: ctx.uncertainty[i], reverse=True)
