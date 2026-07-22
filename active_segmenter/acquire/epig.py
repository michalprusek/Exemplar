"""EPIG-style acquisition (target distribution = the whole pool).

Published EPIG (Expected Predictive Information Gain) targets generalisation: it
prefers labels that most reduce predictive uncertainty ACROSS the target set, not
inputs that are merely globally uncertain (plain BALD's failure mode). Exact EPIG
is a classification-model quantity; here we use a documented non-parametric
approximation suited to the frozen-feature + memory-bank setting:

    score(i) ≈ uncertainty(i) × representativeness(i, pool)

where representativeness = mean cosine similarity of image i's CLS embedding to the
pool's embeddings. Labeling an uncertain AND representative image reduces pool-wide
uncertainty most; an uncertain outlier (high uncertainty, low representativeness)
is down-weighted, which is exactly EPIG's advantage over BALD.
"""
from __future__ import annotations

import numpy as np

from active_segmenter.acquire.base import AcqContext


def epig_scores(pool: list[int], ctx: AcqContext) -> dict[int, float]:
    cls = ctx.cls
    target = cls[pool]  # target distribution = the pool itself
    scores = {}
    for i in pool:
        rep = float(np.mean(cls[i] @ target.T))  # mean cosine sim to the pool
        scores[i] = float(ctx.uncertainty[i]) * rep
    return scores


class EpigAcq:
    def rank(self, pool: list[int], ctx: AcqContext) -> list[int]:
        s = epig_scores(pool, ctx)
        return sorted(pool, key=lambda i: s[i], reverse=True)
