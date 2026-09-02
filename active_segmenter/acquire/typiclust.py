"""TypiClust acquisition wrapper for the AL orchestrator.

The benchmarked winner (al_testbed on DSB2018): beats random and the
uncertainty/EPIG acquisitions, closes ~25% of the random->oracle gap, and its
ranking correlates +0.77 with the actual per-image accuracy gain (vs the old
EPIG's -0.19). Model-free — uses only the cached DINOv3 CLS embeddings of the
uploaded dataset (transductive; no extrapolation beyond the pool).
"""
from __future__ import annotations

from active_segmenter.acquire.base import AcqContext
from active_segmenter.acquire import transductive


class TypiClustAcq:
    def rank(self, pool: list[int], ctx: AcqContext) -> list[int]:
        typ = ctx.typicality
        if typ is None:
            typ = transductive.typicality(ctx.cls, k=20)
        labeled = ctx.labeled or []
        scores = transductive.typiclust_rank_scores(ctx.cls, labeled, pool, typ, seed=0)
        return sorted(pool, key=lambda i: scores[i], reverse=True)
