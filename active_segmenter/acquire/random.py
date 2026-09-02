"""Random control arm — mandatory. AL gains often collapse to random once SSL +
augmentation are in play (nnActive, "Parting with Illusions"); every AL run is
benchmarked against this arm."""
from __future__ import annotations

from active_segmenter.acquire.base import AcqContext


class RandomAcq:
    def rank(self, pool: list[int], ctx: AcqContext) -> list[int]:
        order = ctx.rng.permutation(len(pool))
        return [pool[i] for i in order]
