"""Acquisition protocol + context.

An :class:`Acquisition` ranks the unlabeled pool best-first; the orchestrator takes
the top ``topk_batch``. The context carries the cheap, precomputed signals every
strategy might need: per-image non-parametric uncertainty and per-image CLS
embeddings (for representativeness/diversity), plus a seeded RNG.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class AcqContext:
    uncertainty: np.ndarray                 # [N] per-image uncertainty (0 where unknown)
    cls: np.ndarray                         # [N, D] per-image CLS embeddings (L2-normalised)
    rng: np.random.Generator
    labeled: list | None = None             # indices already annotated (for coverage methods)
    typicality: np.ndarray | None = None    # [N] precomputed density (TypiClust)


class Acquisition(Protocol):
    def rank(self, pool: list[int], ctx: AcqContext) -> list[int]:
        ...
