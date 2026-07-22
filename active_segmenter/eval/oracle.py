"""GT-as-oracle: the simulated human annotator.

The AL loop proposes an image; the oracle reveals its ground-truth annotation for
*that image only* — exactly the "propose -> get its labels -> adapt -> measure"
protocol used to validate active learning without any manual annotation.
"""
from __future__ import annotations

import numpy as np


class GtOracle:
    def __init__(self, dataset):
        # dataset: list[(image, label_map)]
        self.dataset = dataset
        self.revealed: set[int] = set()

    def reveal(self, index: int) -> np.ndarray:
        """Return the per-instance label map for ``index`` and record the reveal."""
        self.revealed.add(index)
        return self.dataset[index][1]

    def image(self, index: int) -> np.ndarray:
        return self.dataset[index][0]

    @property
    def n_revealed(self) -> int:
        return len(self.revealed)
