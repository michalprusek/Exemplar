"""Refiner protocol.

A refiner maps coarse per-instance masks (grid- or native-resolution) to crisp
native-resolution per-instance masks. Each instance stays its own channel — a
refiner must never merge instances into a shared label raster.
"""
from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from active_segmenter.types import InstanceMask


class Refiner(Protocol):
    def refine(
        self,
        image: np.ndarray,
        instance_masks: list[InstanceMask],
        feat_grid: Optional[np.ndarray] = None,
    ) -> list[InstanceMask]:
        ...
