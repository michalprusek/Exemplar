"""One instance exemplar in the memory bank.

The bank is non-parametric: an exemplar stores the masked DINOv3 patch features of
a single corrected instance (its foreground signature), plus background patches
from the same image and the instance polygon. No gradients, ever.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class InstanceExemplar:
    class_id: int
    instance_id: int
    fg_feats: np.ndarray            # [N_fg, D]
    bg_feats: np.ndarray            # [M_bg, D]
    polygon: Optional[np.ndarray]   # [K, 2] xy, or None
    round: int

    def centroid(self) -> np.ndarray:
        """Mean fg direction — the exemplar's single-vector summary for curation."""
        c = self.fg_feats.mean(0)
        return c / (np.linalg.norm(c) + 1e-8)
