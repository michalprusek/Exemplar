"""Core data types shared across the pipeline.

All types are plain, serialisable dataclasses so the library's state can later be
lifted into the parent platform's store without logic changes. The overlap rule
lives here in spirit: an :class:`InstanceMask` is always one independent binary
mask — instances are never collapsed into a shared label raster.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class ClassLabel:
    id: int
    name: str
    color: str = "#00ff00"


@dataclass
class InstanceMask:
    """One instance. Either a boolean ``mask`` (grid- or native-resolution) or a
    ``points`` polygon (``[K, 2]`` xy). Overlap-safe: each instance owns its mask."""

    mask: Optional[np.ndarray]
    points: Optional[np.ndarray]
    class_id: int
    instance_id: int
    score: float = 1.0


@dataclass
class MemoryBankEntry:
    dataset_id: str
    class_id: int
    instance_id: int
    embedding: np.ndarray       # [N_fg_patches, D]
    exemplar_mask: np.ndarray   # boolean mask of the exemplar instance
    round: int


@dataclass
class ALState:
    round: int
    selected: list[int] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    convergence: dict[str, Any] = field(default_factory=dict)
    bank_size: int = 0

    def to_json(self) -> str:
        return json.dumps(
            {
                "round": self.round,
                "selected": list(self.selected),
                "scores": self.scores,
                "convergence": self.convergence,
                "bank_size": self.bank_size,
            }
        )

    @classmethod
    def from_json(cls, s: str) -> "ALState":
        d = json.loads(s)
        return cls(
            round=d["round"],
            selected=list(d.get("selected", [])),
            scores=d.get("scores", {}),
            convergence=d.get("convergence", {}),
            bank_size=d.get("bank_size", 0),
        )
