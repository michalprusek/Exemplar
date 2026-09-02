"""AL loop state + per-round result records."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RoundResult:
    n_annotated: int
    fg_iou: float
    instance_ap: float
    arm: str = "al"
    converged: bool = False


@dataclass
class ALState:
    round: int = 0
    selected: list[int] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    bank_size: int = 0
