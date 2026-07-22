"""Composite convergence / stopping signal.

No single signal is reliable, so we combine several: validation IoU plateau over a
window, falling human-correction rate, and flattening acquisition scores. Reported
to the user as a live "model is converging" readout.
"""
from __future__ import annotations

import numpy as np


def _plateau(values: list[float], eps: float, window: int) -> bool:
    if len(values) < window:
        return False
    w = values[-window:]
    return (max(w) - min(w)) <= eps


def _falling(values: list[float], window: int) -> bool:
    if len(values) < window:
        return False
    w = values[-window:]
    return w[-1] <= w[0]


def composite(history: list[dict], iou_eps: float = 0.01, window: int = 3) -> dict:
    """``history``: list of per-round dicts with keys ``iou``, ``correction_rate``,
    ``acq_score``. Returns the individual signals + an overall ``converged`` flag."""
    ious = [h["iou"] for h in history]
    corr = [h.get("correction_rate", 1.0) for h in history]
    acq = [h.get("acq_score", 1.0) for h in history]
    iou_plateau = _plateau(ious, iou_eps, window)
    corr_falling = _falling(corr, window)
    acq_flat = _plateau(acq, max(iou_eps, 0.05), window) or _falling(acq, window)
    converged = bool(iou_plateau and corr_falling and acq_flat)
    return {
        "converged": converged,
        "iou_plateau": iou_plateau,
        "correction_falling": corr_falling,
        "acq_flattening": acq_flat,
        "rounds": len(history),
    }
