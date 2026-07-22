"""Refine stage: turn coarse per-instance proposal masks into crisp native-res
masks. Pluggable so the pipeline can run fully in-context (`identity`) or with a
promptable SAM refiner (`sam`)."""
from __future__ import annotations

from active_segmenter.config import RefineConfig
from active_segmenter.refine.base import Refiner
from active_segmenter.refine.identity import IdentityRefiner


def build_refiner(cfg: RefineConfig, device: str) -> Refiner:
    if cfg.kind == "identity":
        return IdentityRefiner(cfg)
    if cfg.kind == "sam":
        from active_segmenter.refine.sam import SamRefiner

        return SamRefiner(cfg, device)
    raise ValueError(f"unknown refine kind: {cfg.kind}")


__all__ = ["build_refiner", "Refiner", "IdentityRefiner"]
