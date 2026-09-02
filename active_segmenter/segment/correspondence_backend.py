"""Baseline backend: the current propose->cluster->(optional SAM) path behind the
SegmenterBackend interface. ``foreground`` reproduces ``corr.score_map > 0`` exactly so the
AL-testbed numbers are unchanged — this is the regression anchor for the race."""
from __future__ import annotations

import numpy as np

from active_segmenter.config import ClusterConfig, MatchConfig
from active_segmenter.membank.bank import MemoryBank
from active_segmenter.propose import correspondence as corr
from active_segmenter.propose import instances as inst
from active_segmenter.segment.base import LabeledExample, foreground_from_score
from active_segmenter.types import InstanceMask


class CorrespondenceBackend:
    # `fit` REBUILDS `self.bank` from scratch each draw and nothing else is derived from the support.
    stateless_support = True

    def __init__(self, match_cfg: MatchConfig, cluster_cfg: ClusterConfig,
                 refiner=None, device: str | None = None):
        self.mc = match_cfg
        self.cc = cluster_cfg
        self.refiner = refiner
        self.device = device
        self.bank = MemoryBank()

    def fit(self, support: list[LabeledExample]) -> None:
        self.bank = MemoryBank()
        for ex in support:
            # collapse per-instance ids to a single fg class id=1 for the bank
            fg = (np.asarray(ex.label_map) > 0).astype(int)
            self.bank.add_from_annotation(ex.feat_grid, fg, {1: 1} if fg.any() else {}, 0)

    def score_map(self, image, feat_grid, class_id: int = 1) -> np.ndarray:
        return corr.score_map(feat_grid, self.bank, class_id, self.mc, device=self.device)

    def foreground(self, image, feat_grid, class_id: int = 1) -> np.ndarray:
        s = self.score_map(image, feat_grid, class_id)
        return foreground_from_score(s, np.asarray(image).shape[:2])

    def predict(self, image, feat_grid, class_id: int = 1) -> list[InstanceMask]:
        s = self.score_map(image, feat_grid, class_id)
        masks = inst.decompose(s, self.cc, class_id, feat_grid=None)
        masks = inst.upsample_masks(masks, np.asarray(image).shape[:2])
        if self.refiner is not None:
            masks = self.refiner.refine(image, masks, feat_grid=feat_grid)
        return masks
