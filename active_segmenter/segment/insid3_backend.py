"""INSID3-style frozen backend: correspondence at high resolution + CRF/guided-filter
boundary recovery -> native-res fg. Training-free; the ``hi_res`` knob (features supplied
at high resolution by the harness) and the edge-aware refine are the levers for small
objects. Reuses the memory bank + ``corr.score_map``; the encoder is expected to have
produced ``feat_grid`` at the desired resolution.
"""
from __future__ import annotations

import numpy as np

from active_segmenter.config import ClusterConfig, MatchConfig
from active_segmenter.membank.bank import MemoryBank
from active_segmenter.propose import correspondence as corr
from active_segmenter.propose import instances as inst
from active_segmenter.segment.crf import refine_probability
from active_segmenter.types import InstanceMask


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float32)))


class Insid3FrozenBackend:
    # `fit` REBUILDS `self.bank` as a fresh MemoryBank before adding the draw, so no exemplar from a
    # previous support set survives; nothing else is derived from the draw.
    stateless_support = True

    def __init__(self, match_cfg: MatchConfig, device: str | None = None, hi_res: int = 1024,
                 cluster_cfg: ClusterConfig | None = None):
        self.mc = match_cfg
        self.device = device
        self.hi_res = hi_res
        self.cc = cluster_cfg or ClusterConfig()
        self.bank = MemoryBank()

    def fit(self, support) -> None:
        self.bank = MemoryBank()
        for ex in support:
            fg = (np.asarray(ex.label_map) > 0).astype(int)
            self.bank.add_from_annotation(ex.feat_grid, fg, {1: 1} if fg.any() else {}, 0)

    def score_map(self, image, feat_grid, class_id: int = 1) -> np.ndarray:
        return corr.score_map(feat_grid, self.bank, class_id, self.mc, device=self.device)

    def foreground(self, image, feat_grid, class_id: int = 1) -> np.ndarray:
        from skimage.transform import resize

        s = self.score_map(image, feat_grid, class_id)
        hw = np.asarray(image).shape[:2]
        prob = resize(_sigmoid(s), tuple(hw), order=1, mode="edge", anti_aliasing=True)
        refined = refine_probability(image, prob.astype(np.float32))
        return refined > 0.5

    def predict(self, image, feat_grid, class_id: int = 1) -> list[InstanceMask]:
        # decompose on the coarse grid, then keep instances whose native mask survives CRF
        s = self.score_map(image, feat_grid, class_id)
        masks = inst.upsample_masks(inst.decompose(s, self.cc, class_id),
                                    np.asarray(image).shape[:2])
        fg = self.foreground(image, feat_grid, class_id)
        out = []
        for m in masks:
            mm = np.logical_and(m.mask, fg)
            if mm.any():
                out.append(InstanceMask(mask=mm, points=None, class_id=class_id,
                                        instance_id=m.instance_id, score=m.score))
        return out
