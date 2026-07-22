"""Compose any SegmenterBackend with a Refiner.

``predict()`` proposals are passed through the refiner; ``fit``/``foreground``/``score_map``
delegate to the inner backend. This lets the trained head be SAM-refined and raced head-to-head
against INSID3 on instance-AP without editing every backend.
"""
from __future__ import annotations


class RefiningBackend:
    def __init__(self, inner, refiner):
        self.inner = inner
        self.refiner = refiner

    def fit(self, support):
        return self.inner.fit(support)

    def foreground(self, image, feat_grid, class_id: int = 1):
        return self.inner.foreground(image, feat_grid, class_id)

    def score_map(self, image, feat_grid, class_id: int = 1):
        return self.inner.score_map(image, feat_grid, class_id)

    def predict(self, image, feat_grid, class_id: int = 1):
        masks = self.inner.predict(image, feat_grid, class_id)
        return self.refiner.refine(image, masks, feat_grid=feat_grid)
