"""CPU test for the RefiningBackend wrapper (spec 2026-07-12 refine-stage)."""
import numpy as np

from active_segmenter.segment.refining_backend import RefiningBackend
from active_segmenter.types import InstanceMask


class _FakeInner:
    def fit(self, s):
        self.fitted = True

    def foreground(self, im, fg, class_id=1):
        return np.ones((4, 4), bool)

    def score_map(self, im, fg, class_id=1):
        return np.zeros((4, 4), np.float32)

    def predict(self, im, fg, class_id=1):
        return [InstanceMask(mask=np.ones((4, 4), bool), points=None, class_id=1, instance_id=0)]


class _FakeRefiner:
    def refine(self, im, masks, feat_grid=None):
        return [InstanceMask(mask=m.mask, points=None, class_id=m.class_id,
                             instance_id=m.instance_id, score=0.9) for m in masks]


def test_refining_backend_delegates_and_refines():
    be = RefiningBackend(_FakeInner(), _FakeRefiner())
    be.fit([])
    assert be.inner.fitted
    assert be.foreground(np.zeros((4, 4)), None).all()  # delegated
    out = be.predict(np.zeros((4, 4)), None)
    assert len(out) == 1 and out[0].score == 0.9  # refined
