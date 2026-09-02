import numpy as np
import pytest

from active_segmenter.config import MatchConfig
from active_segmenter.segment.base import LabeledExample
from active_segmenter.segment.insid3_backend import Insid3FrozenBackend


def _sup(seed=0):
    rng = np.random.default_rng(seed)
    feat = rng.standard_normal((6, 6, 8)).astype(np.float32)
    lm = np.zeros((24, 24), int)
    lm[4:12, 4:12] = 1
    return [LabeledExample(np.zeros((24, 24, 3), np.uint8), feat, lm)]


def test_insid3_foreground_native_bool():
    pytest.importorskip("pydensecrf", reason="INSID3 predict refines with the dense CRF")
    be = Insid3FrozenBackend(MatchConfig(topk=5, bidirectional=False), device="cpu")
    be.fit(_sup())
    q = np.random.default_rng(1).standard_normal((6, 6, 8)).astype(np.float32)
    img = np.zeros((24, 24, 3), np.uint8)
    fg = be.foreground(img, q)
    assert fg.shape == (24, 24) and fg.dtype == bool


def test_insid3_predict_overlap_safe():
    pytest.importorskip("pydensecrf", reason="INSID3 predict refines with the dense CRF")
    be = Insid3FrozenBackend(MatchConfig(topk=5, bidirectional=False), device="cpu")
    be.fit(_sup())
    q = np.random.default_rng(2).standard_normal((6, 6, 8)).astype(np.float32)
    insts = be.predict(np.zeros((24, 24, 3), np.uint8), q)
    assert isinstance(insts, list)
    for m in insts:
        assert m.mask.dtype == bool and m.mask.shape == (24, 24)
