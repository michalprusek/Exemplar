"""Native-resolution ConvNeXt yields NON-SQUARE feature grids [G0, G1] with G0 != G1.
The memory bank and the trainable head must handle that (they used to assume [g, g])."""
import numpy as np
import pytest

from active_segmenter.config import MatchConfig, ClusterConfig
from active_segmenter.membank.bank import MemoryBank
from active_segmenter.propose import correspondence as corr
from active_segmenter.segment.base import LabeledExample


def test_bank_handles_nonsquare_grid():
    feat = np.random.default_rng(0).standard_normal((5, 8, 6)).astype(np.float32)  # [G0=5, G1=8, D]
    label = np.zeros((40, 64), int)
    label[8:24, 16:48] = 1
    bank = MemoryBank()
    bank.add_from_annotation(feat, label, {1: 1}, 0)     # must not raise on non-square
    assert bank.fg(1).shape[1] == 6                      # D preserved
    # correspondence over a non-square query grid works and keeps the shape
    q = np.random.default_rng(1).standard_normal((5, 8, 6)).astype(np.float32)
    s = corr.score_map(q, bank, 1, MatchConfig(topk=3, bidirectional=False), device="cpu")
    assert s.shape == (5, 8)


def test_head_backend_nonsquare_grid():
    torch = pytest.importorskip("torch")
    from active_segmenter.segment.head_backend import TrainableHeadBackend

    rng = np.random.default_rng(0)
    # two support examples with DIFFERENT non-square grid sizes (as native mode produces)
    sup = []
    for gh, gw in [(5, 8), (6, 4)]:
        feat = rng.standard_normal((gh, gw, 6)).astype(np.float32)
        lm = np.zeros((40, 64), int)
        lm[8:24, 16:48] = 1
        sup.append(LabeledExample(np.zeros((40, 64)), feat, lm))
    be = TrainableHeadBackend(device="cpu", epochs=5, cluster_cfg=ClusterConfig(min_patches=1))
    be.fit(sup)                                          # must not raise (no stacking)
    fg = be.foreground(np.zeros((40, 64)), sup[0].feat_grid)
    assert fg.shape == (40, 64) and fg.dtype == bool
