import numpy as np
import pytest

from active_segmenter.acquire.badge import badge_select, egl_scores


def test_egl_is_gradient_norm():
    emb = {0: [3.0, 4.0], 1: [0.0, 0.0], 2: [1.0, 0.0]}
    s = egl_scores(emb)
    assert abs(s[0] - 5.0) < 1e-6 and s[1] == 0.0 and abs(s[2] - 1.0) < 1e-6


def test_badge_first_pick_is_highest_egl():
    emb = {0: [0.1, 0.0], 1: [5.0, 0.0], 2: [0.2, 0.0]}   # idx 1 has the largest gradient
    picks = badge_select(emb, k=1)
    assert picks == [1]


def test_badge_selects_diverse_gradients():
    # two tight clusters near +x and +y; BADGE should pick one from each, not two from one
    rng = np.random.default_rng(0)
    emb = {}
    for i in range(6):
        emb[i] = [5.0 + rng.normal(0, 0.01), rng.normal(0, 0.01)]        # +x cluster
    for i in range(6, 12):
        emb[i] = [rng.normal(0, 0.01), 5.0 + rng.normal(0, 0.01)]        # +y cluster
    picks = badge_select(emb, k=2, seed=1)
    assert len(picks) == 2
    clusters = {0 if p < 6 else 1 for p in picks}
    assert clusters == {0, 1}                                            # one from each


def test_badge_k_capped():
    emb = {0: [1.0], 1: [2.0]}
    assert sorted(badge_select(emb, k=5)) == [0, 1]


def test_head_grad_embedding_shape_and_zero_when_untrained():
    torch = pytest.importorskip("torch")
    from active_segmenter.segment.head_backend import TrainableHeadBackend
    from active_segmenter.segment.base import LabeledExample

    be = TrainableHeadBackend(device="cpu", hidden=16, epochs=3)
    feat = np.random.default_rng(0).standard_normal((5, 5, 8)).astype(np.float32)
    # untrained -> zero vector of size hidden+1
    assert np.allclose(be.grad_embedding(feat), 0.0)
    assert be.grad_embedding(feat).shape == (16 + 1,)
    # after fit -> finite, generally nonzero, correct size
    lm = np.zeros((20, 20), int); lm[4:12, 4:12] = 1
    be.fit([LabeledExample(np.zeros((20, 20)), feat, lm)])
    g = be.grad_embedding(feat)
    assert g.shape == (16 + 1,) and np.all(np.isfinite(g))
