"""Unit tests for the multi-prototype support→query correspondence (Lever 1).

Parity: k=j=1 (mean centroids) must reduce EXACTLY to the current single-prototype corr channel.
Discrimination: on a bimodal foreground the max-pooled multi-prototype separates better than the
diluted single mean. All CPU, no DINOv3 needed.
"""
import numpy as np

from active_segmenter.segment import multiproto as mp


def _grid(G=8, D=16, seed=0):
    return np.random.default_rng(seed).standard_normal((G, G, D)).astype(np.float32)


def test_kmeans_protos_unit_norm_and_count():
    pts = np.random.default_rng(0).standard_normal((50, 16)).astype(np.float32)
    c = mp.kmeans_protos(pts, k=4, seed=0)
    assert c.shape == (4, 16)
    assert np.allclose(np.linalg.norm(c, axis=1), 1.0, atol=1e-5)


def test_kmeans_protos_fewer_points_than_k():
    pts = np.random.default_rng(0).standard_normal((3, 16)).astype(np.float32)
    c = mp.kmeans_protos(pts, k=8, seed=0)
    assert 1 <= c.shape[0] <= 3 and c.shape[1] == 16
    assert np.allclose(np.linalg.norm(c, axis=1), 1.0, atol=1e-5)


def test_multiproto_k1_equals_single_prototype():
    """PARITY: k=j=1 with mean centroids == the current single-prototype corr channel."""
    g = _grid()
    fg = np.random.default_rng(1).standard_normal((20, 16)).astype(np.float32)
    bg = np.random.default_rng(2).standard_normal((30, 16)).astype(np.float32)
    single = mp.single_proto_corr(g, fg, bg)
    fg1 = mp.kmeans_protos(fg, k=1)      # one centroid = the (normalized) mean direction
    bg1 = mp.kmeans_protos(bg, k=1)
    multi = mp.multiproto_corr(g, fg1, bg1)
    assert np.allclose(single, multi, atol=1e-4)


def test_multiproto_beats_single_on_bimodal_fg():
    """A bimodal fg (two appearance clusters) separates better with multi than the single mean."""
    rng = np.random.default_rng(0)
    D = 16
    a = np.zeros(D, np.float32); a[0] = 1.0
    b = np.zeros(D, np.float32); b[1] = 1.0            # two orthogonal fg modes
    fg = np.concatenate([a + 0.01 * rng.standard_normal((25, D)),
                         b + 0.01 * rng.standard_normal((25, D))]).astype(np.float32)
    bg = (0.01 * rng.standard_normal((50, D))).astype(np.float32)  # near-origin bg
    qa = (a / np.linalg.norm(a)).astype(np.float32)               # a query patch from mode a
    single = mp.single_proto_corr(qa.reshape(1, 1, D), fg, bg)[0, 0]
    multi = mp.multiproto_corr(qa.reshape(1, 1, D),
                               mp.kmeans_protos(fg, 2), mp.kmeans_protos(bg, 2))[0, 0]
    assert multi > single    # the mean sits between modes and dilutes; max-pool locks onto mode a
