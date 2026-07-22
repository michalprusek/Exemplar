"""CPU unit tests for training-free feature super-resolution (spec 2026-07-12).

All pure array ops — no DINOv3 / GPU. A synthetic ``forward_grid_fn`` stands in for the encoder.
"""
import numpy as np

from active_segmenter.encoder.superres import jbu_snap, shift_merge


def test_shift_merge_shape_and_factor1_identity():
    img = np.random.rand(64, 64).astype(np.float32)
    fwd = lambda im, res: np.full((res // 16, res // 16, 3), 0.5, np.float32)
    out1 = shift_merge(fwd, img, resolution=64, patch_stride=16, factor=1)
    assert out1.shape == (4, 4, 3)
    np.testing.assert_array_equal(out1, fwd(img, 64))
    out2 = shift_merge(fwd, img, resolution=64, patch_stride=16, factor=2)
    assert out2.shape == (8, 8, 3)


def test_shift_merge_interleaves_subgrids_in_order():
    """Call order is (ky,kx) row-major: (0,0)=0 (0,1)=1 (1,0)=2 (1,1)=3; each lands on its lattice."""
    img = np.zeros((64, 64), np.float32)
    state = {"n": 0}

    def fwd(im, res):
        val = state["n"]
        state["n"] += 1
        return np.full((res // 16, res // 16, 1), val, np.float32)

    out = shift_merge(fwd, img, resolution=64, patch_stride=16, factor=2)
    assert out[0::2, 0::2].mean() == 0
    assert out[0::2, 1::2].mean() == 1
    assert out[1::2, 0::2].mean() == 2
    assert out[1::2, 1::2].mean() == 3


def test_jbu_uniform_guide_smooths_and_unit_norm():
    rng = np.random.default_rng(0)
    feat = rng.standard_normal((8, 8, 4)).astype(np.float32)
    feat /= np.linalg.norm(feat, axis=2, keepdims=True)
    guide = np.ones((32, 32), np.float32)  # no edges -> pure spatial smoothing
    out = jbu_snap(feat, guide, sigma_spatial=1.0, sigma_range=0.5)
    assert out.shape == feat.shape
    np.testing.assert_allclose(np.linalg.norm(out, axis=2), 1.0, atol=1e-5)
    assert np.var(np.diff(out, axis=0)) < np.var(np.diff(feat, axis=0))


def test_jbu_edge_guide_prevents_bleed():
    feat = np.zeros((4, 8, 2), np.float32)
    feat[:, :4] = np.array([1.0, 0.0])
    feat[:, 4:] = np.array([0.0, 1.0])
    guide = np.zeros((4, 8), np.float32)
    guide[:, 4:] = 1.0  # hard edge at col 4
    out = jbu_snap(feat, guide, sigma_spatial=2.0, sigma_range=0.05)
    # a left-side cell stays on the left prototype despite a wide spatial kernel
    assert out[0, 3] @ np.array([1.0, 0.0]) > out[0, 3] @ np.array([0.0, 1.0])


def test_cache_tag_includes_superres_knobs():
    from active_segmenter.config import EncoderConfig
    from active_segmenter.encoder.factory import cache_tag

    base = cache_tag(EncoderConfig())
    assert "sr" not in base and "jbu" not in base
    t2 = cache_tag(EncoderConfig(superres_factor=2))
    t4j = cache_tag(EncoderConfig(superres_factor=4, jbu=True))
    assert t2.endswith("-sr2")
    assert "-sr4" in t4j and t4j.endswith("-jbu")
    assert t2 != base and t4j != t2  # never collide on disk


def test_extract_superres_uses_shift_merge(monkeypatch):
    from active_segmenter.config import EncoderConfig
    from active_segmenter.encoder import dinov3 as m

    enc = m.Dinov3Encoder.__new__(m.Dinov3Encoder)  # bypass model load
    enc.cfg = EncoderConfig(resolution=64, superres_factor=2, tile=False)
    enc._forward_grid = lambda image, res: np.full((res // 16, res // 16, 3), 0.5, np.float32)
    out = m.Dinov3Encoder.extract(enc, np.zeros((64, 64), np.float32))
    assert out.shape == (8, 8, 3)  # 4x4 base -> 8x8 at factor 2
