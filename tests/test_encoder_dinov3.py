import numpy as np
import pytest

from active_segmenter.encoder.dinov3 import _to_rgb01, _feather


def test_to_rgb01_from_grayscale():
    g = np.array([[0, 255], [128, 64]], np.uint8)
    out = _to_rgb01(g)
    assert out.shape == (2, 2, 3)
    assert abs(out.min()) < 1e-6 and abs(out.max() - 1.0) < 1e-6


def test_to_rgb01_drops_alpha():
    rgba = np.zeros((3, 3, 4), np.uint8)
    assert _to_rgb01(rgba).shape[2] == 3


def test_feather_peaks_center():
    w = _feather(5, 5)
    assert w[2, 2] == w.max()
    assert (w >= 0.05).all()


@pytest.mark.gpu
def test_dinov3_extract_shape():
    """Runs on tulen only: real DINOv3-L, assert grid shape + unit-norm rows."""
    from active_segmenter.config import EncoderConfig
    from active_segmenter.encoder.dinov3 import Dinov3Encoder

    cfg = EncoderConfig(resolution=672)
    enc = Dinov3Encoder(cfg, device="cuda")
    img = (np.random.RandomState(0).rand(64, 64) * 255).astype(np.uint8)
    feat = enc.extract(img)
    assert feat.shape == (42, 42, 1024)
    norms = np.linalg.norm(feat.reshape(-1, 1024), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
    cls = enc.extract_cls(img)
    assert cls.shape == (1024,)
