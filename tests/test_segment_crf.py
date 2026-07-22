import numpy as np
import pytest

from active_segmenter.segment.crf import guided_upsample, refine_probability


def test_guided_upsample_sharpens_to_edge():
    # image: left half dark, right half bright; blurry prob straddling the edge as a ramp
    img = np.zeros((32, 32), np.float32)
    img[:, 16:] = 1.0
    prob = np.linspace(0, 1, 32)[None, :].repeat(32, 0).astype(np.float32)  # smooth ramp
    out = guided_upsample(img, prob, radius=6, eps=1e-4)
    # sharpened prob on the dark (left) side should be pulled DOWN toward the image step
    assert out[:, :14].mean() < prob[:, :14].mean() + 0.1
    assert out.shape == prob.shape


def test_refine_probability_returns_same_shape():
    # refine_probability defaults to the dense CRF and REFUSES to fall back to the guided filter,
    # because a silent fallback would undersell the INSID3 baseline that uses it. That refusal is
    # correct behaviour, so without the optional wheel this is a skip, not a failure.
    pytest.importorskip("pydensecrf", reason="optional dense-CRF dependency (see requirements)")
    img = np.random.default_rng(0).random((24, 24)).astype(np.float32)
    prob = np.random.default_rng(1).random((24, 24)).astype(np.float32)
    out = refine_probability(img, prob, n_iters=3)
    assert out.shape == (24, 24) and out.dtype == np.float32
