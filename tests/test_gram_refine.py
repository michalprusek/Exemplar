import numpy as np
import pytest

torch = pytest.importorskip("torch")

from active_segmenter.encoder.dinov3 import _gram_refine


def test_gram_refine_keeps_shape_and_unit_norm():
    feat = torch.nn.functional.normalize(torch.randn(4, 5, 8), dim=2)
    out = _gram_refine(feat)
    assert tuple(out.shape) == (4, 5, 8)
    norms = out.reshape(-1, 8).norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_gram_refine_pulls_similar_patches_together():
    # two clusters: patches 0-1 near +e0, patches 2-3 near +e1; refine should tighten clusters
    base = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    feat = torch.nn.functional.normalize(base, dim=1).reshape(2, 2, 2)
    out = _gram_refine(feat, temp=0.05).reshape(4, 2)
    # after refinement, within-cluster cosine >= before (patches pulled together)
    before = float(torch.nn.functional.normalize(base, dim=1)[0] @
                   torch.nn.functional.normalize(base, dim=1)[1])
    after = float(out[0] @ out[1])
    assert after >= before - 1e-4
