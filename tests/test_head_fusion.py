"""CPU smoke test for the classical-fusion head backend (DINOv3 semantics ⊕ classical priors).

Fake DINOv3 features (small grid) + a synthetic native blob; checks fit/foreground/grad_embedding
run and produce a native-res segmentation with the right shape. No real DINOv3 needed.
"""
import numpy as np

from active_segmenter.segment.base import LabeledExample
from active_segmenter.segment.head_fusion_backend import HeadFusionBackend


def _example(H=256, D=16, G=8):
    yy, xx = np.mgrid[0:H, 0:H]
    m = (yy - 128) ** 2 + (xx - 128) ** 2 < 48 ** 2
    img = (m * 200 + np.random.default_rng(0).integers(0, 20, (H, H))).astype(np.float32)
    feat = np.random.default_rng(1).standard_normal((G, G, D)).astype(np.float32)
    return LabeledExample(image=img, feat_grid=feat, label_map=m.astype(int))


def test_head_fusion_fits_and_predicts_native():
    ex = _example()
    be = HeadFusionBackend(device="cpu", epochs=6, proj_dim=8, amp=False)  # bf16 autocast is CUDA-only
    be.fit([ex, ex])
    fg = be.foreground(ex.image, ex.feat_grid)
    assert fg.shape == (256, 256) and fg.dtype == bool
    # grad_embedding (EGL hook) returns a finite non-trivial vector over the fused penultimate
    g = be.grad_embedding(ex.image, ex.feat_grid)
    assert g.shape[0] == be.proj_dim + be._n_classical + 1
    assert np.isfinite(g).all()


def _blob_example(seed, H=128, D=16, G=8):
    yy, xx = np.mgrid[0:H, 0:H]
    m = (yy - 64) ** 2 + (xx - 64) ** 2 < 34 ** 2
    rng = np.random.default_rng(seed)
    feat = rng.standard_normal((G, G, D)).astype(np.float32)
    feat /= (np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-6)      # unit per patch (like DINOv3)
    return LabeledExample(image=(m * 200).astype(np.float32), feat_grid=feat, label_map=m.astype(int))


def test_backend_builds_multiprotos_when_nproto_gt1():
    be = HeadFusionBackend(device="cpu", epochs=2, proj_dim=8, corr_prior=True, n_proto=4, amp=False)
    be.fit([_blob_example(1), _blob_example(2), _blob_example(3), _blob_example(4)])
    assert be._fg_protos is not None and be._fg_protos.shape[1] == 16   # [k, D] centroids
    assert be._bg_protos is not None and be._bg_protos.shape[1] == 16
    assert be._fg_proto is not None                                     # mean still built (FiLM uses it)


def test_backend_nproto1_leaves_multiprotos_none():
    """PARITY: default n_proto=1 → no centroid stacks (single-prototype path, == best_v2)."""
    be = HeadFusionBackend(device="cpu", epochs=2, proj_dim=8, corr_prior=True, amp=False)
    be.fit([_blob_example(1), _blob_example(2)])
    assert be.n_proto == 1 and be._fg_protos is None and be._bg_protos is None


def _multi_blob_example(seed, n, H=128, D=16, G=8):
    """`n` round blobs with distinct instance ids → varying id sets across images make _detect_n_classes
    read them as INSTANCES (binary), exercising the dense-instance branch where Boundary DoU fires."""
    yy, xx = np.mgrid[0:H, 0:H]
    centers = [(32, 32), (32, 96), (96, 32), (96, 96)]
    lab = np.zeros((H, H), int)
    for i, (cy, cx) in enumerate(centers[:n], 1):
        lab[(yy - cy) ** 2 + (xx - cx) ** 2 < 18 ** 2] = i
    rng = np.random.default_rng(seed)
    feat = rng.standard_normal((G, G, D)).astype(np.float32)
    feat /= (np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-6)
    return LabeledExample(image=((lab > 0) * 200).astype(np.float32), feat_grid=feat, label_map=lab)


def test_boundary_dou_fit_runs_on_dense_instances():
    """The bdou term fires only under adaptive_loss with high instance-density; fit + predict must run."""
    be = HeadFusionBackend(device="cpu", epochs=2, proj_dim=8, adaptive_loss=True,
                           boundary_dou=True, amp=False)
    be.fit([_multi_blob_example(1, n=3), _multi_blob_example(2, n=4)])   # varying id sets → binary/instances
    ex = _multi_blob_example(3, n=4)
    fg = be.foreground(ex.image, ex.feat_grid)
    assert fg.shape == (128, 128) and fg.dtype == bool


def test_head_corr_maxpool_parity_and_shape():
    """PARITY: max-pool corr with k=j=1 (single-row stacks) == the single-prototype corr channel,
    and multi-prototype keeps the SAME +1 channel width (1×1 classifier / EGL unchanged)."""
    import torch
    from active_segmenter.segment.head_fusion import DINOHeadFusion
    G, D, Nc = 8, 16, 5
    h = DINOHeadFusion(in_dim=D, hidden=32, proj_dim=8, n_classical=Nc, corr_prior=True)
    fg = torch.randn(D); fg /= fg.norm()
    bg = torch.randn(D); bg /= bg.norm()
    feat = torch.randn(1, D, G, G); cls = torch.randn(1, Nc, 32, 32)
    h.set_prototypes(fg, bg)                                             # single-prototype path
    z_single = h._penultimate(feat, cls, (32, 32))
    h.set_prototypes(fg, bg, fg_protos=fg.view(1, D), bg_protos=bg.view(1, D))   # k=j=1 max-pool
    z_k1 = h._penultimate(feat, cls, (32, 32))
    assert torch.allclose(z_single, z_k1, atol=1e-4)                     # parity
    h.set_prototypes(fg, bg, fg_protos=torch.randn(4, D), bg_protos=torch.randn(3, D))  # k=4,j=3
    z_multi = h._penultimate(feat, cls, (32, 32))
    assert z_multi.shape == z_single.shape                              # same width (one corr channel)
