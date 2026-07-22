"""Lever 3 — precision-favouring Tversky (α>β) for dense-compact instances.

The monuseg diagnostic showed diffuse INTERIOR over-prediction (precision 0.68 vs recall 0.89); a Tversky
with α>β penalises false positives harder, so over-prediction (bleed) must cost MORE than under it. The term
is gated dense-compact and behind the `prec_loss` flag → default-off parity.
"""
import numpy as np
import torch

from active_segmenter.segment.base import LabeledExample
from active_segmenter.segment.head_fusion_backend import HeadFusionBackend, _tversky


def test_precision_tversky_penalizes_over_prediction_more_than_recall_tversky():
    t = torch.zeros(1, 1, 32, 32); t[..., 8:24, 8:24] = 1.0
    over = torch.zeros(1, 1, 32, 32); over[..., 4:28, 4:28] = 1.0        # fg bled past the target (FP-heavy)
    logit = over * 20 - 10
    prec = _tversky(logit, t, alpha=0.7, beta=0.3).item()                # precision-favouring (penalise FP)
    rec = _tversky(logit, t, alpha=0.3, beta=0.7).item()                 # recall-favouring (current default)
    assert prec > rec                                                    # over-prediction costs more under α>β


def _multi_blob(seed, n, H=128, D=16, G=8):
    yy, xx = np.mgrid[0:H, 0:H]
    centers = [(32, 32), (32, 96), (96, 32), (96, 96)]
    lab = np.zeros((H, H), int)
    for i, (cy, cx) in enumerate(centers[:n], 1):
        lab[(yy - cy) ** 2 + (xx - cx) ** 2 < 18 ** 2] = i
    rng = np.random.default_rng(seed)
    feat = rng.standard_normal((G, G, D)).astype(np.float32)
    feat /= (np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-6)
    return LabeledExample(image=((lab > 0) * 200).astype(np.float32), feat_grid=feat, label_map=lab)


def test_prec_loss_fit_runs_on_dense_instances():
    be = HeadFusionBackend(device="cpu", epochs=2, proj_dim=8, adaptive_loss=True,
                           prec_loss=True, amp=False)
    be.fit([_multi_blob(1, n=3), _multi_blob(2, n=4)])                    # varying id sets → binary/instances
    fg = be.foreground(_multi_blob(3, n=4).image, _multi_blob(3, n=4).feat_grid)
    assert fg.shape == (128, 128) and fg.dtype == bool
