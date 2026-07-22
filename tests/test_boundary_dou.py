"""Boundary DoU loss (Sun et al., MICCAI 2023) — binary adaptation for the adaptive-loss menu.

Verified against the reference impl (github sunfan-bvb/BoundaryDoULoss): alpha = 1 − 2C/S clamped to
≤0.8 (C = boundary-pixel count, S = target area); loss = (z²+y²−2·inter)/(z²+y²−(1+alpha)·inter).
Motivated by the monuseg fg diagnostic: precision 0.68 vs recall 0.89 (fg over-prediction / bleed).
"""
import torch

from active_segmenter.segment.head_fusion_backend import _boundary_dou


def _square_target(H=48, lo=12, hi=36):
    t = torch.zeros(1, 1, H, H)
    t[..., lo:hi, lo:hi] = 1.0
    return t


def test_boundary_dou_near_zero_on_perfect_prediction():
    t = _square_target()
    assert _boundary_dou(t * 20 - 10, t).item() < 0.05          # near-perfect logits → ~0 loss


def test_boundary_dou_larger_on_worse_prediction():
    t = _square_target()
    good = _boundary_dou(t * 20 - 10, t).item()
    empty = _boundary_dou(torch.full_like(t, -10.0), t).item()  # predicts all background
    assert empty > good


def test_boundary_dou_penalizes_over_prediction():
    """The monuseg failure mode: over-prediction (fg bleed) must cost more than a tight prediction."""
    t = _square_target()
    tight = _boundary_dou(t * 20 - 10, t).item()
    bled = torch.zeros_like(t); bled[..., 8:40, 8:40] = 1.0      # fg bled well past the target
    over = _boundary_dou(bled * 20 - 10, t).item()
    assert over > tight


def test_boundary_dou_finite_and_scalar():
    t = _square_target()
    v = _boundary_dou(torch.randn(1, 1, 48, 48), t)
    assert v.ndim == 0 and torch.isfinite(v)


def test_bdou_weight_gated_off_on_thin_multicomponent():
    """Review fix: the density gate alone fires on thin filaments (many components); the (1−thinness)
    co-gate must damp bdou on thin-and-dense (microtubules/vessels) while keeping it on compact-and-dense
    (touching nuclei) — protecting the no-regression bar on thin controls best_v2 already wins."""
    from active_segmenter.segment.head_fusion_backend import _ADAPTIVE_LOSS_CFG, _adaptive_weights
    compact_dense = dict(thinness=0.05, fg_frac=0.2, complexity=0.1, inst_density=8, mean_radius=6.0)
    thin_dense = dict(thinness=0.85, fg_frac=0.05, complexity=2.0, inst_density=8, mean_radius=2.0)
    w_c = _adaptive_weights(compact_dense, True, _ADAPTIVE_LOSS_CFG)["bdou"]
    w_t = _adaptive_weights(thin_dense, True, _ADAPTIVE_LOSS_CFG)["bdou"]
    assert w_c > 0.2                      # dense compact touching nuclei → bdou fires
    assert w_t < 0.15 and w_c > 3 * w_t   # dense THIN filaments → strongly damped
