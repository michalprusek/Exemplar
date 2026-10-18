"""(4) The blend choice on constructed held-out maps.

Each held-out image has foreground pixels that enter the blend a*L_plain + (1-a)*L_bal > 1/2 for a > 0.25 and
background pixels that enter for a > 0.55. Every a in (0.25, 0.55) is then perfect: the 0.2 grid finds 0.4, the
refinement adds 0.3, 0.35, 0.45 and 0.5, and the tie goes to the larger a, 0.5. The one-standard-error rule keeps that
a when the gain over a = 1 is consistent across the held-out images, and falls back to the plain readout (a = 1) when
it is not."""
import numpy as np
import pytest
import torch

from exemplar.readout import choose_blend

LP_FG, LB_FG = 1.1, 0.3                    # crosses 1/2 at a = (0.5 - 0.3) / (1.1 - 0.3) = 0.25
LB_LATE = 0.2
LP_LATE = 0.2 + 0.3 / 0.55                 # crosses 1/2 at a = 0.55


def _image(n_fg_early, n_fg_late, n_fp, side=10):
    """One held-out image: (L_plain, L_bal, mask). Pixels in raster order: early foreground, late foreground,
    late background (false positives for a > 0.55), the rest background that never enters."""
    n = side * side
    lp, lb, gt = torch.zeros(n), torch.zeros(n), torch.zeros(n, dtype=torch.bool)
    i = 0
    lp[i:i + n_fg_early], lb[i:i + n_fg_early], gt[i:i + n_fg_early] = LP_FG, LB_FG, True
    i += n_fg_early
    lp[i:i + n_fg_late], lb[i:i + n_fg_late], gt[i:i + n_fg_late] = LP_LATE, LB_LATE, True
    i += n_fg_late
    lp[i:i + n_fp], lb[i:i + n_fp] = LP_LATE, LB_LATE
    return lp.view(side, side), lb.view(side, side), gt.view(side, side)


def _choose(imgs):
    Mp = [(lp, gt, None) for lp, lb, gt in imgs]
    Mb = [(lb, gt, None) for lp, lb, gt in imgs]
    pm = dict(lam=0.1, w=np.array([1.0, 2.0, 3.0], np.float32))
    bm = dict(lam=0.2, w=np.array([3.0, 2.0, -1.0], np.float32))
    return choose_blend(Mp, Mb, "fg_iou", pm, bm)


def test_blend_picks_largest_perfect_a_and_keeps_it():
    imgs = [_image(32, 0, n_fp) for n_fp in (4, 8, 16)]     # IoU at a = 1: 32/36, 32/40, 32/48
    blend, chosen = _choose(imgs)
    assert blend["a"] == pytest.approx(0.5) and blend["variant"] == "ba050"
    np.testing.assert_allclose(blend["per_img"], 1.0)
    np.testing.assert_allclose(blend["per_img_half"], [32 / 36, 32 / 40, 32 / 48], rtol=1e-12)
    assert chosen["a"] == pytest.approx(0.5) and chosen["gain"] > chosen["se"]
    np.testing.assert_array_equal(chosen["w"], np.array([2.0, 2.0, 1.0], np.float32))   # 0.5 w_plain + 0.5 w_bal
    assert chosen["lam"] == 0.1 and chosen["lam_bal"] == 0.2 and chosen["tau"] == 0.5


def test_one_standard_error_rule_keeps_the_plain_readout():
    # A: a = 0.5 gains +0.3 over a = 1 (28 of 40 at a = 1); B: a = 0.5 loses 4 late foreground pixels (32/36 vs 1)
    imgs = [_image(28, 0, 12), _image(32, 4, 0), _image(28, 0, 12), _image(32, 4, 0)]
    blend, chosen = _choose(imgs)
    assert blend["a"] == pytest.approx(0.5)                  # the best mean held-out IoU ...
    assert chosen["a"] == 1.0 and chosen["variant"] == "plain"   # ... does not beat a = 1 by one standard error
    assert 0 < chosen["gain"] <= chosen["se"]
    np.testing.assert_array_equal(chosen["w"], np.array([1.0, 2.0, 3.0], np.float32))   # the plain weights
    np.testing.assert_allclose(chosen["per_img"], [0.7, 1.0, 0.7, 1.0], rtol=1e-12)
