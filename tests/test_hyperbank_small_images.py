"""The classical bank must not crash on images smaller than its own largest filter, and the fix
must not move any number on images where it never crashed.

`ours_k1_bacteria` died in gaussian_blur:

    RuntimeError: Padding size should be less than the corresponding input dimension,
    but got: padding (64, 64) at dimension 3 of input [1, 1, 66, 58]

A first fix touched only gaussian_blur; the SAME reflect-pad defect then killed the rerun in
_box_avg ("padding (75, 75) ... [1, 1, 66, 58]"). The bank reflect-pads by a scale-derived radius
in at least five operators, so the fix is ONE shared `reflect_pad`, used everywhere.

Measured blast radius: `bacteria` holds 168 images whose smallest side ranges 58-2038 px, median
596; exactly TWO are under 65 px -- (66,58) and (62,113). Two images in 168 destroyed the dataset,
on the reviewer-requested set. bbbc010 (all 520 px) and fisbe (>=680 px) are unaffected. This is the
mixed-size folder a deployed tool meets, not an unusual input.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from active_segmenter.segment.hyperbank_bank import (
    HyperBank, gaussian_blur, hessian, reflect_pad)


def test_reflect_pad_is_byte_identical_to_plain_reflect_where_reflect_fits():
    """The decisive no-op guarantee: on any image large enough for reflect, the shared helper must
    return exactly what the old `F.pad(..., mode='reflect')` returned, or the fix would silently
    move every dataset that currently runs. Checked across the pad radii the bank actually uses."""
    torch.manual_seed(0)
    img = torch.rand(1, 1, 200, 200)
    for pad in (1, 3, 8, 32, 64, 199):          # 199 = W-1, the largest reflect can do here
        got = reflect_pad(img, pad)
        ref = F.pad(img, (pad, pad, pad, pad), mode="reflect")
        assert torch.equal(got, ref), f"reflect_pad diverged from plain reflect at pad={pad}"


def test_reflect_pad_keeps_the_full_output_size_on_tiny_images():
    """Output must be input + 2*pad on every axis even when the image is smaller than the pad, or
    the pool/conv that follows would shrink the map and break the equal-size concatenation."""
    img = torch.rand(1, 1, 12, 8)
    for pad in (7, 20, 64):
        out = reflect_pad(img, pad)
        assert out.shape[-2:] == (12 + 2 * pad, 8 + 2 * pad), f"wrong size at pad={pad}"
        assert torch.isfinite(out).all()


@pytest.mark.parametrize("h,w", [(66, 58), (62, 113), (8, 8), (1, 1), (3, 200)])
def test_blur_survives_images_smaller_than_the_kernel(h, w):
    out = gaussian_blur(torch.rand(1, 1, h, w), sigma=16.0)
    assert out.shape == (1, 1, h, w)
    assert torch.isfinite(out).all()


def test_the_full_bank_runs_on_the_two_tiny_bacteria_images():
    """The real defect: the WHOLE bank, not just one operator. Reproduce the exact shapes that
    crashed the run and the rerun, through feature_maps, which is what fit() calls."""
    bank = HyperBank(frangi_sigmas=(1.0, 2.0, 4.0, 8.0, 16.0),
                     sauvola_windows=(7, 15, 51, 151), struct_sigmas=(2.0, 8.0), use_log=True).eval()
    for h, w in [(66, 58), (62, 113)]:
        with torch.no_grad():
            feats = bank.feature_maps(torch.rand(1, 1, h, w))
        assert feats.shape[-2:] == (h, w), f"bank changed the map size on {h}x{w}"
        assert torch.isfinite(feats).all(), f"bank produced non-finite features on {h}x{w}"


def test_the_bank_output_is_unchanged_on_a_normal_sized_image():
    """End-to-end no-op check: on a 256x256 image, where nothing ever clamped, the full bank output
    must be bit-for-bit identical between two runs -- a guard that the shared helper did not perturb
    the common path."""
    bank = HyperBank(frangi_sigmas=(1.0, 2.0, 4.0, 8.0, 16.0),
                     sauvola_windows=(7, 15, 51, 151), struct_sigmas=(2.0, 8.0), use_log=True).eval()
    img = torch.rand(1, 1, 256, 256)
    with torch.no_grad():
        a = bank.feature_maps(img)
        b = bank.feature_maps(img)
    assert torch.equal(a, b)
    assert a.shape[-2:] == (256, 256)


def test_the_frangi_hessian_survives_the_smallest_image():
    ixx, iyy, ixy = hessian(torch.rand(1, 1, 62, 58), sigma=16.0)
    for t in (ixx, iyy, ixy):
        assert t.shape == (1, 1, 62, 58) and torch.isfinite(t).all()
