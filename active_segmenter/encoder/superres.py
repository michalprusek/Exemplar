"""Training-free feature super-resolution for the frozen DINOv3 grid.

Two model-agnostic operators over a ``forward_grid_fn`` that maps an image to an
L2-normalised ``[G, G, D]`` patch grid (``G = resolution // patch_stride``):

- :func:`shift_merge` densifies the grid ``factor×`` per axis by interleaving the grids from
  ``factor²`` sub-patch-shifted forward passes. Training-free, ``factor²`` passes — far cheaper
  than an exhaustive tile sweep — and it densifies small images too, fixing DINOv3-ViT's coarse
  patch-16 grid (its one documented loss vs ConvNeXt).
- :func:`jbu_snap` edge-snaps the (denser) grid with a parameter-free joint bilateral filter — a
  feature-level stand-in for INSID3's post-hoc CRF, applied *before* matching.

Both are pure array ops (numpy), so they unit-test on CPU with a synthetic ``forward_grid_fn``
(no DINOv3 needed). The backbone stays frozen; nothing here trains.
"""
from __future__ import annotations

import numpy as np


def shift_merge(forward_grid_fn, image, resolution, patch_stride, factor):
    """Interleave ``factor²`` sub-patch-shifted forward passes into a ``factor×`` finer grid.

    The shift for interleave cell ``(ky, kx)`` is ``patch_stride·k/factor`` px in *model-input*
    space, applied to the raw image by :func:`numpy.roll` scaled to the image's own size (the
    encoder resizes to ``resolution`` internally). ``factor == 1`` returns exactly the base grid.

    Args:
        forward_grid_fn: ``(image, resolution) -> [G, G, D]`` unit-normed patch grid.
        image: raw ``H×W`` or ``H×W×C`` array.
        resolution: encoder input resolution (multiple of ``patch_stride``).
        patch_stride: encoder patch stride (16 for DINOv3-ViT).
        factor: densification factor per axis (1 = off).

    Returns:
        ``[factor·G, factor·G, D]`` float32 grid.
    """
    img = np.asarray(image)
    if factor <= 1:
        return np.asarray(forward_grid_fn(img, resolution), np.float32)
    h, w = img.shape[:2]
    g = resolution // patch_stride
    out = None
    for ky in range(factor):
        for kx in range(factor):
            oy = int(round(patch_stride * ky / factor / resolution * h))
            ox = int(round(patch_stride * kx / factor / resolution * w))
            shifted = np.roll(img, shift=(-oy, -ox), axis=(0, 1)) if (oy or ox) else img
            grid = np.asarray(forward_grid_fn(shifted, resolution), np.float32)
            if out is None:
                out = np.zeros((factor * g, factor * g, grid.shape[-1]), np.float32)
            out[ky::factor, kx::factor] = grid
    return out


def _to_gray01(a):
    a = np.asarray(a).astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    return (a - a.min()) / (np.ptp(a) + 1e-6)


def jbu_snap(feat, guide_image, sigma_spatial=1.0, sigma_range=0.1):
    """Parameter-free joint bilateral filter at grid resolution.

    Each cell is re-weighted by a spatial Gaussian × a range Gaussian on the guide (the image
    box-resampled to the grid). Snaps features to object boundaries before matching — a
    feature-level analogue of INSID3's post-hoc CRF. Output keeps ``feat``'s shape and is
    L2-normalised. A near-uniform guide ⇒ Gaussian smoothing; a step-edge guide ⇒ no bleed.

    Args:
        feat: ``[g0, g1, D]`` unit-normed grid.
        guide_image: any ``H×W[×C]`` image (the high-res edge guide).
        sigma_spatial: spatial Gaussian width in grid cells (kernel radius = ``round(2σ)``).
        sigma_range: range Gaussian width on the 0..1 guide intensity.

    Returns:
        ``[g0, g1, D]`` float32, L2-normalised.
    """
    feat = np.asarray(feat, np.float32)
    g0, g1, _ = feat.shape
    gray = _to_gray01(guide_image)
    ys = (np.arange(g0) * gray.shape[0] / g0).astype(int).clip(0, gray.shape[0] - 1)
    xs = (np.arange(g1) * gray.shape[1] / g1).astype(int).clip(0, gray.shape[1] - 1)
    guide = gray[np.ix_(ys, xs)]                               # [g0, g1]
    radius = max(1, int(round(2 * sigma_spatial)))
    acc = np.zeros_like(feat)
    wsum = np.zeros((g0, g1, 1), np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            w_s = np.exp(-(dy * dy + dx * dx) / (2 * sigma_spatial ** 2))
            fs = np.roll(feat, (dy, dx), axis=(0, 1))
            gs = np.roll(guide, (dy, dx), axis=(0, 1))
            w_r = np.exp(-((guide - gs) ** 2) / (2 * sigma_range ** 2))
            w = (w_s * w_r)[..., None].astype(np.float32)
            acc += fs * w
            wsum += w
    out = acc / np.maximum(wsum, 1e-6)
    return (out / np.maximum(np.linalg.norm(out, axis=2, keepdims=True), 1e-6)).astype(np.float32)
