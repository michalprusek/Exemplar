"""The magnified fine view and the rule that decides its magnification from the support masks.

A 16-px patch cannot describe a structure narrower than itself, so the backbone is read a second time on the image
magnified by an integer z, tiled into 672-px windows; the windows' patch grids are stitched into one grid with a stride
of 16 / z native pixels, whatever the field size.

Magnification (`decide_zoom`): the smallest z at which a labelling constant on each 16-px patch can reproduce the
support masks well enough, median over the support of the skeleton-length-weighted patch Dice >= 0.8, with
z <= min(4, 8192 // longest support side). The pixel weights 1 / local thickness make a structure count by its length,
not its area. Decided from the K support masks alone, before any fine grid is read.

Fine view (`fine_view`): at z > 1 the image is magnified bilinearly (half-pixel centres) to sides rounded UP to
multiples of 16 and tiled with a step of 496 px, so every window lies on the patch grid (registered). At z = 1 the
native tiling (quarter overlap, step 504) is used. Each window is reflect-padded to 672 px, min-max normalised over
the window, ImageNet-normalised and encoded (6 windows per forward); the grids are averaged where windows overlap. The
stitched grid is capped at ceil(2560 z / 16) cells per side (bilinear to an explicit size when registered).

Source: scripts/closed_form_bench.py:369-463 (`fast_fine`) and 666-695 (`zoom_unified_parallel`),
active_segmenter/segment/head_fusion_backend.py:761-834 and acquire/crop_tiles.py:25-39 (AutoSeg).
Line references point to the authors' research code, which is not public.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from exemplar.encoder import LAYERS, N_PREFIX, PATCH, RESOLUTION, fuse_blocks, hidden_states_needed

ZOOM_MAX = 4              # fine_zoom_max (al_testbed.py:358)
SIDE_BUDGET = 8192        # fine_zoom_side: z x longest support side <= 8192 px (al_testbed.py:361)
ZOOM_TAU = 0.8            # fine_zoom_tau (head_fusion_backend.py:1855)
TILE_BATCH = 6            # windows per encoder forward (head_fusion_backend.py:1944)
NATIVE_CAP = 2560         # the fine grid is resampled above 2560 native px per side

_POOL = None


def _executor():
    """The thread pool the per-mask magnification measurements run in (closed_form_bench.py:2242)."""
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=int(os.environ.get("OMP_NUM_THREADS", "8")))
    return _POOL


def fine_max_grid(z: int) -> int:
    """The cap of the fine grid in cells per side: ceil(2560 z / 16). (closed_form_bench.py:2425)"""
    return -(-NATIVE_CAP * int(z) // PATCH)


# ---------------------------------------------------------------------------------------------------------------------
# tiling (acquire/crop_tiles.py:25-39)

def _positions(n: int, crop: int, step: int) -> list:
    if n <= crop:
        return [0]
    ps = list(range(0, n - crop + 1, step))
    if ps[-1] != n - crop:
        ps.append(n - crop)               # always cover the right/bottom edge
    return ps


def tile_grid(h: int, w: int, crop: int, overlap: float = 0.25) -> list:
    """Top-left (y, x) of every crop covering an h x w image with fractional overlap."""
    step = max(1, int(round(crop * (1.0 - overlap))))
    return [(y, x) for y in _positions(h, crop, step) for x in _positions(w, crop, step)]


def _reflect_idx(n, total, device):
    """Indices of np.pad(mode="reflect") for any pad width (period 2(n - 1)). (closed_form_bench.py:369-374)"""
    i = torch.arange(total, device=device)
    if n == 1:
        return torch.zeros_like(i)
    j = i % (2 * (n - 1))
    return torch.where(j >= n, 2 * (n - 1) - j, j)


# ---------------------------------------------------------------------------------------------------------------------
# the fine view

def fine_view(model, image, zoom: int, max_grid: int, device, batch: int = TILE_BATCH, layers=LAYERS) -> torch.Tensor:
    """The fine view of one image at magnification `zoom`: [1, 2048, hg, wg] float32 on `device` ([1, 1024, hg, wg]
    with layers=(-1,), the ablation "last layer only").

    `model` is the DINOv3 model (called with pixel_values and output_hidden_states, through the bf16 wrapper).
    (closed_form_bench.py:376-463, `fast_fine`, for the reported configuration: one view, the main encoder,
    registered where magnified)"""
    _imean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    _istd = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    res, ps = RESOLUTION, PATCH
    arr0 = np.asarray(image)
    t = torch.from_numpy(np.ascontiguousarray(arr0, dtype=np.float32)).to(device)
    t = t[None, None] if arr0.ndim == 2 else t.permute(2, 0, 1)[None]          # [1, C, H, W]
    hh0, ww0 = t.shape[-2:]
    z = int(zoom)
    registered = z > 1                                   # `fam`: registered only where magnified
    if registered:
        size = (-(-hh0 * z // ps) * ps, -(-ww0 * z // ps) * ps)
        t = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
    hh, ww = t.shape[-2:]
    hg, wg = max(1, hh // ps), max(1, ww // ps)
    if registered:
        step = ((res * 3 // 4) // ps) * ps               # 496
        positions = [(y, x) for y in _positions(hh, res, step) for x in _positions(ww, res, step)]
    else:
        positions = list(tile_grid(hh, ww, res, 0.25))
    npre = N_PREFIX
    acc = cnt = None
    bsz = int(batch)
    for i in range(0, len(positions), bsz):
        px = []
        for (y, x) in positions[i:i + bsz]:
            c = t[:, :, y:y + res, x:x + res]
            ph, pw = res - c.shape[-2], res - c.shape[-1]
            if ph or pw:
                # np.pad(mode="reflect") for any pad width; torch's reflect padding refuses a pad wider than the window
                c = c[:, :, _reflect_idx(c.shape[-2], res, device)][:, :, :, _reflect_idx(c.shape[-1], res, device)]
            # grey -> 3 channels, alpha dropped; min-max over the whole window
            if c.shape[1] == 1:
                c = c.repeat(1, 3, 1, 1)
            elif c.shape[1] == 4:
                c = c[:, :3]
            c = (c - c.min()) / ((c.max() - c.min()) + 1e-6)
            px.append((c - _imean) / _istd)
        px = torch.cat(px, 0)
        with torch.no_grad():
            out = model(pixel_values=px, output_hidden_states=hidden_states_needed(layers))
            parts = [(out.last_hidden_state if li == -1 else out.hidden_states[li])[:, npre:] for li in layers]
            f_ = fuse_blocks([p_.float() for p_ in parts], 2)
        g_ = res // ps
        f_ = f_.reshape(f_.shape[0], g_, g_, -1)
        if acc is None:
            acc = torch.zeros(hg, wg, f_.shape[-1], device=device)
            cnt = torch.zeros(hg, wg, 1, device=device)
        for (y, x), gg in zip(positions[i:i + bsz], f_):
            gy0, gx0 = min(y // ps, hg - 1), min(x // ps, wg - 1)
            gy1, gx1 = min(gy0 + g_, hg), min(gx0 + g_, wg)
            acc[gy0:gy1, gx0:gx1] += gg[:gy1 - gy0, :gx1 - gx0]
            cnt[gy0:gy1, gx0:gx1] += 1.0
    grid = (acc / cnt.clamp_min(1e-6)).permute(2, 0, 1)[None]
    m = max(hg, wg)
    if m > max_grid:
        sc = max_grid / m
        if registered:
            grid = F.interpolate(grid, size=(max(1, round(hg * sc)), max(1, round(wg * sc))),
                                 mode="bilinear", align_corners=False)
        else:
            grid = F.interpolate(grid, scale_factor=sc, mode="bilinear", align_corners=False)
    return grid


# ---------------------------------------------------------------------------------------------------------------------
# the magnification rule (head_fusion_backend.py:761-834, closed_form_bench.py:666-695)

def _thickness_weights(label_map):
    """Pixel weights u_p = 1 / t_p, t_p the local thickness (twice the distance transform) at the pixel's nearest
    skeleton pixel, at least 1 px. None for a mask with fewer than 4 foreground pixels or no skeleton.
    (head_fusion_backend.py:761-780)"""
    from scipy import ndimage
    from skimage.morphology import skeletonize

    m = np.asarray(label_map)
    if m.ndim == 3:
        m = m[..., 0]
    m = m > 0
    if int(m.sum()) < 4:
        return None
    sk = skeletonize(m)
    if not sk.any():
        return None
    dt = ndimage.distance_transform_edt(m)
    iy, ix = ndimage.distance_transform_edt(~sk, return_distances=False, return_indices=True)
    return 1.0 / np.maximum(2.0 * dt[iy, ix], 1.0)


def _patch_dice_weighted(label_map, z: int, patch: int = 16, u=None) -> float:
    """The best Dice a labelling constant on each patch can reach on the mask magnified z times (nearest), every pixel
    weighted by u: max_S 2 sum_{c in S} g_c / (G + sum_{c in S} a_c), reached on a prefix of the cells sorted by
    g_c / a_c; mean over four phases of the grid. (head_fusion_backend.py:783-813)"""
    m = np.asarray(label_map)
    if m.ndim == 3:
        m = m[..., 0]
    m = m > 0
    if not m.any():
        return float("nan")
    t = torch.from_numpy(m.astype(np.float32))[None, None]
    w = torch.ones_like(t) if u is None else torch.from_numpy(np.asarray(u, np.float32))[None, None]
    if z > 1:
        t = torch.nn.functional.interpolate(t, scale_factor=int(z), mode="nearest")
        w = torch.nn.functional.interpolate(w, scale_factor=int(z), mode="nearest")
    h = patch // 2

    def cells(x, dy, dx):
        return torch.nn.functional.avg_pool2d(torch.nn.functional.pad(x, (dx, 0, dy, 0)), patch, ceil_mode=True,
                                              divisor_override=1).flatten().double()

    vals = []
    for dy, dx in ((0, 0), (h, h), (0, h), (h, 0)):
        g, a = cells(t * w, dy, dx), cells(w, dy, dx)
        keep = g > 0
        g, a = g[keep], a[keep]
        o = torch.argsort(g / a, descending=True)
        vals.append(float((2 * torch.cumsum(g[o], 0) / (g.sum() + torch.cumsum(a[o], 0))).max()))
    return float(np.mean(vals))


def decide_zoom(label_maps, sides, measures=None) -> int:
    """The magnification of the fine view for a support set: the smallest z with median weighted patch Dice >= 0.8.

    label_maps: the K support masks; sides: the longest side of each support image. measures (for an exact refit): a
    list with one dict per mask that keeps that mask's weights and patch Dice values, so adding a mask measures only
    the new one; the decision is the same with or without it. (closed_form_bench.py:666-695,
    `zoom_unified_parallel`, the same decision as head_fusion_backend.py:816-834 `_support_zoom_unified`)"""
    pool = _executor()
    zmax, sbud, tau = ZOOM_MAX, SIDE_BUDGET, ZOOM_TAU
    ps = PATCH
    if measures is None:
        measures = [None] * len(label_maps)
    if len(measures) != len(label_maps):
        raise ValueError("decide_zoom: one measurement dict per mask")

    def weights(mi):
        m, c = mi
        if c is None:
            return _thickness_weights(m)
        if "u" not in c:
            c["u"] = _thickness_weights(m)
        return c["u"]
    us = list(pool.map(weights, zip(label_maps, measures)))
    keep_ = [(m, u, c) for m, u, c in zip(label_maps, us, measures) if u is not None]
    if not keep_:
        return 1
    z_s = max(1, int(sbud // max(1, int(max(sides)))))
    z_top = max(1, min(zmax, z_s))

    def du(muc, zz):                                     # the weighted patch Dice of one mask, kept for a refit
        m, u, c = muc
        if c is not None and ("du", zz) in c:
            return c[("du", zz)]
        v = _patch_dice_weighted(m, zz, ps, u)
        if c is not None:
            c[("du", zz)] = v
        return v
    for zz in range(1, z_top + 1):
        ds_ = list(pool.map(lambda muc, zz=zz: du(muc, zz), keep_))
        if float(np.median(ds_)) >= tau:
            return zz
    return z_top
