"""Synthetic support images and frozen inputs for the CPU tests (no DINOv3 weights needed).

The readout only sees grids and maps, so small deterministic stand-ins for the coarse view, the fine view and the
priors exercise exactly the code that runs on the real inputs: grids coarser than the image (bilinear interpolation,
the Kronecker Gram), a fine grid whose size follows the magnification, native-resolution prior maps.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def blob_image(i, H=44, W=52, thin=False, seed=0, noise=0.25):
    """A noisy image with discs (or, thin=True, one-pixel lines) as foreground; returns (image float32, mask int32).
    noise: amplitude of the uniform noise against a foreground contrast of 0.5."""
    rs = np.random.RandomState(seed * 1000 + i)
    yy, xx = np.mgrid[:H, :W]
    m = np.zeros((H, W), bool)
    if thin:
        for k in range(3):
            r0 = rs.randint(4, H - 4)
            m[r0, 3:W - 3] = True
            m[3:H - 3, rs.randint(4, W - 4)] = True
    else:
        for k in range(2 + i % 2):
            cy, cx, r = rs.randint(8, H - 8), rs.randint(8, W - 8), rs.randint(4, 8)
            m |= (yy - cy) ** 2 + (xx - cx) ** 2 < r * r
    img = (0.2 + noise * rs.rand(H, W) + 0.5 * m + 0.15 * np.sin(xx / 3.0 + i)).astype(np.float32)
    return img, m.astype(np.int32)


def disc_image(r, H=80, W=96, seed=0):
    """One disc of radius r in a noisy image; returns (image float32, mask int32)."""
    rs = np.random.RandomState(seed)
    yy, xx = np.mgrid[:H, :W]
    m = (yy - H // 2) ** 2 + (xx - W // 2) ** 2 < r * r
    img = (0.2 + 0.25 * rs.rand(H, W) + 0.5 * m).astype(np.float32)
    return img, m.astype(np.int32)


def features(image, zoom=1):
    """Stand-ins for (coarse view, fine view at magnification zoom, priors, hw), deterministic in the image."""
    t = torch.from_numpy(np.asarray(image, np.float32))[None, None]
    H, W = t.shape[-2:]
    ch = torch.cat([t, t ** 2, 1 - t, torch.sin(3 * t), torch.cos(5 * t), F.avg_pool2d(t, 5, 1, 2)], 1)
    xc = F.interpolate(ch, size=(6, 7), mode="area")
    hf, wf = max(2, -(-H * zoom // 8)), max(2, -(-W * zoom // 8))
    fine = F.interpolate(torch.cat([t, t.sqrt(), F.max_pool2d(t, 3, 1, 1), torch.exp(-t), t * (1 - t)], 1),
                         size=(hf, wf), mode="bilinear", align_corners=False)
    bank = torch.cat([t, F.avg_pool2d(t, 3, 1, 1), -F.max_pool2d(-t, 3, 1, 1), F.avg_pool2d(t, 7, 1, 3)], 1)
    return xc, fine, bank, (H, W)


class SyntheticInputs:
    """A drop-in for exemplar.FrozenInputs: (image, zoom) -> (xc, fine, bank, hw)."""

    def __call__(self, image, zoom):
        return features(image, zoom)

    def coarse(self, image):
        return None


def support(n, thin=False, seed=0, **kw):
    pairs = [blob_image(i, thin=thin, seed=seed, **kw) for i in range(n)]
    return [p[0] for p in pairs], [p[1] for p in pairs]
