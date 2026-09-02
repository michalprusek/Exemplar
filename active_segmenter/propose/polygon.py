"""Mask <-> polygon round-trip.

mask -> polygon: marching squares (skimage ``find_contours``) + RDP simplification
(``approximate_polygon``). polygon -> mask: rasterise EACH instance to its own
boolean channel. Overlap survives iff instances are never merged into one raster.
Polygon points are ``[K, 2]`` in ``(row, col)`` order.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from skimage import measure
from skimage.draw import polygon as _draw_polygon


def mask_to_polygon(mask_bool: np.ndarray, rdp_eps: float = 2.0) -> Optional[np.ndarray]:
    mask = np.asarray(mask_bool, bool)
    if not mask.any():
        return None
    padded = np.pad(mask.astype(np.float32), 1)  # pad so edge-touching contours close
    contours = measure.find_contours(padded, 0.5)
    if not contours:
        return None
    contour = max(contours, key=len) - 1.0  # undo pad offset
    poly = measure.approximate_polygon(contour, tolerance=rdp_eps)
    if len(poly) < 3:
        return None
    return poly.astype(np.float32)


def polygons_to_instance_masks(polys, hw) -> list[np.ndarray]:
    h, w = hw
    masks = []
    for p in polys:
        p = np.asarray(p, np.float32)
        rr, cc = _draw_polygon(p[:, 0], p[:, 1], shape=(h, w))
        m = np.zeros((h, w), bool)
        m[rr, cc] = True
        masks.append(m)
    return masks
