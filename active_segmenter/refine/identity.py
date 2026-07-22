"""Identity / no-SAM refiner (the most in-context option).

Upsamples each grid-resolution instance mask to native resolution (nearest). If
``cfg.use_crf`` and pydensecrf is installed, applies a dense-CRF to snap mask
boundaries to image edges (INSID3-style); otherwise a plain upsample. Masks that
are already native-resolution pass through unchanged.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from skimage.transform import resize

from active_segmenter.config import RefineConfig
from active_segmenter.types import InstanceMask


class IdentityRefiner:
    def __init__(self, cfg: RefineConfig):
        self.cfg = cfg

    def refine(self, image, instance_masks, feat_grid: Optional[np.ndarray] = None):
        h, w = np.asarray(image).shape[:2]
        out = []
        for m in instance_masks:
            if m.mask.shape == (h, w):
                mask = m.mask
            else:
                mask = resize(m.mask.astype(np.float32), (h, w), order=0, mode="edge",
                              anti_aliasing=False) > 0.5
            if self.cfg.use_crf:
                mask = _maybe_crf(np.asarray(image), mask)
            out.append(InstanceMask(mask=mask, points=None, class_id=m.class_id,
                                    instance_id=m.instance_id, score=m.score))
        return out


def _maybe_crf(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_labels
    except Exception:
        return mask  # CRF optional — skip cleanly if unavailable
    if image.ndim == 2:
        image = np.stack([image] * 3, -1)
    img = np.ascontiguousarray(image[..., :3].astype(np.uint8))
    h, w = mask.shape
    d = dcrf.DenseCRF2D(w, h, 2)
    labels = mask.astype(np.int32)
    U = unary_from_labels(labels, 2, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=30, srgb=13, rgbim=img, compat=10)
    q = d.inference(5)
    return np.argmax(np.array(q).reshape(2, h, w), axis=0).astype(bool)
