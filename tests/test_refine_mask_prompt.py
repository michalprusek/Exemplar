"""CPU tests for the pure SAM mask-prompt helpers (spec 2026-07-12 refine-stage)."""
import numpy as np

from active_segmenter.refine.sam import mask_bbox_xyxy, mask_to_lowres_logits


def test_lowres_logits_shape_and_sign():
    m = np.zeros((100, 100), bool)
    m[20:80, 30:70] = True
    lo = mask_to_lowres_logits(m, size=256)
    assert lo.shape == (256, 256)
    assert lo.max() > 0 > lo.min()  # inside positive, outside negative
    assert lo[128, 128] > 0  # interior maps to positive logits


def test_bbox_tight_and_none_on_empty():
    m = np.zeros((50, 60), bool)
    m[10:20, 15:40] = True
    assert mask_bbox_xyxy(m) == [15, 10, 39, 19]
    assert mask_bbox_xyxy(np.zeros((5, 5), bool)) is None
