import numpy as np

from active_segmenter.propose import polygon


def test_mask_to_polygon_and_back_square():
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    poly = polygon.mask_to_polygon(mask, rdp_eps=1.0)
    assert poly is not None and poly.shape[1] == 2
    back = polygon.polygons_to_instance_masks([poly], (20, 20))[0]
    inter = np.logical_and(mask, back).sum()
    union = np.logical_or(mask, back).sum()
    assert inter / union > 0.9  # round-trip preserves the region


def test_polygons_to_masks_keeps_each_channel():
    p1 = np.array([[1, 1], [1, 5], [5, 5], [5, 1]], float)
    p2 = np.array([[4, 4], [4, 8], [8, 8], [8, 4]], float)  # overlaps p1 near (4,4)-(5,5)
    masks = polygon.polygons_to_instance_masks([p1, p2], (10, 10))
    assert len(masks) == 2
    # overlap region belongs to both, independently
    assert masks[0][4, 4] and masks[1][4, 4]


def test_empty_mask_returns_none():
    assert polygon.mask_to_polygon(np.zeros((5, 5), bool)) is None
