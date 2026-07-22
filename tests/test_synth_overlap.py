import numpy as np

from active_segmenter.eval.datasets import make_synthetic_overlap


def test_generates_overlapping_instances():
    data = make_synthetic_overlap(n_images=1, n_instances=3, overlap_frac=0.3, seed=0)
    assert len(data) == 1
    image, masks = data[0]
    assert image.ndim in (2, 3)
    assert len(masks) >= 2
    # at least one pair of instances shares a pixel
    shared = any(
        np.logical_and(masks[i], masks[j]).any()
        for i in range(len(masks)) for j in range(i + 1, len(masks))
    )
    assert shared
    # overlap is real: the union is strictly smaller than the sum of areas
    union_area = np.any(masks, axis=0).sum()
    sum_area = sum(int(m.sum()) for m in masks)
    assert union_area < sum_area


def test_deterministic_under_seed():
    a = make_synthetic_overlap(2, 3, 0.3, seed=1)
    b = make_synthetic_overlap(2, 3, 0.3, seed=1)
    assert np.array_equal(a[0][0], b[0][0])
    assert np.array_equal(a[0][1][0], b[0][1][0])


def test_masks_are_boolean_and_nonempty():
    data = make_synthetic_overlap(1, 4, 0.25, seed=2)
    _, masks = data[0]
    for m in masks:
        assert m.dtype == bool
        assert m.any()
