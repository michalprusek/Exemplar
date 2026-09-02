import numpy as np

from active_segmenter.config import ClusterConfig
from active_segmenter.propose import instances


def _square_score(val=1.0, size=10):
    s = np.full((size, size), -1.0, np.float32)
    s[4:6, 4:6] = val
    return s


def test_two_blobs_two_instances():
    s = np.full((10, 10), -1.0, np.float32)
    s[1:3, 1:3] = 1.0   # blob A
    s[7:9, 7:9] = 1.0   # blob B, spatially separate
    out = instances.decompose(s, ClusterConfig(score_thresh=0.0, min_patches=2), class_id=1)
    assert len(out) == 2
    assert all(m.class_id == 1 for m in out)
    assert all(m.mask.shape == (10, 10) for m in out)


def test_single_blob_one_instance():
    s = _square_score()
    out = instances.decompose(s, ClusterConfig(score_thresh=0.0, min_patches=1), class_id=2)
    assert len(out) == 1
    assert out[0].class_id == 2


def test_min_patches_filters_specks():
    s = np.full((10, 10), -1.0, np.float32)
    s[5, 5] = 1.0  # a single-patch speck
    out = instances.decompose(s, ClusterConfig(score_thresh=0.0, min_patches=2), class_id=1)
    assert len(out) == 0


def test_empty_when_all_background():
    s = np.full((8, 8), -1.0, np.float32)
    out = instances.decompose(s, ClusterConfig(score_thresh=0.0), class_id=1)
    assert out == []


def test_overlap_preserved_across_classes():
    # two classes independently claim an overlapping region; both keep the pixel
    s = _square_score()
    a = instances.decompose(s, ClusterConfig(min_patches=1), class_id=1)
    b = instances.decompose(s, ClusterConfig(min_patches=1), class_id=2)
    assert a[0].mask[4, 4] and b[0].mask[4, 4]  # same pixel in both -> overlap survives


def test_max_instances_cap():
    # a grid of many separate specks, capped
    s = np.full((20, 20), -1.0, np.float32)
    for i in range(2, 18, 4):
        for j in range(2, 18, 4):
            s[i:i + 2, j:j + 2] = 1.0
    out = instances.decompose(s, ClusterConfig(min_patches=1, max_instances=3), class_id=1)
    assert len(out) <= 3
