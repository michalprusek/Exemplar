import numpy as np

from active_segmenter.membank.curation import kcenter
from active_segmenter.membank.bank import MemoryBank


def _norm_grid(seed, g=4, d=8):
    grid = np.random.RandomState(seed).randn(g, g, d).astype(np.float32)
    grid /= np.linalg.norm(grid, axis=2, keepdims=True)
    return grid


def test_kcenter_spreads():
    pts = np.array([[0, 0], [0, 0.01], [10, 10], [10, 10.01]], float)
    idx = set(kcenter(pts, 2, seed=0))
    assert len(idx) == 2
    assert (idx & {0, 1}) and (idx & {2, 3})  # one from each cluster


def test_kcenter_k_geq_n_returns_all():
    pts = np.zeros((3, 2))
    assert sorted(kcenter(pts, 5, seed=0)) == [0, 1, 2]


def test_bank_add_and_query():
    grid = _norm_grid(0)
    label = np.zeros((16, 16), int)
    label[:8, :8] = 1  # instance 1 in the top-left quadrant
    b = MemoryBank()
    b.add_from_annotation(grid, label, class_of={1: 7}, round=0)
    assert 7 in b.classes()
    assert b.fg(7).shape[1] == 8
    assert b.fg(7).shape[0] >= 1
    assert b.bg(7).shape[0] >= 1
    assert b.size(7) == 1  # one exemplar instance


def test_bank_two_instances_same_class():
    grid = _norm_grid(1)
    label = np.zeros((16, 16), int)
    label[:8, :8] = 1
    label[8:, 8:] = 2
    b = MemoryBank()
    b.add_from_annotation(grid, label, class_of={1: 3, 2: 3}, round=0)
    assert b.size(3) == 2  # two exemplars, same class


def test_bank_curate_caps_size():
    b = MemoryBank()
    for r in range(10):
        grid = _norm_grid(r)
        label = np.zeros((16, 16), int)
        label[:8, :8] = 1
        b.add_from_annotation(grid, label, class_of={1: 1}, round=r)
    assert b.size(1) == 10
    b.curate(cap=4)
    assert b.size(1) == 4


def test_bank_json_roundtrip():
    grid = _norm_grid(2)
    label = np.zeros((16, 16), int)
    label[:8, :8] = 1
    b = MemoryBank()
    b.add_from_annotation(grid, label, class_of={1: 5}, round=0)
    b2 = MemoryBank.from_json(b.to_json())
    assert b2.classes() == [5]
    assert b2.fg(5).shape == b.fg(5).shape
