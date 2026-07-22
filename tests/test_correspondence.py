import numpy as np

from active_segmenter.config import MatchConfig
from active_segmenter.membank.bank import MemoryBank
from active_segmenter.propose import correspondence as corr


def _planted_bank(d=8, class_id=1):
    """fg cluster near +e0, bg cluster near -e0."""
    rng = np.random.RandomState(0)
    fg = np.zeros((20, d), np.float32)
    fg[:, 0] = 1.0
    fg += 0.02 * rng.randn(20, d)
    bg = np.zeros((20, d), np.float32)
    bg[:, 0] = -1.0
    bg += 0.02 * rng.randn(20, d)
    fg /= np.linalg.norm(fg, axis=1, keepdims=True)
    bg /= np.linalg.norm(bg, axis=1, keepdims=True)
    b = MemoryBank()
    from active_segmenter.membank.exemplar import InstanceExemplar

    b._by_class[class_id].append(InstanceExemplar(class_id, 1, fg, bg, None, 0))
    return b


def test_score_map_positive_on_fg_negative_on_bg():
    d = 8
    b = _planted_bank(d)
    grid = np.zeros((2, 2, d), np.float32)
    grid[0, 0, 0] = 1.0   # fg-like
    grid[1, 1, 0] = -1.0  # bg-like
    grid[0, 1, 0] = 1.0
    grid[1, 0, 0] = -1.0
    grid /= np.linalg.norm(grid, axis=2, keepdims=True)
    s = corr.score_map(grid, b, 1, MatchConfig(bidirectional=False))
    assert s[0, 0] > 0 and s[0, 1] > 0
    assert s[1, 1] < 0 and s[1, 0] < 0


def test_prelabel_returns_per_class_maps():
    d = 8
    b = _planted_bank(d, class_id=3)
    grid = np.zeros((2, 2, d), np.float32)
    grid[:, :, 0] = 1.0
    grid /= np.linalg.norm(grid, axis=2, keepdims=True)
    out = corr.prelabel(grid, b, MatchConfig(bidirectional=False))
    assert set(out.keys()) == {3}
    assert out[3].shape == (2, 2)


def test_margin_small_when_ambiguous():
    d = 8
    b = _planted_bank(d)
    grid = np.zeros((1, 1, d), np.float32)
    grid[0, 0, 1] = 1.0  # orthogonal to fg/bg axis -> ambiguous
    s = corr.score_map(grid, b, 1, MatchConfig(bidirectional=False))
    assert abs(s[0, 0]) < 0.3


def test_knn_beats_prototype_multimodal_fg():
    """Multimodal foreground (two distinct appearances) is where per-patch kNN
    beats an averaged prototype: the prototype sits between modes and matches
    neither, while kNN matches whichever mode is nearest the query."""
    d = 8
    rng = np.random.RandomState(1)
    modeA = np.zeros((15, d), np.float32); modeA[:, 0] = 1.0; modeA += 0.01 * rng.randn(15, d)
    modeB = np.zeros((15, d), np.float32); modeB[:, 1] = 1.0; modeB += 0.01 * rng.randn(15, d)
    fg = np.concatenate([modeA, modeB])
    bg = np.zeros((30, d), np.float32); bg[:, 0] = -1.0; bg += 0.01 * rng.randn(30, d)
    fg /= np.linalg.norm(fg, axis=1, keepdims=True)
    bg /= np.linalg.norm(bg, axis=1, keepdims=True)
    b = MemoryBank()
    from active_segmenter.membank.exemplar import InstanceExemplar
    b._by_class[1].append(InstanceExemplar(1, 1, fg, bg, None, 0))

    q = np.zeros((1, 1, d), np.float32); q[0, 0, 0] = 1.0  # a mode-A query patch
    q /= np.linalg.norm(q, axis=2, keepdims=True)
    s_knn = corr.score_map(q, b, 1, MatchConfig(topk=5, bidirectional=False))[0, 0]

    # averaged-prototype score for the same query
    fgp = fg.mean(0); fgp /= np.linalg.norm(fgp)
    bgp = bg.mean(0); bgp /= np.linalg.norm(bgp)
    s_proto = float(q[0, 0] @ fgp - q[0, 0] @ bgp)

    assert s_knn > s_proto  # kNN locks onto mode A; the prototype is diluted by mode B
