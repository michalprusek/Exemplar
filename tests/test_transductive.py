import numpy as np

from active_segmenter.config import MatchConfig
from active_segmenter.membank.bank import MemoryBank
from active_segmenter.acquire import transductive as T


def test_typicality_higher_in_dense_region():
    # 5 points tightly clustered near origin + 1 far outlier
    cls = np.array([[0, 0], [0.01, 0], [0, 0.01], [-0.01, 0], [0, -0.01], [10, 10]], np.float32)
    typ = T.typicality(cls, k=2)
    assert typ[:5].mean() > typ[5]  # dense points more typical than the outlier


def test_typiclust_picks_from_uncovered_cluster():
    # two clear clusters; one already has a labeled member -> pick from the other
    cls = np.concatenate([
        np.zeros((5, 2)) + [0, 0] + 0.01 * np.random.RandomState(0).randn(5, 2),
        np.zeros((5, 2)) + [10, 10] + 0.01 * np.random.RandomState(1).randn(5, 2),
    ]).astype(np.float32)
    typ = T.typicality(cls, k=3)
    labeled = [0]  # covers cluster A (indices 0-4)
    pool = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    scores = T.typiclust_rank_scores(cls, labeled, pool, typ, seed=0)
    pick = max(pool, key=lambda i: scores[i])
    assert pick >= 5  # picks from the uncovered cluster B (indices 5-9)


def _grid_bank(seed, fgval=1.0):
    rng = np.random.RandomState(seed)
    grid = np.zeros((4, 4, 6), np.float32)
    grid[:2, :, 0] = fgval      # top half = fg direction
    grid[2:, :, 0] = -1.0       # bottom half = bg
    grid += 0.01 * rng.randn(4, 4, 6)
    grid /= np.linalg.norm(grid, axis=2, keepdims=True)
    return grid


def test_proxy_eer_runs_and_scores_finite():
    feats = [_grid_bank(s) for s in range(5)]
    label = np.zeros((16, 16), int); label[:8, :] = 1
    bank = MemoryBank()
    bank.add_from_annotation(feats[0], label, {1: 1}, 0)
    pool = [1, 2, 3, 4]
    scores = T.proxy_eer_scores(pool, bank, feats, feats[1:3], MatchConfig(bidirectional=False), "cpu")
    assert set(scores) == set(pool)
    assert all(np.isfinite(v) for v in scores.values())


def test_pool_confidence_increases_with_matching_exemplar():
    feats = [_grid_bank(s) for s in range(4)]
    label = np.zeros((16, 16), int); label[:8, :] = 1
    empty = MemoryBank()
    empty.add_from_annotation(feats[0], label, {1: 1}, 0)
    base = T.pool_confidence(feats[1:], empty, MatchConfig(bidirectional=False), "cpu")
    # adding another matching exemplar should not reduce confidence
    b2 = empty.copy()
    b2.add_from_grid_mask(feats[1], _grid_bank(1)[..., 0] > 0, 1, 0)
    after = T.pool_confidence(feats[1:], b2, MatchConfig(bidirectional=False), "cpu")
    assert after >= base - 1e-6
