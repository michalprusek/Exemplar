"""Tests for SMEC — Support-Marginal Error Coverage acquisition.

SMEC has three pure, GPU-free pieces, each tested in isolation:
- ``mask_disagreement``      : the label-free error signal (committee IoU spread)
- ``coverage_weighted_scores``: facility-location over the error field
- ``smec_scores``            : the two composed via an injected frozen segmenter
"""
import numpy as np

from active_segmenter.acquire import smec


# ---- mask_disagreement: the label-free error signal ---------------------------
def test_mask_disagreement_identical_masks_is_zero():
    m = np.array([[True, False, True], [False, True, False]])
    assert smec.mask_disagreement([m, m.copy(), m.copy()]) == 0.0


def test_mask_disagreement_disjoint_masks_is_one():
    a = np.array([[True, True], [False, False]])
    b = ~a  # zero intersection, full union -> IoU 0 -> disagreement 1
    assert smec.mask_disagreement([a, b]) == 1.0


def test_mask_disagreement_single_member_is_zero():
    assert smec.mask_disagreement([np.ones((2, 2), bool)]) == 0.0


def test_mask_disagreement_half_overlap_is_one_half():
    a = np.array([[True, True], [False, False]])   # 2 fg
    b = np.array([[True, False], [False, False]])  # subset of a -> IoU 1/2
    assert abs(smec.mask_disagreement([a, b]) - 0.5) < 1e-9


# ---- coverage_weighted_scores: facility-location over the error field ---------
def _two_clusters(d=8):
    """Cluster A (idx 0-4) near e0, cluster B (idx 5-9) near e1, unit-normalised."""
    rng = np.random.RandomState(0)
    cls = np.zeros((10, d), np.float32)
    cls[:5, 0] = 1.0
    cls[5:, 1] = 1.0
    cls += 0.01 * rng.randn(10, d)
    cls /= np.linalg.norm(cls, axis=1, keepdims=True)
    return cls


def test_coverage_weighted_prefers_representative_of_high_error_region():
    cls = _two_clusters()
    labeled = []                       # nothing covered yet
    pool = list(range(10))
    err = {u: (1.0 if u >= 5 else 0.0) for u in pool}  # all error mass in cluster B
    scores = smec.coverage_weighted_scores(cls, labeled, pool, err)
    assert max(pool, key=lambda i: scores[i]) >= 5     # picks a cluster-B frame


def test_coverage_weighting_downweights_already_covered_region():
    cls = _two_clusters()
    labeled = [0]                      # cluster A covered by exemplar 0
    pool = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    err = {u: 1.0 for u in pool}       # equal raw error; novelty must route to B
    scores = smec.coverage_weighted_scores(cls, labeled, pool, err)
    assert max(pool, key=lambda i: scores[i]) >= 5     # covered A is downweighted


# ---- smec_scores: committee disagreement x coverage, end to end ---------------
def _three_clusters(d=8):
    """A (0-3) near e0, B (4-7) near e1, C (8-11) near e2 — unit-normalised."""
    rng = np.random.RandomState(0)
    cls = np.zeros((12, d), np.float32)
    cls[0:4, 0] = 1.0
    cls[4:8, 1] = 1.0
    cls[8:12, 2] = 1.0
    cls += 0.01 * rng.randn(12, d)
    cls /= np.linalg.norm(cls, axis=1, keepdims=True)
    return cls


def test_smec_cold_start_falls_back_to_coverage():
    # below min_support there is no committee -> pure coverage of the uncovered mass
    cls = _two_clusters()
    labeled = [0]                      # 1 label < min_support=2
    pool = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def never_called(support, target):
        raise AssertionError("predict_fn must not run below min_support")

    scores = smec.smec_scores(pool, labeled, cls, never_called, min_support=2)
    assert max(pool, key=lambda i: scores[i]) >= 5     # picks uncovered cluster, TypiClust-like


def test_smec_prefers_unstable_region_over_stable_uncovered_region():
    # A is covered; B and C are equally uncovered, but only B's predictions flip
    # across support subsets. Coverage alone would tie B and C; disagreement breaks it.
    cls = _three_clusters()
    labeled = [0, 1, 2]                       # cluster A, covered
    pool = [3, 4, 5, 6, 7, 8, 9, 10, 11]      # 3:A  4-7:B  8-11:C
    stable = np.ones((4, 4), bool)

    def predict_fn(support, target):
        if 4 <= target <= 7:                  # cluster B: mask depends on support subset
            m = np.zeros((4, 4), bool)
            m[int(sum(support)) % 4] = True
            return m
        return stable                         # A and C: same mask regardless of support

    scores = smec.smec_scores(pool, labeled, cls, predict_fn,
                              n_committee=6, subset_frac=0.67, seed=0, min_support=2)
    assert 4 <= max(pool, key=lambda i: scores[i]) <= 7   # unstable B beats stable C


def test_smec_all_agree_falls_back_to_coverage():
    cls = _two_clusters()
    labeled = [0, 1]                          # cluster A
    pool = [2, 3, 4, 5, 6, 7, 8, 9]
    stable = np.ones((4, 4), bool)

    def predict_fn(support, target):
        return stable                         # model agrees everywhere -> zero error field

    scores = smec.smec_scores(pool, labeled, cls, predict_fn, min_support=2, seed=0)
    assert max(pool, key=lambda i: scores[i]) >= 5        # degenerate -> cover uncovered cluster


# ---- zscore_fuse: combine complementary acquisition signals -------------------
def test_zscore_fuse_agreeing_scorers_reinforce():
    a = {0: 5.0, 1: 1.0, 2: 0.0}   # prefers 0
    b = {0: 9.0, 1: 2.0, 2: 1.0}   # also prefers 0
    fused = smec.zscore_fuse([a, b])
    assert max(fused, key=lambda k: fused[k]) == 0


def test_zscore_fuse_ignores_zero_variance_scorer():
    a = {0: 1.0, 1: 5.0, 2: 2.0}   # prefers 1
    flat = {0: 3.0, 1: 3.0, 2: 3.0}  # no information -> must not change the ranking
    fused = smec.zscore_fuse([a, flat])
    assert max(fused, key=lambda k: fused[k]) == 1
    order = sorted(fused, key=lambda k: fused[k])
    assert order == sorted(a, key=lambda k: a[k])  # ranking is exactly a's


def test_zscore_fuse_weight_shifts_winner():
    a = {0: 10.0, 1: 0.0}   # strongly prefers 0
    b = {0: 0.0, 1: 1.0}    # mildly prefers 1
    # equal weights: a's larger spread is z-normalised away -> both contribute ±1, tie broken
    # but weighting b heavily must flip the winner to 1
    fused = smec.zscore_fuse([a, b], weights=[1.0, 5.0])
    assert max(fused, key=lambda k: fused[k]) == 1
