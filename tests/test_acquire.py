import numpy as np

from active_segmenter.acquire import coldstart, convergence, diversity
from active_segmenter.acquire.base import AcqContext
from active_segmenter.acquire.random import RandomAcq
from active_segmenter.acquire.uncertainty import UncertaintyAcq, ambiguous_fraction
from active_segmenter.acquire.epig import EpigAcq


def _clustered_cls(d=8):
    """Three tight clusters of CLS embeddings."""
    rng = np.random.RandomState(0)
    centers = np.eye(3, d)
    cls = []
    for c in centers:
        cls.append(c + 0.02 * rng.randn(10, d))
    cls = np.concatenate(cls).astype(np.float32)
    cls /= np.linalg.norm(cls, axis=1, keepdims=True)
    return cls  # 30 points, 3 clusters of 10


def test_typiclust_picks_spread_across_clusters():
    cls = _clustered_cls()
    picks = coldstart.typiclust(cls, k=3, seed=0)
    assert len(picks) == 3
    # the three picks should land in three different clusters (indices 0-9,10-19,20-29)
    buckets = {p // 10 for p in picks}
    assert len(buckets) == 3


def test_random_acq_deterministic_under_seed():
    ctx = AcqContext(uncertainty=np.zeros(5), cls=np.zeros((5, 4), np.float32),
                     rng=np.random.default_rng(1))
    a = RandomAcq().rank([0, 1, 2, 3, 4], ctx)
    ctx2 = AcqContext(uncertainty=np.zeros(5), cls=np.zeros((5, 4), np.float32),
                      rng=np.random.default_rng(1))
    b = RandomAcq().rank([0, 1, 2, 3, 4], ctx2)
    assert a == b
    assert sorted(a) == [0, 1, 2, 3, 4]


def test_uncertainty_ranks_high_uncertainty_first():
    unc = np.array([0.1, 0.9, 0.5, 0.3])
    ctx = AcqContext(uncertainty=unc, cls=np.zeros((4, 4), np.float32), rng=np.random.default_rng(0))
    ranked = UncertaintyAcq().rank([0, 1, 2, 3], ctx)
    assert ranked[0] == 1  # highest uncertainty first
    assert ranked[-1] == 0


def test_ambiguous_fraction():
    score = np.array([[0.0, 1.0], [-1.0, 0.01]], np.float32)  # 2 of 4 ambiguous at eps .03
    assert abs(ambiguous_fraction(score, eps=0.03) - 0.5) < 1e-9


def test_epig_prefers_uncertain_and_representative():
    # two candidates equally uncertain; one sits in the dense pool region (representative),
    # the other is an outlier. EPIG (target=pool) prefers the representative one.
    d = 6
    cls = np.zeros((6, d), np.float32)
    cls[:5, 0] = 1.0                 # dense pool cluster near e0
    cls[5, 1] = 1.0                  # outlier near e1
    cls += 0.01 * np.random.RandomState(0).randn(6, d)
    cls /= np.linalg.norm(cls, axis=1, keepdims=True)
    unc = np.zeros(6); unc[4] = 1.0; unc[5] = 1.0  # candidates 4 (representative) and 5 (outlier)
    ctx = AcqContext(uncertainty=unc, cls=cls, rng=np.random.default_rng(0))
    ranked = EpigAcq().rank([4, 5], ctx)
    assert ranked[0] == 4  # representative uncertain point beats the uncertain outlier


def test_kcenter_select_adds_far_point():
    feats = np.array([[0, 0], [0.1, 0], [10, 10]], np.float32)
    chosen = [0]
    pick = diversity.kcenter_select(feats, k=1, chosen=chosen)
    assert pick == [2]  # farthest from the chosen point


def test_convergence_flags_plateau():
    history = [
        {"iou": 0.40, "correction_rate": 0.9, "acq_score": 1.0},
        {"iou": 0.55, "correction_rate": 0.7, "acq_score": 0.8},
        {"iou": 0.60, "correction_rate": 0.4, "acq_score": 0.4},
        {"iou": 0.605, "correction_rate": 0.3, "acq_score": 0.2},
        {"iou": 0.607, "correction_rate": 0.25, "acq_score": 0.15},
    ]
    out = convergence.composite(history, iou_eps=0.01, window=3)
    assert out["converged"] is True
    assert out["iou_plateau"] is True


def test_convergence_not_converged_early():
    history = [
        {"iou": 0.30, "correction_rate": 0.9, "acq_score": 1.0},
        {"iou": 0.45, "correction_rate": 0.8, "acq_score": 0.9},
    ]
    out = convergence.composite(history, iou_eps=0.01, window=3)
    assert out["converged"] is False
