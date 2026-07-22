import numpy as np

from active_segmenter.acquire.support_loo import (
    propagate_errors_to_pool,
    support_loo_errors,
)


def test_loo_error_is_one_minus_iou_against_known_gt():
    # support s=0 predicts perfectly, s=1 predicts empty (error 1)
    gt = {0: np.array([[1, 1], [0, 0]], bool), 1: np.array([[1, 0], [0, 0]], bool)}

    def predict_fn(others, s):
        return gt[s] if s == 0 else np.zeros((2, 2), bool)

    errs = support_loo_errors([0, 1], predict_fn, lambda s: gt[s])
    assert abs(errs[0] - 0.0) < 1e-9      # perfect -> error 0
    assert abs(errs[1] - 1.0) < 1e-9      # empty pred vs non-empty gt -> IoU 0 -> error 1


def test_loo_leaves_the_target_out_of_its_own_support():
    seen = {}

    def predict_fn(others, s):
        seen[s] = list(others)
        return np.ones((2, 2), bool)

    support_loo_errors([0, 1, 2], predict_fn, lambda s: np.ones((2, 2), bool))
    assert 0 not in seen[0] and 1 not in seen[1] and 2 not in seen[2]
    assert sorted(seen[0]) == [1, 2]


def test_propagate_inherits_nearby_support_error():
    # candidate A is close to high-error support s1; B is close to low-error s0
    support_errs = {0: 0.0, 1: 1.0}
    cls = np.array([[1, 0], [0, 1], [0.9, 0.1], [0.1, 0.9]], np.float32)  # s0,s1,A,B
    scores = propagate_errors_to_pool([2, 3], support_errs, cls)
    assert scores[3] > scores[2]  # B (near high-error s1) scores higher
