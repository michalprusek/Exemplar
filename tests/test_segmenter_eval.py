import numpy as np

from active_segmenter.config import ClusterConfig, MatchConfig
from active_segmenter.segment.base import LabeledExample
from active_segmenter.segment.correspondence_backend import CorrespondenceBackend
from scripts.segmenter_eval import evaluate_backend


def test_evaluate_backend_returns_metric_dict():
    rng = np.random.default_rng(0)

    def ex():
        f = rng.standard_normal((6, 6, 8)).astype(np.float32)
        lm = np.zeros((24, 24), int)
        lm[4:12, 4:12] = 1
        return LabeledExample(np.zeros((24, 24, 3), np.uint8), f, lm)

    sup, test = [ex()], [ex()]
    be = CorrespondenceBackend(
        MatchConfig(topk=3, bidirectional=False), ClusterConfig(min_patches=1), device="cpu"
    )
    out = evaluate_backend(be, sup, test)
    assert set(out) >= {"fg_iou", "ap", "bf"}
    assert 0.0 <= out["fg_iou"] <= 1.0
    assert 0.0 <= out["bf"] <= 1.0
