import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_dinohead_forward_shape():
    from active_segmenter.segment.head import DINOHead

    head = DINOHead(in_dim=8, hidden=16, n_classes=1)
    x = torch.randn(2, 8, 6, 6)  # [B, D, G, G]
    y = head(x)
    assert tuple(y.shape) == (2, 1, 6, 6)


def test_head_backend_overfits_two_examples():
    """After fit on 2 toy examples the head separates fg/bg on a held example better
    than chance — proves the trainable path learns (not just that it runs)."""
    from active_segmenter.segment.base import LabeledExample
    from active_segmenter.segment.head_backend import TrainableHeadBackend

    rng = np.random.default_rng(0)

    def make(fg_val):
        feat = rng.standard_normal((6, 6, 8)).astype(np.float32)
        lm = np.zeros((24, 24), int)
        feat[1:4, 1:4] += fg_val          # a feature signature for fg
        lm[4:16, 4:16] = 1
        return LabeledExample(np.zeros((24, 24)), feat, lm)

    sup = [make(3.0), make(3.0)]
    be = TrainableHeadBackend(device="cpu", in_dim=8, epochs=150)
    be.fit(sup)
    fg = be.foreground(np.zeros((24, 24)), sup[0].feat_grid)
    inter = np.logical_and(fg, sup[0].label_map > 0).sum()
    union = np.logical_or(fg, sup[0].label_map > 0).sum()
    iou = inter / max(1, union)
    assert iou > 0.4
