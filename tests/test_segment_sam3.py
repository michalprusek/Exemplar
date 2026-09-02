import numpy as np

from active_segmenter.segment.base import BackendUnavailable
from active_segmenter.segment.sam3_backend import Sam3PcsBackend


def test_sam3_unavailable_when_no_python_bin(tmp_path):
    be = Sam3PcsBackend(python_bin=str(tmp_path / "does-not-exist"))
    assert be.available() is False


def test_sam3_predict_raises_when_unavailable(tmp_path):
    be = Sam3PcsBackend(python_bin=str(tmp_path / "nope"))
    try:
        be.predict(np.zeros((8, 8, 3), np.uint8), None)
        assert False
    except BackendUnavailable:
        pass


def test_sam3_score_map_shape_is_grid():
    be = Sam3PcsBackend(python_bin="/nonexistent")
    feat = np.zeros((5, 5, 8), np.float32)
    s = be.score_map(np.zeros((8, 8, 3)), feat)
    assert s.shape == (5, 5)
