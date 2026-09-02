import numpy as np

from active_segmenter.segment.base import (
    BackendUnavailable,
    LabeledExample,
    foreground_from_score,
)


def test_foreground_from_score_upsamples_and_thresholds():
    grid = np.array([[-1.0, 1.0], [1.0, -1.0]], np.float32)  # 2x2
    fg = foreground_from_score(grid, (4, 4), thresh=0.0)
    assert fg.shape == (4, 4) and fg.dtype == bool
    # top-right and bottom-left quadrants are foreground
    assert fg[0, 3] and fg[3, 0] and not fg[0, 0] and not fg[3, 3]


def test_labeled_example_and_unavailable_exist():
    ex = LabeledExample(
        image=np.zeros((4, 4)), feat_grid=np.zeros((2, 2, 3)), label_map=np.zeros((4, 4), int)
    )
    assert ex.image.shape == (4, 4)
    assert issubclass(BackendUnavailable, RuntimeError)
