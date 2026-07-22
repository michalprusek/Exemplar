"""`channel_separability` was extracted from `_choose_contrast_source` so the gate probe can measure
the rule the segmenter actually runs instead of re-deriving it. These pin the behaviour that was
extracted, since a silent change here reselects the classical bank's input channel on every stained
dataset and the only symptom is a slightly worse number.
"""
from __future__ import annotations

import numpy as np

from active_segmenter.segment.head_fusion_backend import channel_separability


class _Shot:
    __slots__ = ("image", "label_map")

    def __init__(self, image, label_map):
        self.image, self.label_map = image, label_map


def _shot(fg_rgb, bg_rgb, size=64):
    """One image with a central square of `fg_rgb` on a `bg_rgb` field, plus its mask."""
    img = np.full((size, size, 3), bg_rgb, dtype=np.float32)
    lab = np.zeros((size, size), dtype=np.int32)
    img[24:40, 24:40] = fg_rgb
    lab[24:40, 24:40] = 1
    return _Shot(img, lab)


def test_a_channel_carrying_the_contrast_outscores_channels_that_do_not():
    """Green differs by 200 between foreground and background; red and blue do not differ at all.

    The comparison is against R and B, deliberately NOT against gray. Gray can legitimately beat a
    single channel here: averaging three independently-noisy channels suppresses noise faster than
    it suppresses a signal present in only one, so gray's Fisher ratio can exceed G's. That is
    correct behaviour of the measure, and asserting otherwise would pin a bug rather than a feature.
    """
    shots = [_shot(fg_rgb=(50, 250, 50), bg_rgb=(50, 50, 50)) for _ in range(3)]
    scores, n_used = channel_separability(shots)
    assert n_used == 3
    g = float(np.median(scores["G"]))
    assert g > float(np.median(scores["R"]))
    assert g > float(np.median(scores["B"]))


def test_every_channel_is_scored_once_per_usable_image():
    """The decision drops any channel not scored on EVERY image, so the counts must line up: a
    short list silently disqualifies a channel rather than averaging it over a subset."""
    shots = [_shot((10, 200, 10), (10, 10, 10)) for _ in range(4)]
    scores, n_used = channel_separability(shots)
    assert n_used == 4
    assert all(len(v) == n_used for v in scores.values()), \
        {k: len(v) for k, v in scores.items()}


def test_degenerate_masks_are_excluded_from_the_sample():
    """An all-background or all-foreground mask has no fg/bg contrast to measure. Counting it would
    dilute the median with a meaningless value."""
    good = [_shot((10, 200, 10), (10, 10, 10)) for _ in range(2)]
    empty = _Shot(np.full((64, 64, 3), 10, np.float32), np.zeros((64, 64), np.int32))
    allfg = _Shot(np.full((64, 64, 3), 10, np.float32), np.ones((64, 64), np.int32))
    scores, n_used = channel_separability(good + [empty, allfg])
    assert n_used == 2, "degenerate masks must not enter the sample"
    assert all(len(v) == 2 for v in scores.values())


def _noisy_shot(fg_rgb, bg_rgb, sigma=6.0, size=64, seed=0):
    """The REALISTIC regime: real images have within-class variance, so the Fisher denominator is
    dominated by the variance term and the floor never binds. Tests written in the noiseless limit
    describe a degenerate case that does not occur in the data."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), bg_rgb, dtype=np.float32)
    lab = np.zeros((size, size), dtype=np.int32)
    img[24:40, 24:40] = fg_rgb
    lab[24:40, 24:40] = 1
    return _Shot(img + rng.normal(0, sigma, img.shape).astype(np.float32), lab)


def test_a_near_flat_channel_loses_to_a_separating_one_under_realistic_noise():
    """Red differs by ONE intensity unit between foreground and background; green differs by 200.

    This is the property the selection actually depends on, and it holds because the variance term
    dominates once the data has any noise. Asserting it in the NOISELESS limit does not work and the
    earlier version of this test was wrong to try: every channel is min-max normalised on the way in,
    so with zero within-class variance the floor caps every channel that spans the range at the same
    value and red TIES green at 1000. That tie is an artefact of the fixture, not a defect.
    """
    shots = [_noisy_shot((11, 210, 10), (10, 10, 10), seed=s) for s in range(4)]
    scores, _ = channel_separability(shots)
    r, g = float(np.median(scores["R"])), float(np.median(scores["G"]))
    assert g > r, f"a 1-unit channel outscored a 200-unit one under noise (R={r:.3g}, G={g:.3g})"


def test_the_variance_floor_bounds_an_otherwise_unbounded_ratio():
    """What the floor is actually for.

    With zero within-class variance a pure Fisher ratio divides by machine epsilon and returns an
    astronomically confident score that no real channel can compete with. The floor caps it. This
    asserts the BOUND, which is what the code guarantees, rather than an ordering, which it does not.
    """
    shots = [_shot(fg_rgb=(11, 210, 10), bg_rgb=(10, 10, 10)) for _ in range(3)]
    scores, _ = channel_separability(shots)
    r = float(np.median(scores["R"]))       # differs by 1 unit, zero variance within each class
    assert r <= 1001, f"near-flat channel produced an unbounded score: {r:.3g}"


def test_a_shape_mismatched_support_image_is_not_silently_tolerated():
    """A resize or registration bug in a loader makes image and mask disagree on H x W.

    `n_used` is incremented before any channel is scored, so EVERY channel including gray is skipped
    for that image; the eligibility filter downstream then disqualifies all of them and the bank
    silently falls back to grayscale for the whole dataset. This test pins the current (bad)
    behaviour so the fix is visible when it lands -- see the note in the campaign doc. It is
    deliberately an assertion about what happens today, not about what should happen.
    """
    good = [_shot((10, 200, 10), (10, 10, 10)) for _ in range(2)]
    # a NON-degenerate mask (so it passes the all-fg/all-bg check and is counted) whose shape
    # disagrees with its image -- the loader-bug case, not the empty-mask case
    bad_lab = np.zeros((64, 64), np.int32)
    bad_lab[10:20, 10:20] = 1
    bad = _Shot(np.full((32, 32, 3), 10, np.float32), bad_lab)
    scores, n_used = channel_separability(good + [bad])
    assert n_used == 3, "the mismatched image is counted as used"
    assert all(len(v) == 2 for v in scores.values()), \
        "but no channel was scored on it, so every channel is short by one"
    # and therefore the consumer's `len(v) == n_used` filter disqualifies every candidate
    assert not [k for k, v in scores.items() if len(v) == n_used], \
        "every channel is disqualified -> the whole dataset silently falls back to gray"


def test_the_colour_decision_has_exactly_one_definition():
    """`colour_gate` is shared with `scripts/gate_constants_probe.py`.

    The probe is the evidence behind the paper's "no per-dataset tuning" claim. It previously
    re-implemented the eligibility filter and the median/margin comparison by hand, so only the
    MEASUREMENT was shared and the DECISION could drift -- which is the exact defect the `thin_gate`
    extraction was made to remove, left in place for colour.
    """
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "scripts"
           / "gate_constants_probe.py").read_text()
    assert "colour_gate" in src, "the probe must import the shared decision"
    assert "* margin" not in src, "the probe is re-deriving the margin comparison again"


def test_colour_gate_keeps_gray_unless_a_channel_clears_the_margin():
    from active_segmenter.segment.head_fusion_backend import colour_gate
    # a colour channel only 2% better than gray does NOT clear a 1.05 margin
    choice, _ = colour_gate({"gray": [1.00, 1.00], "G": [1.02, 1.02]}, 2, 1.05)
    assert choice == "gray"
    choice, _ = colour_gate({"gray": [1.00, 1.00], "G": [1.40, 1.40]}, 2, 1.05)
    assert choice == "G"


def test_colour_gate_disqualifies_a_channel_missing_from_any_support_image():
    """Averaging a channel over a biased subset is how rgb2hed failing on the hard tiles would
    inflate hematoxylin against gray."""
    from active_segmenter.segment.head_fusion_backend import colour_gate
    choice, means = colour_gate({"gray": [1.0, 1.0], "hematoxylin": [9.0]}, 2, 1.05)
    assert choice == "gray", "a channel scored on 1 of 2 images must not win"
    assert "hematoxylin" not in means
