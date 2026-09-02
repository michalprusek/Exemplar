"""Tests for the single-pass streaming annotation selector.

The selector decides, for each arriving frame, whether to spend an annotation NOW —
no access to future frames — while pacing a fixed budget over the stream. Pure decision
logic, tested without any segmentation model (frame value is injected as a scalar).
"""
from active_segmenter.acquire.streaming import StreamingSelector


def test_warmup_frames_annotated_regardless_of_value():
    sel = StreamingSelector(budget=10, warmup=2, target_frames=100)
    assert sel.should_annotate(0.0) is True   # frame 1 — warmup seeds the committee
    assert sel.should_annotate(0.0) is True   # frame 2 — warmup


def test_never_exceeds_budget():
    sel = StreamingSelector(budget=3, warmup=0, target_frames=50)
    n = sum(sel.should_annotate(1.0) for _ in range(50))
    assert n == 3


def test_budget_exhausted_then_skips():
    sel = StreamingSelector(budget=1, warmup=1, target_frames=10)
    assert sel.should_annotate(0.5) is True    # warmup spends the only budget
    assert sel.should_annotate(9.9) is False   # budget gone -> skip even a huge value


def test_selects_high_value_over_low_under_budget():
    sel = StreamingSelector(budget=3, warmup=1, target_frames=6)
    stream = [0.5, 0.1, 0.1, 0.9, 0.9, 0.9]
    decisions = [sel.should_annotate(v) for v in stream]
    assert decisions[1] is False and decisions[2] is False   # low-value frames skipped
    assert decisions[3] is True and decisions[4] is True      # high-value frames taken
    assert sum(decisions) == 3                                # warmup + two high-value


def test_annotated_and_budget_bookkeeping():
    sel = StreamingSelector(budget=5, warmup=2, target_frames=10)
    for _ in range(10):
        sel.should_annotate(0.5)
    assert sel.annotated == 5
    assert sel.budget == 0
