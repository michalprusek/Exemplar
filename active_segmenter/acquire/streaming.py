"""Single-pass streaming annotation selector for the frozen in-context segmenter.

The online / per-frame regime the biologist loop actually runs in: frames arrive one at
a time, and for each one the selector must decide — irrevocably, without seeing the
future — whether to spend one of a fixed annotation budget on it. Annotated frames enter
the support set (memory bank), so the frozen segmenter's predictions update after every
single frame with no gradient step.

The decision rule (a non-parametric analogue of VeSSAL's rate-matching streaming AL):

- The first ``warmup`` frames are always annotated, to seed the disagreement committee
  (you need a few labels before "how much does the support set disagree on this frame?"
  is even defined).
- After warmup, the frame's scalar ``value`` (computed elsewhere as committee
  disagreement x novelty against the CURRENT support set) is compared to the running
  distribution of values seen so far. It is annotated iff it lands in the top ``pace``
  fraction, where ``pace = budget_remaining / frames_remaining`` — so the budget is
  spread across the whole stream instead of being burned on the first few frames.
- Budget is a hard cap; once spent, everything is skipped.

This class is deliberately value-agnostic (the value is injected) so the streaming
policy is unit-testable without any segmentation model.
"""
from __future__ import annotations

import numpy as np


class StreamingSelector:
    def __init__(self, budget, *, warmup=2, target_frames=None, seed=0):
        self.budget = int(budget)          # annotations still affordable
        self.warmup = int(warmup)          # first N frames auto-annotated (seed committee)
        self.target_frames = target_frames  # expected stream length, for budget pacing
        self.seed = seed
        self.seen = 0                      # frames encountered so far
        self.annotated = 0                 # frames annotated so far
        self._values: list[float] = []     # running record of frame values (for the quantile)

    def should_annotate(self, value: float) -> bool:
        """Decide, at encounter time, whether to annotate the current frame."""
        self.seen += 1
        value = float(value)

        if self.budget <= 0:               # hard budget cap
            self._values.append(value)
            return False

        if self.annotated < self.warmup:   # bootstrap the committee
            self._values.append(value)
            self._spend()
            return True

        # budget pacing: fraction of remaining frames we can still afford to annotate
        if self.target_frames:
            remaining = max(1, int(self.target_frames) - self.seen + 1)
            pace = self.budget / remaining
        else:
            pace = 0.5
        pace = min(1.0, max(0.0, pace))

        prior = self._values
        thresh = float(np.quantile(prior, 1.0 - pace)) if prior else float("-inf")
        self._values.append(value)

        if value >= thresh:
            self._spend()
            return True
        return False

    def _spend(self):
        self.budget -= 1
        self.annotated += 1
