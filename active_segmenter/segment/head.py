"""Small trainable decoder over FROZEN DINOv3 patch features.

~2-5M params (default in_dim=1024, hidden=256). Two 3x3 conv blocks on the patch grid
plus a 1x1 classifier -> per-class logits at grid resolution. Deliberately light: the
labeled set is tiny, and the frozen DINOv3 features already carry the semantics — the
head only has to carve a decision surface, and its weights/gradients are what Spec B's
acquisition will read.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DINOHead(nn.Module):
    def __init__(self, in_dim: int = 1024, hidden: int = 256, n_classes: int = 1):
        super().__init__()
        groups = min(8, hidden)  # GroupNorm needs groups <= channels (small hidden in tests)
        self.body = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 3, padding=1), nn.GroupNorm(groups, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(groups, hidden), nn.GELU(),
        )
        self.classifier = nn.Conv2d(hidden, n_classes, 1)

    def forward(self, feat_grid_bchw: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.body(feat_grid_bchw))

    def forward_with_penultimate(self, feat_grid_bchw: torch.Tensor):
        """Return (logits, penultimate) — the penultimate is the input to the 1x1 classifier.
        Because the classifier is 1x1, the loss-gradient w.r.t. its weights at a patch is
        ``(sigmoid(logit) - y) * penultimate`` — the closed-form BADGE gradient embedding the
        weight-coupled acquisition reads (no autograd needed)."""
        h = self.body(feat_grid_bchw)
        return self.classifier(h), h
