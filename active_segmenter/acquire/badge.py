"""Weight-coupled acquisition (Spec B) — BADGE / expected-gradient-length over the trainable
head's OWN gradients.

The fair-panel result showed the mismatch: geoloop/SMEC score candidates from the *frozen
correspondence* geometry, but the segmenter being trained is the *head* — so their signal is
not about what actually moves the model. BADGE (Ash et al. ICLR'20) fixes that by embedding
each candidate as the loss-gradient w.r.t. the head's last layer (under the head's pseudo-label)
and selecting a batch by k-means++ over those gradients: k-means++ jointly maximises gradient
MAGNITUDE (expected-gradient-length — uncertain/high-impact images) and DIVERSITY (spread in
gradient space), the two things a good AL batch needs, with no separate uncertainty term.

Pure functions over precomputed embeddings (the head's ``grad_embedding``), so testable without
the model. ``badge_select`` returns pool indices; ``egl_scores`` is the scalar ||grad|| ranking.
"""
from __future__ import annotations

import numpy as np


def egl_scores(embeddings: dict) -> dict:
    """Expected-gradient-length: ``{idx: ||grad_embedding||}``. Higher = this image would move
    the head's weights more (uncertain / high-impact). A cheap 1-D weight-coupled signal."""
    return {i: float(np.linalg.norm(np.asarray(g, np.float32))) for i, g in embeddings.items()}


def badge_select(embeddings: dict, k: int, seed: int = 0) -> list[int]:
    """k-means++ seeding over gradient embeddings — the BADGE batch. Picks the first centre by
    largest gradient norm (most impactful), then each next centre with probability ∝ squared
    distance to the nearest chosen centre (diversity in gradient space). Deterministic given
    ``seed``. Returns ``k`` distinct pool indices."""
    idxs = list(embeddings)
    if k >= len(idxs):
        return idxs
    X = np.stack([np.asarray(embeddings[i], np.float32) for i in idxs])   # [N, D]
    rng = np.random.default_rng(seed)
    first = int(np.argmax(np.linalg.norm(X, axis=1)))                     # highest-EGL seed
    chosen = [first]
    d2 = ((X - X[first]) ** 2).sum(1)                                     # nearest-centre dist²
    for _ in range(1, k):
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(len(idxs), 1 / len(idxs))
        nxt = int(rng.choice(len(idxs), p=probs))
        while nxt in chosen:                                             # avoid duplicates
            nxt = int(rng.choice(len(idxs), p=probs))
        chosen.append(nxt)
        d2 = np.minimum(d2, ((X - X[nxt]) ** 2).sum(1))
    return [idxs[c] for c in chosen]
