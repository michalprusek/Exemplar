"""Batch diversity — avoid picking k near-duplicate images in one round.

- ``kcenter_select``: greedy farthest-point given an already-chosen set (works on
  frozen CLS features).
- ``badge``: k-means++ seeding on gradient/embedding vectors (magnitude ~ model
  sensitivity), the BADGE recipe — diversity and uncertainty in one selection.
"""
from __future__ import annotations

import numpy as np


def kcenter_select(feats: np.ndarray, k: int, chosen: list[int]) -> list[int]:
    """Pick ``k`` new indices farthest (min-distance) from ``chosen``."""
    feats = np.asarray(feats, np.float32)
    n = len(feats)
    picks: list[int] = []
    if chosen:
        dist = np.min(
            np.linalg.norm(feats[:, None, :] - feats[chosen][None, :, :], axis=2), axis=1
        )
    else:
        dist = np.full(n, np.inf)
    dist[chosen] = -1
    for _ in range(k):
        j = int(np.argmax(dist))
        if dist[j] < 0:
            break
        picks.append(j)
        dist = np.minimum(dist, np.linalg.norm(feats - feats[j], axis=1))
        dist[picks] = -1
        dist[chosen] = -1
    return picks


def badge(grad_embeddings: np.ndarray, k: int, seed: int = 0) -> list[int]:
    """k-means++ seeding on gradient embeddings -> k diverse, high-magnitude picks."""
    x = np.asarray(grad_embeddings, np.float32)
    n = len(x)
    if k >= n:
        return list(range(n))
    rng = np.random.default_rng(seed)
    first = int(np.argmax(np.linalg.norm(x, axis=1)))  # highest-magnitude (most uncertain) seed
    picks = [first]
    d2 = np.linalg.norm(x - x[first], axis=1) ** 2
    for _ in range(k - 1):
        probs = d2 / (d2.sum() + 1e-12)
        nxt = int(rng.choice(n, p=probs))
        picks.append(nxt)
        d2 = np.minimum(d2, np.linalg.norm(x - x[nxt], axis=1) ** 2)
    return picks
