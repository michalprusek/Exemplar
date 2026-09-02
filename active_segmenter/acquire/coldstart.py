"""Cold-start seed selection (round 0, no uncertainty yet).

TypiClust: cluster the pool's CLS embeddings into k clusters and, from each, pick
the most TYPICAL (highest local density) sample — diverse across clusters, typical
within. ProbCover: greedily cover the embedding space with balls of a fixed radius.
Both avoid the round-0 uncertainty pathology (uncertainty is meaningless with an
empty bank).
"""
from __future__ import annotations

import numpy as np


def _typicality(cls: np.ndarray, idx: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    """Local density = inverse mean distance to the ``n_neighbors`` nearest points
    within ``idx`` (higher = more typical)."""
    pts = cls[idx]
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    k = min(n_neighbors + 1, len(idx))
    nn = np.sort(d, axis=1)[:, 1:k]  # drop self
    mean_d = nn.mean(1) if nn.shape[1] else np.ones(len(idx))
    return 1.0 / (mean_d + 1e-6)


def typiclust(cls_embeddings: np.ndarray, k: int, seed: int = 0) -> list[int]:
    from scipy.cluster.vq import kmeans2

    cls = np.asarray(cls_embeddings, np.float32)
    n = len(cls)
    if k >= n:
        return list(range(n))
    _, labels = kmeans2(cls, k, seed=seed, minit="++", missing="raise")
    picks = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        typ = _typicality(cls, idx)
        picks.append(int(idx[int(np.argmax(typ))]))
    return picks


def probcover(cls_embeddings: np.ndarray, k: int, radius: float = 0.5, seed: int = 0) -> list[int]:
    cls = np.asarray(cls_embeddings, np.float32)
    n = len(cls)
    if k >= n:
        return list(range(n))
    rng = np.random.default_rng(seed)
    covered = np.zeros(n, bool)
    picks: list[int] = []
    d = np.linalg.norm(cls[:, None, :] - cls[None, :, :], axis=2)
    for _ in range(k):
        gain = ((d <= radius) & ~covered[None, :]).sum(1)
        gain[picks] = -1
        if gain.max() <= 0:
            remaining = [i for i in range(n) if i not in picks]
            picks.append(int(rng.choice(remaining)))
        else:
            picks.append(int(np.argmax(gain)))
        covered |= d[picks[-1]] <= radius
    return picks
