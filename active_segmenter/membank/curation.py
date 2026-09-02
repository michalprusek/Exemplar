"""k-center coreset selection.

Greedy farthest-point sampling: repeatedly pick the point farthest from the set
already chosen. This is the diversity primitive shared by bank curation (keep the
bank dataset-representative under a size cap) and batch-diverse acquisition —
they are the same operation on different point sets.
"""
from __future__ import annotations

import numpy as np


def kcenter(points: np.ndarray, k: int, seed: int = 0) -> list[int]:
    pts = np.asarray(points, np.float32)
    n = len(pts)
    if k >= n:
        return list(range(n))
    rng = np.random.default_rng(seed)
    first = int(rng.integers(n))
    chosen = [first]
    dist = np.linalg.norm(pts - pts[first], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(pts - pts[nxt], axis=1))
    return chosen
