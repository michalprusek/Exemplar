"""Submodular batch selection — weighted facility location with the (1−1/e) guarantee.

Sequential AL re-scores the whole pool after every single label; a batch method picks K at
once, which is K× cheaper but risks selecting K near-duplicates. Facility location avoids that
by construction: maximizing ``f(S) = sum_i weights[i] * max_{s in S} sim[i, s]`` — how well the
selected set ``S`` *covers* the (error-)weighted pool — is monotone submodular, so the greedy
algorithm is within ``(1 − 1/e) ≈ 0.63`` of optimal and each greedy step adds the candidate
with the largest *marginal* coverage gain (so a near-duplicate of an already-picked item has
almost zero gain and is skipped). This is the theoretically-grounded batch term for geoloop:
``weights`` = the geoloop error/value field, ``sim`` = on-sphere cosine among pool items.
"""
from __future__ import annotations

import numpy as np


def facility_location_greedy(sim: np.ndarray, weights: np.ndarray, k: int) -> list[int]:
    """Greedily select ``k`` candidate columns maximizing weighted facility-location
    coverage. ``sim`` is ``[N_pool, N_cand]`` (coverage of pool item i by candidate j);
    ``weights`` is ``[N_pool]`` (per-pool-item importance, e.g. the error field). Returns the
    selected candidate indices (columns), in selection order."""
    sim = np.asarray(sim, np.float32)
    weights = np.asarray(weights, np.float32)
    n_pool, n_cand = sim.shape
    k = min(k, n_cand)
    covered = np.zeros(n_pool, np.float32)          # best coverage of each pool item so far
    selected: list[int] = []
    available = list(range(n_cand))
    for _ in range(k):
        best_gain, best_j = -np.inf, None
        for j in available:
            gain = float(np.dot(weights, np.maximum(covered, sim[:, j]) - covered))
            if gain > best_gain:
                best_gain, best_j = gain, j
        selected.append(best_j)
        covered = np.maximum(covered, sim[:, best_j])
        available.remove(best_j)
    return selected
