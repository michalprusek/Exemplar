"""Transductive pool-based acquisitions — select which images of the UPLOADED
dataset to annotate next (no extrapolation beyond the pool).

- TypiClust (Hacohen et al., ICML 2022): cluster the pool's frozen embeddings, then
  from the largest not-yet-covered cluster pick the most TYPICAL (densest) image —
  typical + diverse coverage of the dataset manifold, the canonical low-budget method.
- Proxy Expected-Error-Reduction: a one-step look-ahead made exact-and-cheap by the
  non-parametric bank — insert a candidate's PSEUDO-exemplar (the model's own current
  prediction) and measure how much it sharpens predictions across the pool. No test
  labels needed; purely transductive.
"""
from __future__ import annotations

import numpy as np

from active_segmenter.propose import correspondence as corr


def typicality(cls: np.ndarray, k: int = 20) -> np.ndarray:
    """Density = inverse mean L2 distance to the k nearest neighbours (higher =
    more typical). Computed once over the whole uploaded dataset's embeddings."""
    cls = np.asarray(cls, np.float32)
    n = len(cls)
    if n == 1:
        return np.array([1.0], np.float32)
    d = np.linalg.norm(cls[:, None, :] - cls[None, :, :], axis=2)
    kk = min(k, n - 1)
    nn = np.sort(d, axis=1)[:, 1:kk + 1]
    return (1.0 / (nn.mean(1) + 1e-5)).astype(np.float32)


def typiclust_rank_scores(cls, labeled, pool, typ, seed: int = 0) -> dict[int, float]:
    """Score pool candidates so argmax = the most typical image of the LARGEST
    not-yet-covered cluster. n_clusters grows with the label budget (nLabeled+1)."""
    from sklearn.cluster import KMeans

    cls = np.asarray(cls, np.float32)
    n = len(cls)
    k = min(max(len(labeled) + 1, 1), n)
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(cls)
    covered = {int(labels[i]) for i in labeled}
    sizes = np.bincount(labels, minlength=k)
    out = {}
    for i in pool:
        c = int(labels[i])
        uncovered = 0 if c in covered else 1
        # lexicographic: uncovered first, then larger cluster, then more typical
        out[i] = uncovered * 1e6 + float(sizes[c]) * 1e3 + float(typ[i])
    return out


def pool_confidence(eval_feats, bank, match_cfg, device) -> float:
    """Mean per-patch decision margin |score| over the eval pool — a label-free proxy
    for prediction quality (higher = more decisive fg/bg calls)."""
    if not bank.classes():
        return 0.0
    return float(np.mean([
        np.mean(np.abs(corr.score_map(f, bank, 1, match_cfg, device=device)))
        for f in eval_feats
    ]))


def proxy_eer_scores(pool, bank, feats, eval_feats, match_cfg, device) -> dict[int, float]:
    """For each candidate: insert its pseudo-exemplar (current prediction) and measure
    the pool-wide confidence gain. This is Expected Error Reduction with an EXACT,
    free retrain (the memory bank) instead of DAO's influence-function approximation."""
    base = pool_confidence(eval_feats, bank, match_cfg, device)
    out = {}
    for i in pool:
        s = corr.score_map(feats[i], bank, 1, match_cfg, device=device)
        b2 = bank.copy()
        b2.add_from_grid_mask(feats[i], s > 0, 1, 0)
        out[i] = pool_confidence(eval_feats, b2, match_cfg, device) - base
    return out
