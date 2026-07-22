"""SMEC — Support-Marginal Error Coverage.

A cold-start-through-high-budget acquisition for a FROZEN in-context segmenter whose
"model state" is its support set (the memory bank). Classical active learning
(uncertainty / BADGE / VeSSAL) assumes a *trainable* network with meaningful
last-layer gradients; here there is none — growing the support set IS the learning —
so the informative question is not "which frame most reduces the loss" but:

    "which unlabeled frame, once added to the support set, best COVERS the region of
     the dataset where the frozen model's predictions are currently unstable?"

SMEC answers it in two steps:

1. **Label-free error field** (``mask_disagreement``). For each pool frame it runs the
   SAME frozen segmenter several times, each conditioned on a different SUBSET of the
   currently-labeled exemplars — a committee whose members are support subsets, not
   separately-trained nets. The spread of the predicted masks (mean pairwise
   ``1 - IoU``) is a direct, model-specific estimate of the segmenter's error on that
   frame, needing no ground-truth label.

2. **Coverage-weighted facility location** (``coverage_weighted_scores``). It then
   picks the frame whose annotation best covers that error mass over the pool,
   downweighting regions already close to a labeled exemplar (a submodular
   representativeness term in DINOv3 embedding space).

Why the two regimes are handled automatically: at cold start the committee disagrees
mostly where the support set fails to *cover* the manifold, so SMEC behaves like a
diversity/coverage method (matching low-budget theory — TypiClust/ProbCover). As the
support set grows and covers the manifold, residual disagreement isolates genuinely
*hard* frames, so SMEC shifts toward uncertainty. No manual low/high-budget switch.
"""
from __future__ import annotations

import numpy as np


def mask_disagreement(masks) -> float:
    """Mean pairwise ``1 - IoU`` over a committee of boolean masks.

    ``0`` = the committee agrees perfectly, ``1`` = it disagrees completely. Two
    all-background masks count as agreement (IoU := 1). Fewer than two members ⇒ 0.
    """
    masks = [np.asarray(m, bool) for m in masks]
    if len(masks) < 2:
        return 0.0
    total, pairs = 0.0, 0
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            union = int((masks[i] | masks[j]).sum())
            inter = int((masks[i] & masks[j]).sum())
            iou = 1.0 if union == 0 else inter / union
            total += 1.0 - iou
            pairs += 1
    return total / pairs


def coverage_weighted_scores(cls, labeled, pool, err) -> dict:
    """Facility-location over the error field.

    ``score[x] = sum_u  err[u] * max(0, sim(x, u)) * novelty(u)`` where
    ``sim`` is cosine similarity of (unit-normalised) CLS embeddings and
    ``novelty(u) = max(0, 1 - max_{l in labeled} sim(u, l))`` downweights pool frames
    already covered by a labeled exemplar. Argmax = the frame most representative of
    the still-uncovered, high-error region of the dataset.
    """
    cls = np.asarray(cls, np.float32)
    pool = list(pool)
    if labeled:
        lab = cls[list(labeled)]                                   # [L, D]
        novelty = {u: max(0.0, 1.0 - float(np.max(cls[u] @ lab.T))) for u in pool}
    else:
        novelty = {u: 1.0 for u in pool}
    weighted_err = np.array([float(err[u]) * novelty[u] for u in pool], np.float32)  # [P]
    pool_feats = cls[pool]                                          # [P, D]
    out = {}
    for x in pool:
        sim = np.maximum(0.0, cls[x] @ pool_feats.T).astype(np.float32)  # [P]
        out[x] = float(np.dot(sim, weighted_err))
    return out


def smec_scores(pool, labeled, cls, predict_fn, *, n_committee=4, subset_frac=0.6,
                seed=0, min_support=2) -> dict:
    """Score pool frames by Support-Marginal Error Coverage; higher = annotate first.

    ``predict_fn(support_idxs, target_idx) -> bool mask`` is the frozen in-context
    segmenter: it returns the foreground prediction for ``target_idx`` conditioned on
    the support set ``support_idxs``. It is injected so the acquisition is testable
    without the heavy model.

    Below ``min_support`` labels there is no committee to disagree, so the error field
    is taken uniform and SMEC reduces to pure coverage (a cold-start seed, TypiClust-
    like). If the committee agrees everywhere (degenerate zero error field) it likewise
    falls back to coverage rather than picking arbitrarily.
    """
    labeled, pool = list(labeled), list(pool)

    if len(labeled) < min_support:
        err = {u: 1.0 for u in pool}
    else:
        rng = np.random.default_rng(seed)
        size = max(1, int(round(subset_frac * len(labeled))))
        subsets = [sorted(int(i) for i in rng.choice(labeled, size=size, replace=False))
                   for _ in range(n_committee)]
        err = {u: mask_disagreement([predict_fn(s, u) for s in subsets]) for u in pool}
        if max(err.values(), default=0.0) < 1e-12:   # model agrees everywhere
            err = {u: 1.0 for u in pool}

    return coverage_weighted_scores(cls, labeled, pool, err)


def zscore_fuse(score_dicts, weights=None) -> dict:
    """Fuse several ``{index: score}`` acquisition maps into one.

    Each map is z-normalised over the shared key set (so scorers on different scales
    combine fairly), then summed with optional per-scorer ``weights``. A map with no
    spread (all scores equal) carries no information and contributes nothing. Used to
    union complementary signals — e.g. SMEC's disagreement-coverage with proxy-EER's
    confidence-sharpening — which the AL literature finds beats either signal alone.
    """
    keys = list(score_dicts[0].keys())
    if weights is None:
        weights = [1.0] * len(score_dicts)
    fused = {k: 0.0 for k in keys}
    for w, sd in zip(weights, score_dicts):
        vals = np.array([sd[k] for k in keys], np.float64)
        sigma = vals.std()
        if sigma < 1e-12:      # no information in this scorer
            continue
        z = (vals - vals.mean()) / sigma
        for k, zi in zip(keys, z):
            fused[k] += w * float(zi)
    return fused
