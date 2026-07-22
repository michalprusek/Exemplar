"""Crop tiling + crop-level acquisition.

The deployment idea (user, 2026-07-12): instead of proposing whole images, tile every uploaded
image into fixed **encoder-native-resolution crops** and (a) let the annotator label only a few
*suitable* crops, (b) segment the whole dataset crop-by-crop at native res (no downscaling → small
objects / thin structures survive), then stitch. This module owns the two *acquisition* halves:

1. ``tile_grid`` — deterministic overlapping tiling geometry (shared by the segmenter for inference
   and by the proposer for the candidate pool).
2. ``propose_crops`` — which crops to show the annotator. Grounded in this session's findings:
   - **content filter** — drop flat-background crops (a crop of empty field teaches nothing); score
     by CLS distance from the pool's background prototype (the central/most-common embedding).
   - **coverage cold-start** — among content-bearing crops, pick DIVERSE ones by k-center on the
     crop CLS embeddings (model-free, works from the first label). Crop-level coverage matters *more*
     than image-level: one big image has many near-duplicate crops, so random crop picks are very
     redundant. See [[lowk-coverage-acquisition]].
   - **novelty** — when some crops are already labeled, seed k-center with them so new proposals are
     FAR from what's labeled (the OOD-as-AL-signal use, which worked on globally-diverse pools).
"""
from __future__ import annotations

import numpy as np


def _positions(n: int, crop: int, step: int) -> list[int]:
    if n <= crop:
        return [0]
    ps = list(range(0, n - crop + 1, step))
    if ps[-1] != n - crop:
        ps.append(n - crop)               # always cover the right/bottom edge
    return ps


def tile_grid(h: int, w: int, crop: int, overlap: float = 0.25) -> list[tuple[int, int]]:
    """Top-left (y, x) of every crop covering an h×w image with fractional ``overlap``. Overlap is
    what makes instance stitching safe (every object is whole in ≥1 crop) and lets the foreground be
    feather-blended across seams."""
    step = max(1, int(round(crop * (1.0 - overlap))))
    return [(y, x) for y in _positions(h, crop, step) for x in _positions(w, crop, step)]


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = x[None]
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def content_scores(feats) -> np.ndarray:
    """Model-free informativeness per crop = **spatial dispersion of its dense patch features**
    (structured content — objects, edges — has high patch-to-patch variance; a flat background crop
    is nearly constant → ~0). This replaces the earlier CLS-distance-from-median heuristic, which
    FAILED when foreground is the majority (then the median IS foreground, so "far from median"
    picked atypical *empty* crops). Variance makes no majority assumption. ``feats`` = iterable of
    per-crop dense grids [gh, gw, D]."""
    out = []
    for f in feats:
        F = np.asarray(f, np.float32)
        out.append(float(F.reshape(-1, F.shape[-1]).var(axis=0).mean()))
    s = np.asarray(out, np.float32)
    return s / (s.max() + 1e-12)                       # normalise to [0, 1]


def _kcenter(cls: np.ndarray, k: int, seed_pts: np.ndarray | None = None) -> list[int]:
    """Farthest-point (k-center) coverage over unit CLS. If ``seed_pts`` (already-labeled crops) is
    given, the first pick is the crop farthest from them (novelty); else start at the medoid so the
    single pick is the most central/representative crop."""
    X = _unit(cls)
    n = len(X)
    if k >= n:
        return list(range(n))
    if seed_pts is not None and len(seed_pts):
        mind = 1.0 - (X @ _unit(seed_pts).T).max(1)     # distance to the labeled set
    else:
        d0 = 1.0 - (X @ X.T)
        mind = d0[int(np.argmin(d0.sum(1)))].copy()      # distances from the medoid
    sel: list[int] = []
    for _ in range(k):
        i = int(np.argmax(mind))
        sel.append(i)
        mind = np.minimum(mind, 1.0 - (X @ X[i]))
    return sel


def propose_crops(crops_emb: np.ndarray, k: int, labeled_emb: np.ndarray | None = None,
                  content: np.ndarray | None = None, content_frac: float = 0.5) -> list[int]:
    """Propose ``k`` crop indices for the annotator to label.

    ``crops_emb`` — [N, D] embedding per candidate crop (CLS, or mean-pooled dense features).
    ``labeled_emb`` — embeddings of already-labeled crops (bias new picks to be novel); None cold.
    ``content`` — [N] precomputed content score (e.g. :func:`content_scores` on dense features); the
    top ``content_frac`` are kept before coverage selection so flat-background crops are dropped.
    Pass None to skip the content filter.

    Cold start → the k most diverse content-bearing crops (coverage). With ``labeled_emb`` → novel +
    diverse crops far from what's labeled. Returns indices into ``crops_emb``."""
    n = len(crops_emb)
    if k >= n:
        return list(range(n))
    keep = np.arange(n)
    if content is not None and content_frac < 1.0:
        m = max(k, int(round(n * content_frac)))
        keep = np.argsort(-np.asarray(content))[:m]        # top-content crops
    sel_local = _kcenter(crops_emb[keep], k, seed_pts=labeled_emb)
    return [int(keep[i]) for i in sel_local]
