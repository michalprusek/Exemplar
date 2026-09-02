"""Multi-prototype support→query correspondence (Lever 1).

The single averaged fg/bg prototype (the current ``corr_prior`` channel) is diluted on
appearance-varied foreground (e.g. H&E nuclei): the mean sits between appearance modes and matches
neither well. Replace it with k-means centroids and a MAX-POOLED cosine correspondence
``max_k cos(x,fg_k) − max_j cos(x,bg_j)`` — ONE channel, so the 1×1 fusion width is unchanged and the
EGL closed-form is untouched. ``k=j=1`` reduces EXACTLY to the single-prototype channel (parity).

The proven in-repo kNN ``propose.correspondence.score_map`` (mean-top-k over ALL exemplar patches) is
the non-parametric upper bound and the pre-screen reference; k-means centroids are the deployment-cheap
(bounded-cost) form fed into the head.
"""
from __future__ import annotations

import numpy as np


def _unit_rows(a: np.ndarray) -> np.ndarray:
    """L2-normalise each row to unit length (a near-zero row → zeros, never NaN)."""
    a = np.asarray(a, np.float32)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(n, 1e-6)


def single_proto_corr(feat_grid, fg_patches, bg_patches) -> np.ndarray:
    """Current head channel: ``cos(x, mean_fg) − cos(x, mean_bg)`` at grid resolution, prototypes
    L2-normed. ``feat_grid`` = [G0, G1, D] (unit per patch); patches = [N, D]."""
    g = np.asarray(feat_grid, np.float32)
    G0, G1, D = g.shape
    fg = np.asarray(fg_patches, np.float32).mean(0)
    bg = np.asarray(bg_patches, np.float32).mean(0)
    fg = fg / (np.linalg.norm(fg) + 1e-6)
    bg = bg / (np.linalg.norm(bg) + 1e-6)
    x = g.reshape(-1, D)
    corr = x @ fg - x @ bg
    return corr.reshape(G0, G1).astype(np.float32)


def kmeans_protos(patches: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """L2-normed k-means centroids of ``patches`` ([N, D]). ``k`` is clamped to N; ``k<=1`` (or a
    single cluster) returns the unit-normed mean direction — so ``kmeans_protos(p, 1)`` fed to
    ``multiproto_corr`` reproduces ``single_proto_corr`` exactly (parity). Never returns empty when
    ``N>=1``."""
    from sklearn.cluster import KMeans

    p = np.asarray(patches, np.float32)
    if p.ndim != 2 or len(p) == 0:
        raise ValueError("kmeans_protos: patches must be [N, D] with N>=1")
    kk = int(min(k, len(p)))
    if kk <= 1:
        return _unit_rows(p.mean(0, keepdims=True))
    km = KMeans(n_clusters=kk, n_init=3, random_state=seed).fit(_unit_rows(p))
    return _unit_rows(km.cluster_centers_)


def multiproto_corr(feat_grid, fg_protos, bg_protos) -> np.ndarray:
    """Max-pooled multi-prototype correspondence: ``max_k cos(x, fg_k) − max_j cos(x, bg_j)`` at grid
    resolution. ``feat_grid`` = [G0, G1, D]; ``fg_protos``/``bg_protos`` = [k, D]/[j, D] unit rows.
    With single mean-centroids this equals :func:`single_proto_corr`."""
    g = np.asarray(feat_grid, np.float32)
    G0, G1, D = g.shape
    x = g.reshape(-1, D)
    fg = _unit_rows(np.atleast_2d(fg_protos))
    bg = _unit_rows(np.atleast_2d(bg_protos))
    corr = (x @ fg.T).max(1) - (x @ bg.T).max(1)
    return corr.reshape(G0, G1).astype(np.float32)
