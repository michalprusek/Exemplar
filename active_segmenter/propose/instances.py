"""Instance decomposition — the novel/risky piece.

Turns a per-class correspondence score map into INDEPENDENT per-instance masks by
clustering the above-threshold patches. Overlap is preserved by construction:
instances are formed independently per class and each is its own boolean mask —
they are never argmax'd into a shared label raster (the one operation that would
destroy amodal overlap). Benchmarked against a connected-components baseline.
"""
from __future__ import annotations

import numpy as np

from active_segmenter.config import ClusterConfig
from active_segmenter.types import InstanceMask


def _cluster_labels(coords: np.ndarray, cfg: ClusterConfig) -> np.ndarray:
    n = len(coords)
    if n == 1:
        return np.zeros(1, int)
    if cfg.algo == "hdbscan":
        try:
            from sklearn.cluster import HDBSCAN

            lab = HDBSCAN(min_cluster_size=max(2, cfg.min_patches)).fit_predict(coords)
            # HDBSCAN marks noise as -1; give each noise point its own singleton label
            nxt = lab.max() + 1 if lab.size and lab.max() >= 0 else 0
            for i in np.where(lab < 0)[0]:
                lab[i] = nxt
                nxt += 1
            return lab
        except Exception:
            pass  # fall through to agglomerative
    from sklearn.cluster import AgglomerativeClustering

    # single linkage: at a spatial distance_threshold this is connected-components-
    # with-tolerance (bridges sub-threshold gaps), the right primitive for grouping
    # contiguous patches. Feature dims (when appended) then split touching instances.
    return AgglomerativeClustering(
        n_clusters=None, distance_threshold=cfg.distance_threshold, linkage="single"
    ).fit_predict(coords)


def _grid_connected_labels(ys, xs, mask, distance_threshold):
    """Label fg patches by grid connectivity (== single-linkage at a spatial
    threshold, in O(n)). 8-connectivity for threshold >= sqrt(2); larger thresholds
    bridge gaps by dilation."""
    from scipy import ndimage

    struct = (np.ones((3, 3)) if distance_threshold >= np.sqrt(2)
              else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    m = np.asarray(mask, bool)
    if distance_threshold > 2.0:
        m = ndimage.binary_dilation(m, iterations=int(distance_threshold) - 1)
    lab, _ = ndimage.label(m, structure=struct)
    return lab[ys, xs]


def decompose(
    score_map: np.ndarray,
    cfg: ClusterConfig,
    class_id: int,
    feat_grid: np.ndarray | None = None,
) -> list[InstanceMask]:
    s = np.asarray(score_map, np.float32)
    g0, g1 = s.shape
    ys, xs = np.where(s > cfg.score_thresh)
    if len(ys) == 0:
        return []
    if not (cfg.use_features and feat_grid is not None):
        # xy-only single-linkage at a spatial threshold == connected components.
        # Compute it directly in O(n) via grid labeling (identical result, and far
        # faster than O(n^2) agglomerative when many patches pass the threshold).
        labels = _grid_connected_labels(ys, xs, s > cfg.score_thresh, cfg.distance_threshold)
        coords = np.stack([ys, xs], axis=1).astype(np.float32)
    else:
        coords = np.stack([ys, xs], axis=1).astype(np.float32)
        # joint (xy + scaled features) space: features let touching-but-distinct
        # instances split, while xy keeps spatial coherence. distance_threshold is
        # in xy patch units; feature_gain calibrates the feature contribution.
        fpts = np.asarray(feat_grid, np.float32)[ys, xs] * cfg.feature_gain
        cluster_pts = np.concatenate([coords, fpts], axis=1)
        labels = _cluster_labels(cluster_pts, cfg)

    clusters = []
    for lab in np.unique(labels):
        pts = coords[labels == lab]
        if len(pts) < cfg.min_patches:
            continue
        mask = np.zeros((g0, g1), bool)
        idx = pts.astype(int)
        mask[idx[:, 0], idx[:, 1]] = True
        mean_score = float(s[idx[:, 0], idx[:, 1]].mean())
        clusters.append((len(pts), mean_score, mask))

    # cap: keep the largest clusters (then by score)
    clusters.sort(key=lambda c: (c[0], c[1]), reverse=True)
    clusters = clusters[: cfg.max_instances]

    return [
        InstanceMask(mask=m, points=None, class_id=class_id, instance_id=i, score=sc)
        for i, (_, sc, m) in enumerate(clusters)
    ]


def upsample_masks(instance_masks: list[InstanceMask], target_hw) -> list[InstanceMask]:
    """Resize each grid-resolution instance mask to native ``(H, W)`` (nearest)."""
    from skimage.transform import resize

    out = []
    for m in instance_masks:
        up = resize(m.mask.astype(np.float32), tuple(target_hw), order=0, mode="edge",
                    anti_aliasing=False) > 0.5
        out.append(InstanceMask(mask=up, points=None, class_id=m.class_id,
                                instance_id=m.instance_id, score=m.score))
    return out


def connected_components(score_map: np.ndarray, cfg: ClusterConfig, class_id: int) -> list[InstanceMask]:
    """Baseline: label the thresholded map by 8-connectivity. Destroys overlap
    (a pixel gets one label) — used only to measure the clustering's added value."""
    from scipy import ndimage

    s = np.asarray(score_map, np.float32) > cfg.score_thresh
    lab, n = ndimage.label(s, structure=np.ones((3, 3)))
    out = []
    for i in range(1, n + 1):
        mask = lab == i
        if mask.sum() < cfg.min_patches:
            continue
        out.append(InstanceMask(mask=mask, points=None, class_id=class_id,
                                instance_id=i - 1, score=1.0))
    return out
