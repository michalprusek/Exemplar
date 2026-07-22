"""Per-patch kNN correspondence pre-label (GPU-accelerated).

The spike's decisive finding: mean-top-k cosine to annotated fg exemplars beats an
averaged prototype (0.624 vs 0.510 IoU at res 672), because a single averaged
prototype is corrupted by any outlier while top-k stays close to the true nearest
exemplars. Score per query patch = ``mean top-k cosine to fg − mean top-k cosine to
bg``. Optional bidirectional (backward) matching suppresses false positives.

The matmul + top-k run on ``device`` via torch (CUDA on the GPU box, CPU locally),
because the AL loop re-scores the whole pool every round — on CPU the ``[Nq × Nfg]``
matmul dominates. Exemplar patch counts are capped (subsampled) for a bound.
"""
from __future__ import annotations

import numpy as np

from active_segmenter.config import MatchConfig


def _resolve_device(device):
    if device is not None:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _prep(arr: np.ndarray, d: int, cap: int) -> np.ndarray:
    if arr is None or arr.size == 0 or arr.shape[1] != d:
        return np.empty((0, d), np.float32)
    if len(arr) > cap:  # deterministic evenly-spaced subsample
        idx = np.linspace(0, len(arr) - 1, cap).astype(int)
        arr = arr[idx]
    return np.ascontiguousarray(arr, np.float32)


def _mean_topk_t(sim, k: int):
    """Mean of the top-``k`` values along dim 1 of a torch tensor ``[Nq, Nex]``."""
    import torch

    if sim.shape[1] == 0:
        return torch.full((sim.shape[0],), -1.0, device=sim.device)
    k = min(k, sim.shape[1])
    return torch.topk(sim, k, dim=1).values.mean(1)


def score_map(feat_grid, bank, class_id: int, cfg: MatchConfig, device=None) -> np.ndarray:
    import torch

    dev = _resolve_device(device)
    feat = np.asarray(feat_grid, np.float32)
    g0, g1, d = feat.shape
    q = torch.from_numpy(feat.reshape(-1, d)).to(dev)
    fg = _prep(bank.fg(class_id), d, cfg.max_fg)
    bg = _prep(bank.bg(class_id), d, cfg.max_bg)
    fgt = torch.from_numpy(fg).to(dev) if len(fg) else None
    bgt = torch.from_numpy(bg).to(dev) if len(bg) else None

    sim_fg = q @ fgt.T if fgt is not None else torch.empty((q.shape[0], 0), device=dev)
    sim_bg = q @ bgt.T if bgt is not None else torch.empty((q.shape[0], 0), device=dev)
    score = _mean_topk_t(sim_fg, cfg.topk) - _mean_topk_t(sim_bg, cfg.topk)

    if cfg.bidirectional and fgt is not None and sim_fg.shape[1] > 0:
        score = _bidirectional_t(sim_fg, score, cfg.topk)
    return score.reshape(g0, g1).cpu().numpy().astype(np.float32)


def _bidirectional_t(sim_fg, score, k: int):
    """Suppress query patches whose best fg exemplar does not rank them in its own
    top-``k`` query neighbours (backward match) -> pushed below zero (background)."""
    import torch

    best_ex = torch.argmax(sim_fg, dim=1)
    kk = min(k, sim_fg.shape[0])
    col_thresh = torch.topk(sim_fg, kk, dim=0).values[-1]  # k-th largest per exemplar
    q_best = sim_fg[torch.arange(sim_fg.shape[0], device=sim_fg.device), best_ex]
    confirmed = q_best >= col_thresh[best_ex]
    out = score.clone()
    unc = ~confirmed
    out[unc] = torch.minimum(out[unc], -torch.abs(out[unc]) - 1e-3)
    return out


def prelabel(feat_grid, bank, cfg: MatchConfig, device=None) -> dict[int, np.ndarray]:
    return {c: score_map(feat_grid, bank, c, cfg, device=device) for c in bank.classes()}


def margin(score_map_arr: np.ndarray) -> np.ndarray:
    """Per-patch |score|; small ⇒ ambiguous. Non-parametric uncertainty surrogate."""
    return np.abs(np.asarray(score_map_arr, np.float32))
