"""Training-free pre-screen for Lever 1 (multi-prototype correspondence) — the FIRST gate.

For monuseg (TARGET) and spheroidj (CONTROL): take the K=8 support, leave-one-support-out, build
prototypes/bank from the other 7, score the held-out image's grid, and measure fg/bg separability
(AUROC + best-threshold fg-IoU at grid resolution) for three correspondence variants:

  * ``single`` — the CURRENT head channel: cos(x, mean_fg) − cos(x, mean_bg).
  * ``kmeans`` — the proposed head channel: max_k cos(x, fg_k) − max_j cos(x, bg_j), k-means centroids.
  * ``knn``    — the proven in-repo upper bound: mean-top-k cosine over ALL exemplar patches
                 (``propose.correspondence.score_map``; already 0.624 vs 0.510 IoU over an averaged
                 prototype per its docstring).

All three consume the SAME grid fg/bg patch split (``_mask_to_grid``), so the comparison is clean. No
head, no training — just the frozen DINOv3 coarse grid (exactly what the head's corr channel consumes).

GO iff on monuseg the kmeans and/or knn AUROC beats single by a clear margin AND spheroidj does not
regress. On NO-GO, do NOT wire the lever into the head — log the negative and stop Lever 1.

Run on tulen (needs the DINOv3 encoder):
    python scripts/mproto_prescreen.py --k 4 --topk 5 --res 672 --seed 0
    for k in 2 4 8; do python scripts/mproto_prescreen.py --k $k --seed 0; done
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path

from active_segmenter.config import EncoderConfig, MatchConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval.registry import PANEL, load_dataset
from active_segmenter.membank.bank import MemoryBank
from active_segmenter.propose import correspondence as corr
from active_segmenter.segment import multiproto as mp


def load_support_with_features(dataset: str, k_support: int, res: int, seed: int,
                               model: str, cache: str):
    """Return a list of (feat_grid[G,G,D], label_map[H,W]) for K=`k_support` support images, using the
    SAME multi-draw fixed-pool loader + cached DINOv3 encoder as scripts/sota_final.py (features match
    training exactly). superres stays OFF (the head's corr channel reads the coarse grid)."""
    spec = PANEL[dataset]
    dev = RunConfig(device="auto").device_resolved()
    cfg = RunConfig(device="auto", cache_dir=cache,
                    encoder=EncoderConfig(model_id=model, resolution=res))
    enc = CachedEncoder(cfg, dev, cache)
    support_pool, _test = load_dataset(spec, 20, 8, seed=0)                 # fixed pool (seed=0), like sota_final
    sub = list(np.random.default_rng(seed).choice(len(support_pool), k_support, replace=False))
    return [(enc.extract(support_pool[i][0]), np.asarray(support_pool[i][1])) for i in sub]


def _grid_fg_bool(label_map, gh, gw) -> np.ndarray:
    """Grid-resolution foreground mask via the SAME nearest downsampler the head's ``_build_prototypes``
    uses (skimage.resize order=0), so the pre-screen builds single/kmeans prototypes from the EXACT fg/bg
    patch split the trained head will use — the gate then validates what ships. Flattened to [G*G]."""
    from skimage.transform import resize
    m = resize(np.asarray(label_map) > 0, (gh, gw), order=0, preserve_range=True,
               anti_aliasing=False) > 0.5
    return m.reshape(-1)


def _sep(score_flat: np.ndarray, fg_flat: np.ndarray):
    """AUROC + best-fixed-threshold fg-IoU of a per-patch score vs the grid fg mask. NaN if degenerate."""
    if fg_flat.all() or not fg_flat.any():
        return np.nan, np.nan
    auroc = float(roc_auc_score(fg_flat.astype(int), score_flat))
    best = 0.0
    for t in np.quantile(score_flat, np.linspace(0.05, 0.95, 19)):
        pred = score_flat >= t
        inter = np.logical_and(pred, fg_flat).sum()
        union = np.logical_or(pred, fg_flat).sum()
        best = max(best, inter / max(union, 1))
    return auroc, float(best)


def prescreen(dataset, k, topk, res, seed, model, cache):
    support = load_support_with_features(dataset, 8, res, seed, model, cache)
    rows = {"single": [], "kmeans": [], "knn": []}
    for i in range(len(support)):
        held_grid, held_lm = support[i]
        others = support[:i] + support[i + 1:]
        G0, G1, D = held_grid.shape
        fg_stack, bg_stack = [], []
        bank = MemoryBank()
        for og, olm in others:
            gh, gw = og.shape[:2]
            fgb = _grid_fg_bool(olm, gh, gw)
            flat = og.reshape(-1, og.shape[-1])
            fg_stack.append(flat[fgb]); bg_stack.append(flat[~fgb])
            bank.add_from_grid_mask(og, fgb.reshape(gh, gw), 1, 0)          # kNN arm: same grid mask
        fg_p = np.concatenate(fg_stack); bg_p = np.concatenate(bg_stack)
        if len(fg_p) == 0 or len(bg_p) == 0:                               # degenerate support → skip fold
            continue
        fgflat = _grid_fg_bool(held_lm, G0, G1)
        s_single = mp.single_proto_corr(held_grid, fg_p, bg_p).reshape(-1)
        s_km = mp.multiproto_corr(held_grid, mp.kmeans_protos(fg_p, k),
                                  mp.kmeans_protos(bg_p, k)).reshape(-1)
        s_knn = corr.score_map(held_grid, bank, 1, MatchConfig(topk=topk, bidirectional=False)).reshape(-1)
        for name, s in (("single", s_single), ("kmeans", s_km), ("knn", s_knn)):
            rows[name].append(_sep(s, fgflat))
    return {n: (float(np.nanmean([a for a, _ in v])), float(np.nanmean([b for _, b in v])))
            for n, v in rows.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4, help="k-means prototypes per class (fg and bg)")
    ap.add_argument("--topk", type=int, default=5, help="kNN mean-top-k reference")
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--cache", default="/disk1/prusek/asg_cache_mproto_prescreen")
    ap.add_argument("--datasets", default="monuseg,spheroidj",
                    help="comma list; first is TARGET, rest CONTROL")
    a = ap.parse_args()
    dsets = [d for d in a.datasets.split(",") if d in PANEL]
    print(f"k-means k={a.k}  kNN topk={a.topk}  res={a.res}  seed={a.seed}")
    print(f"{'dataset':10} {'variant':8} {'AUROC':>7} {'bestIoU':>8}")
    verdict = {}
    for ds in dsets:
        r = prescreen(ds, a.k, a.topk, a.res, a.seed, a.model, a.cache)
        if not all(name in r for name in ("single", "kmeans", "knn")):   # every LOO fold degenerate-skipped
            print(f"{ds:10} DEGENERATE — all support folds all-fg/all-bg; cannot screen")
            verdict[ds] = None
            continue
        verdict[ds] = r
        for name in ("single", "kmeans", "knn"):
            au, iou = r[name]
            print(f"{ds:10} {name:8} {au:7.3f} {iou:8.3f}")
    tgt = verdict[dsets[0]]
    if tgt is None:
        print("\nVERDICT: NO-GO (target dataset degenerate — cannot screen)")
        return
    ctrls = [verdict[d] for d in dsets[1:] if verdict[d] is not None]
    # GATE on the KMEANS arm only — that is what the head ships (n_proto). kNN is the non-parametric
    # upper-bound REFERENCE (printed for headroom), NOT the deployed channel, so it must not flip GO.
    go_target = tgt["kmeans"][0] > tgt["single"][0] + 0.02
    go_ctrl = all(c["kmeans"][0] > c["single"][0] - 0.01 for c in ctrls)
    print(f"\nVERDICT: {'GO' if (go_target and go_ctrl) else 'NO-GO'}  "
          f"({dsets[0]} single→kmeans AUROC {tgt['single'][0]:.3f}→{tgt['kmeans'][0]:.3f} "
          f"[knn ref {tgt['knn'][0]:.3f}]; control ok={go_ctrl})")


if __name__ == "__main__":
    main()
