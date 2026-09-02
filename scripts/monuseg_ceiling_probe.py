#!/usr/bin/env python
"""Training-free pre-screen: is the MoNuSeg foreground ceiling LINEAR-HEAD-specific or FUNDAMENTAL?

The wall (recorded C6 / foreground-is-the-bottleneck): on dense H&E nuclei a LINEAR head on frozen
DINOv3 features plateaus at foreground IoU ~0.62; the oracle with perfect foreground reaches AP
0.886, so the whole gap is foreground quality (a diffuse bleed of predicted foreground into adjacent
same-stain tissue). Self-training, backbone swap, richer matching, losses, threshold, upsampler and
resolution each moved <=0.01.

This probe answers, WITHOUT training the real head and WITHOUT a GPU training loop, the one question
that decides whether any lever can help: does a NONLINEAR decision rule, or LOCAL-background
information, separate a nucleus from ADJACENT tissue in frozen-feature space better than a linear
rule does? It compares, evaluated on HELD-OUT QUERY images (real generalization, not support fit):

  * linear      -- a ridge-logistic probe on the raw features (proxy for the current 1x1 head).
  * knn         -- cosine k-nearest-neighbour vote against support cells (a NONLINEAR boundary).
  * mlp         -- a 1-hidden-layer probe (nonlinearity a linear head cannot express).
  * local_sub   -- linear on features with each cell's LOCAL-RING mean subtracted (new information:
                   the local background statistics the global head never sees).

The negatives are drawn from the LOCAL RING around each nucleus, not the whole image, because the
wall is specifically nucleus-vs-adjacent-tissue, not nucleus-vs-empty-slide. If a nonlinear or
local-aware rule clearly beats linear on query images (grid fg-IoU margin > ~0.03 reproduced across
seeds), a lever exists and is worth building. If all cluster near linear, the ceiling is a property
of the frozen features at K=8 and only backbone fine-tuning breaks it -- itself a useful, paper-worthy
finding.

Runs on one GPU for feature extraction only; the classifiers are closed-form / a few steps. Cheap.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch


def _grid_labels(feat_grid, label_map):
    """Downsample an image-resolution label map to the feature grid by nearest sampling.

    The grid may be non-square ([Gh, Gw, D]), so both axes are taken from the feature map, never
    assumed equal -- a square assumption silently mis-samples the label on non-square images.
    """
    Gh, Gw = feat_grid.shape[0], feat_grid.shape[1]
    lb = np.asarray(label_map)
    H, W = lb.shape[:2]
    ys = np.linspace(0, H - 1, Gh).astype(int)
    xs = np.linspace(0, W - 1, Gw).astype(int)
    return (lb[np.ix_(ys, xs)] > 0).astype(np.int64)


def _local_ring(fg_grid, iters=2):
    """Cells within `iters` of a foreground cell but not foreground -- the adjacent tissue."""
    from scipy.ndimage import binary_dilation
    ring = binary_dilation(fg_grid > 0, iterations=iters) & ~(fg_grid > 0)
    return ring


def _local_ring_mean(feat, fg_grid, iters=3):
    """Per-cell mean feature over a local window, used as a background estimate to subtract."""
    from scipy.ndimage import uniform_filter
    G, _, D = feat.shape
    w = 2 * iters + 1
    return np.stack([uniform_filter(feat[..., d], size=w, mode="reflect") for d in range(D)], -1)


def _sample(feat, fg_grid, ring, rng, n_fg=400, n_bg=400):
    fg_idx = np.argwhere(fg_grid > 0)
    bg_idx = np.argwhere(ring)
    if len(fg_idx) == 0 or len(bg_idx) == 0:
        return None
    fi = fg_idx[rng.choice(len(fg_idx), min(n_fg, len(fg_idx)), replace=False)]
    bi = bg_idx[rng.choice(len(bg_idx), min(n_bg, len(bg_idx)), replace=False)]
    X = np.concatenate([feat[fi[:, 0], fi[:, 1]], feat[bi[:, 0], bi[:, 1]]], 0)
    y = np.concatenate([np.ones(len(fi)), np.zeros(len(bi))])
    return X.astype(np.float32), y.astype(np.float32)


def _ridge_logistic(Xtr, ytr, Xte, dev, steps=300, lam=1e-2):
    """A linear probe: the ceiling we are trying to beat. Trained to convergence on support cells."""
    Xt = torch.tensor(Xtr, device=dev); yt = torch.tensor(ytr, device=dev)
    w = torch.zeros(Xt.shape[1], device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        z = Xt @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt) + lam * w.pow(2).sum()
        loss.backward(); opt.step()
    with torch.no_grad():
        return torch.sigmoid(torch.tensor(Xte, device=dev) @ w + b).cpu().numpy()


def _mlp(Xtr, ytr, Xte, dev, hidden=64, steps=400, lam=1e-3):
    """A 1-hidden-layer probe: nonlinearity a linear head cannot express."""
    Xt = torch.tensor(Xtr, device=dev); yt = torch.tensor(ytr, device=dev)
    net = torch.nn.Sequential(torch.nn.Linear(Xt.shape[1], hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=lam)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt).squeeze(-1), yt)
        loss.backward(); opt.step()
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(Xte, device=dev)).squeeze(-1)).cpu().numpy()


def _knn(Xtr, ytr, Xte, dev, k=25):
    """Cosine k-NN vote: a nonlinear boundary that uses the support cells directly."""
    A = torch.nn.functional.normalize(torch.tensor(Xtr, device=dev), dim=1)
    B = torch.nn.functional.normalize(torch.tensor(Xte, device=dev), dim=1)
    yt = torch.tensor(ytr, device=dev)
    out = np.empty(len(Xte), np.float32)
    for i in range(0, len(Xte), 4096):
        sims = B[i:i + 4096] @ A.T
        idx = sims.topk(min(k, A.shape[0]), dim=1).indices
        out[i:i + 4096] = yt[idx].mean(1).cpu().numpy()
    return out


def _kde(Xtr, ytr, Xte, dev, bw=0.1):
    """FROST-style nonparametric density-ratio readout (research RANK 1): model the fg and bg
    support cells as two clouds on the unit sphere and label each query cell by the ratio of
    cosine-kernel densities. A genuinely nonlinear (hyperspherical) boundary a linear head cannot
    express. Bandwidth is fixed; the point is whether this SHAPE of nonlinearity beats a linear rule
    or collapses to the same gain (the research's own open question about FROST vs multi-prototype)."""
    A = torch.nn.functional.normalize(torch.tensor(Xtr, device=dev), dim=1)
    B = torch.nn.functional.normalize(torch.tensor(Xte, device=dev), dim=1)
    yt = torch.tensor(ytr, device=dev)
    fg = A[yt > 0.5]; bg = A[yt <= 0.5]
    out = np.empty(len(Xte), np.float32)
    for i in range(0, len(Xte), 2048):
        q = B[i:i + 2048]
        d_fg = torch.exp((q @ fg.T) / bw).mean(1)
        d_bg = torch.exp((q @ bg.T) / bw).mean(1)
        out[i:i + 2048] = (d_fg / (d_fg + d_bg + 1e-9)).cpu().numpy()
    return out


def _grid_iou(prob, fg_grid, ring):
    """Best-threshold fg IoU at grid resolution, negatives restricted to the local ring (the wall)."""
    y = (fg_grid > 0).ravel().astype(bool)
    m = (fg_grid > 0) | ring                       # score only nucleus + adjacent tissue
    p = prob.reshape(fg_grid.shape)[m]; yy = y.reshape(fg_grid.shape)[m]
    best = 0.0
    for t in np.linspace(0.1, 0.9, 17):
        pred = p > t
        inter = (pred & yy).sum(); union = (pred | yy).sum()
        if union:
            best = max(best, inter / union)
    return float(best)


METHODS = ["linear", "knn", "mlp", "kde", "local_sub"]


def eval_dataset(enc, spec, args, dev, load_dataset):
    """Mean grid fg-IoU (vs local ring, on query) per method, averaged over seeds, for one encoder."""
    pool, test = load_dataset(spec, args.pool, args.query, seed=0)
    per_seed = {m: [] for m in METHODS}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        sup = [pool[i] for i in rng.choice(len(pool), min(args.support, len(pool)), replace=False)]
        Xtr, ytr, Xs, ys = [], [], [], []
        for im, lb in sup:
            fgrid = np.asarray(enc.extract(im), np.float32)
            fg = _grid_labels(fgrid, lb); ring = _local_ring(fg)
            s = _sample(fgrid, fg, ring, rng)
            if s is not None:
                Xtr.append(s[0]); ytr.append(s[1])
            s2 = _sample(fgrid - _local_ring_mean(fgrid, fg), fg, ring, rng)
            if s2 is not None:
                Xs.append(s2[0]); ys.append(s2[1])
        if not Xtr:
            continue
        Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
        Xs = np.concatenate(Xs) if Xs else Xtr; ys = np.concatenate(ys) if ys else ytr
        acc = {m: [] for m in METHODS}
        for im, lb in test:
            fgrid = np.asarray(enc.extract(im), np.float32)
            fg = _grid_labels(fgrid, lb); ring = _local_ring(fg)
            if fg.sum() == 0 or ring.sum() == 0:
                continue
            flat = fgrid.reshape(-1, fgrid.shape[-1])
            sub = (fgrid - _local_ring_mean(fgrid, fg)).reshape(-1, fgrid.shape[-1])
            preds = {"linear": _ridge_logistic(Xtr, ytr, flat, dev),
                     "knn": _knn(Xtr, ytr, flat, dev),
                     "mlp": _mlp(Xtr, ytr, flat, dev),
                     "kde": _kde(Xtr, ytr, flat, dev),
                     "local_sub": _ridge_logistic(Xs, ys, sub, dev)}
            for m in METHODS:
                acc[m].append(_grid_iou(preds[m], fg, ring))
        for m in METHODS:
            if acc[m]:
                per_seed[m].append(float(np.mean(acc[m])))
    return {m: (float(np.mean(v)) if v else None) for m, v in per_seed.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="monuseg")
    ap.add_argument("--control", default="dsb2018", help="a dataset where the wall is NOT present")
    ap.add_argument("--support", type=int, default=8)
    ap.add_argument("--query", type=int, default=15)
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--layers", default="-1", help="comma DINOv3 layers to sweep; -1=last (selection gap)")
    ap.add_argument("--cache", default="/scratch/prusek/cache_final10")
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    args = ap.parse_args()

    from active_segmenter.config import EncoderConfig, RunConfig
    from active_segmenter.encoder.cached import CachedEncoder
    from active_segmenter.eval.registry import PANEL, load_dataset

    dev = RunConfig(device="auto").device_resolved()
    layers = [int(x) for x in args.layers.split(",") if x.strip()]

    for ds in [args.dataset, args.control]:
        spec = PANEL[ds]
        print(f"\n===== {ds} (control={ds == args.control}) — grid fg-IoU vs LOCAL ring, on QUERY, "
              f"per DINOv3 layer =====", flush=True)
        best_by_layer = {}
        for L in layers:
            cfg = RunConfig(device="auto", cache_dir=args.cache,
                            encoder=EncoderConfig(model_id=args.model, resolution=args.res,
                                                  superres_factor=1, layer=L))
            enc = CachedEncoder(cfg, dev, args.cache)
            r = eval_dataset(enc, spec, args, dev, load_dataset)
            best_by_layer[L] = r
            print(f"  layer {L:>3}  " + "  ".join(
                f"{m}={r[m]:.3f}" for m in METHODS if r[m] is not None), flush=True)
        # Best (layer, method) vs the last-layer linear baseline — the selection+readout headroom.
        base = best_by_layer.get(-1, best_by_layer[layers[0]])["linear"] or 0.0
        cells = [(L, m, best_by_layer[L][m]) for L in layers for m in METHODS
                 if best_by_layer[L][m] is not None]
        bL, bm, bv = max(cells, key=lambda t: t[2])
        gain = bv - base
        if gain > 0.03:
            print(f"  VERDICT[{ds}]: best is layer {bL} / {bm} = {bv:.3f}, {gain:+.3f} over last-layer "
                  f"linear ({base:.3f}) -> a layer/readout lever may help. Worth a real smoke test.",
                  flush=True)
        else:
            print(f"  VERDICT[{ds}]: nothing beats last-layer linear by >0.03 (best layer {bL}/{bm} "
                  f"{gain:+.3f}) -> no accessible layer/readout headroom; supports 'fundamental'.",
                  flush=True)
    print("\nPROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
