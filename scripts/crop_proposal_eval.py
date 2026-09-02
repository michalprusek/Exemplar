#!/usr/bin/env python
"""Validate crop-level acquisition (`acquire.crop_tiles.propose_crops`): does proposing crops by
content-filter + coverage beat proposing RANDOM crops, at low annotation budget K?

For each dataset: tile every image into encoder-res crops, encode each crop's CLS, and record its
GROUND-TRUTH foreground fraction (used only to score the proposal, never to select). Compare, over
many seeds, `propose_crops(K)` vs random-K crops on:
- **content** — mean GT-fg-fraction of the selected crops (higher = less annotator time on empty
  background); **hit@K** = fraction of selected crops that actually contain an object.
- **coverage** — mean nearest-neighbour cosine distance among the selected crops (higher = more
  diverse, less redundant — the crop-level analogue of the low-K coverage win).

A microscopy mosaic is mostly background, so random crop picks are largely empty + redundant; the
proposer should raise both content and coverage.

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. ~/dinov3_env/bin/python scripts/crop_proposal_eval.py \
       --datasets rozpad,dsb2018,hrf --crop 672 --ks 5,10,20 --cache /disk2/prusek/asg_cache_lowk
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from active_segmenter.acquire.crop_tiles import propose_crops, tile_grid
from active_segmenter.config import EncoderConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval.registry import PANEL, load_dataset


def _pad(c, crop):
    ph, pw = crop - c.shape[0], crop - c.shape[1]
    if ph or pw:
        c = np.pad(c, [(0, ph), (0, pw)] + ([(0, 0)] if c.ndim == 3 else []), mode="reflect")
    return c


def crops_of(pairs, enc, crop, overlap):
    """Per crop: mean-pooled dense-feature embedding (for coverage), patch-variance content score
    (for the fixed content filter), and GT foreground fraction (for scoring only)."""
    embs, cont, fg = [], [], []
    for im, l in pairs:
        a = np.asarray(im)
        lab = np.asarray(l) > 0
        h, w = a.shape[:2]
        for (y, x) in tile_grid(h, w, crop, overlap):
            f = np.asarray(enc.enc.extract(_pad(a[y:y + crop, x:x + crop], crop)), np.float32)
            F = f.reshape(-1, f.shape[-1])
            embs.append(F.mean(0))
            cont.append(float(F.var(0).mean()))
            fg.append(float(lab[y:y + crop, x:x + crop].mean()))
    c = np.asarray(cont, np.float32)
    return np.stack(embs).astype(np.float32), c / (c.max() + 1e-12), np.asarray(fg, np.float32)


def _coverage(cls, idx):
    X = np.asarray(cls, np.float32)[idx]
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    if len(X) < 2:
        return 0.0
    sim = X @ X.T
    np.fill_diagonal(sim, -1.0)
    return float((1.0 - sim.max(1)).mean())          # mean nearest-neighbour distance


def run_dataset(name, spec, args, dev):
    cfg = RunConfig(device="auto", cache_dir=args.cache,
                    encoder=EncoderConfig(model_id=args.model, resolution=args.crop))
    enc = CachedEncoder(cfg, dev, args.cache)
    pairs, _ = load_dataset(spec, args.images, 2, seed=0)
    cls, content, fg = crops_of(pairs, enc, args.crop, args.overlap)
    n = len(cls)
    hit_thr = 0.005
    print(f"\n[{name}] {len(pairs)} imgs -> {n} crops; "
          f"{100*float((fg > hit_thr).mean()):.0f}% of crops contain fg (base rate)", flush=True)
    rec = {}
    for k in args.ks:
        if k >= n:
            continue
        rng = np.random.default_rng(0)
        r_content, r_hit, r_cov = [], [], []
        for s in range(args.seeds):
            ridx = rng.choice(n, k, replace=False)
            r_content.append(float(fg[ridx].mean()))
            r_hit.append(float((fg[ridx] > hit_thr).mean()))
            r_cov.append(_coverage(cls, ridx))
        pidx = propose_crops(cls, k, content=content, content_frac=args.content_frac)  # cold start
        p_content = float(fg[pidx].mean())
        p_hit = float((np.asarray(fg)[pidx] > hit_thr).mean())
        p_cov = _coverage(cls, pidx)
        rec[k] = {"rand_content": float(np.mean(r_content)), "prop_content": p_content,
                  "rand_hit": float(np.mean(r_hit)), "prop_hit": p_hit,
                  "rand_cov": float(np.mean(r_cov)), "prop_cov": p_cov}
        print(f"  K={k:>2}: content rand {np.mean(r_content):.3f} -> prop {p_content:.3f} | "
              f"hit@K rand {100*np.mean(r_hit):.0f}% -> prop {100*p_hit:.0f}% | "
              f"coverage rand {np.mean(r_cov):.3f} -> prop {p_cov:.3f}", flush=True)
    return {"dataset": name, "n_crops": n, "ks": rec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="rozpad,dsb2018,hrf")
    ap.add_argument("--images", type=int, default=16, help="images to tile per dataset")
    ap.add_argument("--crop", type=int, default=672)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--ks", default="5,10,20")
    ap.add_argument("--seeds", type=int, default=20, help="random-baseline seeds")
    ap.add_argument("--content_frac", type=float, default=0.5)
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--cache", default="/disk2/prusek/asg_cache_lowk")
    ap.add_argument("--out", default="/disk2/prusek/crop_proposal_results.json")
    args = ap.parse_args()
    args.ks = [int(x) for x in args.ks.split(",")]
    dev = RunConfig(device="auto").device_resolved()
    names = [n for n in args.datasets.split(",") if n in PANEL]
    print(f"device={dev} datasets={names} crop={args.crop} overlap={args.overlap} ks={args.ks}",
          flush=True)
    out = []
    for name in names:
        try:
            out.append(run_dataset(name, PANEL[name], args, dev))
        except Exception as e:
            print(f"\n[{name}] SKIP — {repr(e)[:160]}", flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}\nCROPPROP_DONE", flush=True)


if __name__ == "__main__":
    main()
