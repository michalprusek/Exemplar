#!/usr/bin/env python
"""Visual demo of the pre-label pipeline: for a few test images, build a bank from a
handful of annotated images, then propose -> SAM refine -> per-instance polygons and
render a side-by-side figure (original | correspondence fg | SAM instances | overlay).

Run on tulen:
  ~/dinov3_env/bin/python scripts/demo_prelabel.py --cache /disk1/prusek/asg_cache \
      --data /disk1/prusek/dsb2018 --bank 10 --n 3 --out results/demo
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.transform import resize

from active_segmenter.config import ClusterConfig, EncoderConfig, MatchConfig, RefineConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval.datasets import load_dsb2018
from active_segmenter.membank.bank import MemoryBank
from active_segmenter.propose import correspondence as corr
from active_segmenter.propose import instances, polygon
from active_segmenter.refine import build_refiner

MC = MatchConfig(topk=5, bidirectional=False)


def color_instances(masks, shape):
    rng = np.random.default_rng(0)
    canvas = np.zeros((*shape, 3), np.float32)
    for m in masks:
        c = rng.uniform(0.3, 1.0, 3)
        canvas[m.mask] = c
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/asg_cache")
    ap.add_argument("--data", default="/tmp/dsb2018")
    ap.add_argument("--bank", type=int, default=10)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--out", default="results/demo")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = RunConfig(device="auto", cache_dir=args.cache, encoder=EncoderConfig(resolution=args.res))
    dev = cfg.device_resolved()
    train = load_dsb2018(args.data, "train", args.bank)
    test = load_dsb2018(args.data, "test", args.n)
    enc = CachedEncoder(cfg, dev, args.cache)
    refiner = build_refiner(RefineConfig(kind="sam"), dev)
    ccfg = ClusterConfig(score_thresh=0.0, min_patches=1, distance_threshold=1.5)

    bank = MemoryBank()
    for im, lbl in train:
        bank.add_from_annotation(enc.extract(im), (np.asarray(lbl) > 0).astype(int), {1: 1}, 0)

    for i, (im, lbl) in enumerate(test):
        feat = enc.extract(im)
        shape = np.asarray(lbl).shape
        score = corr.score_map(feat, bank, 1, MC, device=dev)
        fg = resize((score > 0).astype(np.float32), shape, order=0, mode="edge", anti_aliasing=False)
        grid = instances.decompose(score, ccfg, 1)
        refined = refiner.refine(im, instances.upsample_masks(grid, shape), feat_grid=feat)
        inst_rgb = color_instances(refined, shape)
        polys = [polygon.mask_to_polygon(m.mask) for m in refined]

        base = np.asarray(im, np.float32)
        base = (base - base.min()) / (np.ptp(base) + 1e-6)
        if base.ndim == 2:
            base = np.stack([base] * 3, -1)

        fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
        ax[0].imshow(base); ax[0].set_title("input image")
        ax[1].imshow(fg, cmap="magma"); ax[1].set_title("DINOv3 correspondence fg")
        ax[2].imshow(inst_rgb); ax[2].set_title(f"SAM-refined instances ({len(refined)})")
        ax[3].imshow(base)
        for p in polys:
            if p is not None:
                ax[3].plot(p[:, 1], p[:, 0], "-", lw=1.2)
        ax[3].set_title("polygons on image")
        for a in ax:
            a.axis("off")
        fig.suptitle(f"active-segmenter pre-label demo — bank={args.bank} annotated imgs, zero training",
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"demo_{i}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote demo_{i}.png ({len(refined)} instances)", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
