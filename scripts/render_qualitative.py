"""Render a qualitative results panel for the paper: rows = representative morphologies, columns =
input / our prediction boundary (red) / ground-truth boundary (green). Fits best_v2 on K=8 per dataset,
predicts the first test image, and writes one composite PNG."""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from active_segmenter.config import EncoderConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval.registry import PANEL, load_dataset
from active_segmenter.segment.base import LabeledExample
from scripts.al_testbed import make_backend

DATASETS = [("spheroidj", "Spheroids"), ("monuseg", "H\\&E nuclei"),
            ("drive", "Retinal vessels"), ("microtubules", "Microtubules")]
MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
CACHE = "/disk1/prusek/asg_cache_qual"


def _disp(im):
    a = np.asarray(im, np.float32)
    a = a if a.ndim == 2 else a[..., :3]
    return (a - a.min()) / (np.ptp(a) + 1e-6)


def _boundary(mask):
    from skimage.segmentation import find_boundaries
    from scipy.ndimage import binary_dilation
    b = find_boundaries(np.asarray(mask) > 0, mode="outer")
    return binary_dilation(b, iterations=1)


def main():
    dev = RunConfig(device="auto").device_resolved()
    cfg = RunConfig(device="auto", cache_dir=CACHE, encoder=EncoderConfig(model_id=MODEL, resolution=672))
    enc = CachedEncoder(cfg, dev, CACHE)
    n = len(DATASETS)
    fig, axes = plt.subplots(n, 3, figsize=(5.4, 1.75 * n))
    for r, (ds, label) in enumerate(DATASETS):
        pool, test = load_dataset(PANEL[ds], 20, 24, seed=0)
        be = make_backend("head_fusion_best_cgate_film_nobank", cfg, dev, enc=enc)
        sub = list(np.random.default_rng(0).choice(len(pool), 8, replace=False))
        be.fit([LabeledExample(pool[i][0], enc.extract(pool[i][0]), np.asarray(pool[i][1])) for i in sub])
        im, gt = test[0]
        d = _disp(im)
        fg = np.asarray(be.foreground(im, enc.extract(im))) > 0
        cmap = "gray" if d.ndim == 2 else None
        for c, (mask, col) in enumerate([(None, None), (fg, (1, 0, 0)), (np.asarray(gt) > 0, (0, 0.8, 0))]):
            ax = axes[r, c]
            ax.imshow(d, cmap=cmap)
            if mask is not None:
                ov = np.zeros((*np.asarray(mask).shape, 4), np.float32)
                ov[_boundary(mask)] = (*col, 1.0)
                ax.imshow(ov)
            ax.set_xticks([]); ax.set_yticks([])
        axes[r, 0].set_ylabel(label, fontsize=8)
        if r == 0:
            for c, t in enumerate(("Input", "Ours", "Ground truth")):
                axes[r, c].set_title(t, fontsize=9)
    fig.subplots_adjust(wspace=0.04, hspace=0.06, left=0.06, right=0.995, top=0.94, bottom=0.005)
    fig.savefig("/disk1/prusek/active-segmenter/paper_qualitative.png", dpi=200, bbox_inches="tight")
    print("wrote paper_qualitative.png")


if __name__ == "__main__":
    main()
