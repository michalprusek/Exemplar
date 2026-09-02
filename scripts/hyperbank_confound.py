"""Generality probe for HyperBank fusion (spec 2026-07-12 hyperbank-fusion).

Fits the classical and fusion backends on CLEAN rozpad support, then evaluates fg-IoU on test
images with injected REGIONAL confounders — bright textured blobs placed in background (not in GT)
that classical filters may over-detect but the DINOv3 support-fg saliency scores low. If the fusion
(semantic-gated) backend suppresses those false positives, it beats classical-alone on confounded
data while staying ≈ equal on clean data (the safe-monotone + generality claim). GPU script.
"""
from __future__ import annotations

import argparse

import numpy as np

from active_segmenter.config import EncoderConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval.registry import PANEL, load_dataset
from active_segmenter.eval import metrics
from active_segmenter.segment.base import LabeledExample
from active_segmenter.segment.hyperbank_backend import HyperBankBackend


def inject_confounders(image, gt, n, rng, r_range=(24, 64)):
    """Add ``n`` bright textured discs in background (disjoint from GT foreground). Returns the
    confounded grayscale image; GT is unchanged (so detecting a confounder lowers IoU)."""
    img = np.asarray(image, np.float32).copy()
    if img.ndim == 3:
        img = img.mean(axis=2)
    h, w = img.shape[:2]
    gtb = np.asarray(gt) > 0
    yy, xx = np.mgrid[0:h, 0:w]
    hi = float(img.max())
    for _ in range(n):
        r = int(rng.integers(*r_range))
        cy, cx = int(rng.integers(r, h - r)), int(rng.integers(r, w - r))
        disc = (yy - cy) ** 2 + (xx - cx) ** 2 < r * r
        if (disc & gtb).any():
            continue
        # bright disc + high-freq speckle so LoG/structure filters respond
        speckle = rng.normal(0, 0.06 * hi, size=img.shape).astype(np.float32)
        img[disc] = np.clip(hi * 0.95 + speckle[disc], 0, hi)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--support", type=int, default=10)
    ap.add_argument("--test", type=int, default=16)
    ap.add_argument("--n_confound", type=int, default=6)
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--cache", default="/disk2/prusek/asg_cache_superres2")
    args = ap.parse_args()

    spec = PANEL["rozpad"]
    dev = RunConfig(device="auto").device_resolved()
    cfg = RunConfig(device="auto", cache_dir=args.cache,
                    encoder=EncoderConfig(resolution=args.res, superres_factor=1))
    enc = CachedEncoder(cfg, dev, args.cache)
    support, test = load_dataset(spec, args.support, args.test, seed=0)
    sup = [LabeledExample(im, enc.extract(im), np.asarray(l)) for im, l in support]
    tst = [(im, enc.extract(im), np.asarray(l)) for im, l in test]
    rng = np.random.default_rng(0)
    confounded = [inject_confounders(im, l, args.n_confound, rng) for im, _, l in tst]

    print(f"HyperBank fusion generality probe: support={len(sup)} test={len(tst)} "
          f"n_confound={args.n_confound} dev={dev}", flush=True)
    print(f"{'backend':>18} {'clean IoU':>10} {'confounded IoU':>15} {'Δ (drop)':>10}", flush=True)
    rows = {}
    for name, fusion in [("hyperbank", False), ("hyperbank_fusion", True)]:
        be = HyperBankBackend(device=dev, fusion=fusion)
        be.fit(sup)
        clean, conf = [], []
        for (im, feat, lab), cim in zip(tst, confounded):
            clean.append(metrics.foreground_iou(be.foreground(im, feat), lab))
            conf.append(metrics.foreground_iou(be.foreground(cim, feat), lab))
        clean, conf = np.array(clean), np.array(conf)
        rows[name] = (clean, conf)
        print(f"{name:>18} {clean.mean():>10.3f} {conf.mean():>15.3f} "
              f"{(clean.mean() - conf.mean()):>10.3f}", flush=True)

    # generality gain = does fusion drop LESS under confounders than classical?
    from scipy.stats import wilcoxon

    d_classical = rows["hyperbank"][0] - rows["hyperbank"][1]      # per-image drop, classical
    d_fusion = rows["hyperbank_fusion"][0] - rows["hyperbank_fusion"][1]  # drop, fusion
    conf_gain = rows["hyperbank_fusion"][1] - rows["hyperbank"][1]  # fusion − classical on confounded
    try:
        p = wilcoxon(rows["hyperbank_fusion"][1], rows["hyperbank"][1]).pvalue
    except ValueError:
        p = float("nan")
    print(f"\nconfounded: fusion − classical = {conf_gain.mean():+.3f} (Wilcoxon p={p:.4f})", flush=True)
    print(f"robustness: classical drop {d_classical.mean():.3f} vs fusion drop {d_fusion.mean():.3f}",
          flush=True)
    print("CONFOUND_DONE", flush=True)


if __name__ == "__main__":
    main()
