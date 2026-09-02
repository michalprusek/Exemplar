"""Day-1 de-risk for feature super-resolution (spec 2026-07-12).

Falsifies the sub-patch translation-equivariance assumption BEFORE the full benchmark:

1. **Coherence** — average-pooling the ``factor×`` grid back to the base grid must reconstruct
   the plain ``factor=1`` grid (mean cosine near 1). If not, shift-merge is incoherent.
2. **Payoff** — frozen-correspondence fg-IoU / boundary-F must rise with ``factor`` on small
   objects.

KILL criterion: if ``factor=2`` does not beat ``factor=1`` on boundary-F, pivot to a learned
upsampler (spec §5) instead of proceeding to Task 8.

GPU script — run on tulen with ``PYTHONPATH=/disk1/prusek/active-segmenter``.
"""
from __future__ import annotations

import argparse

import numpy as np

from active_segmenter.config import EncoderConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval.datasets import load_fewshot
from active_segmenter.eval.scoring import score_prediction
from active_segmenter.segment.base import LabeledExample
from active_segmenter.segment.correspondence_backend import CorrespondenceBackend


def pooled_cosine(fine: np.ndarray, base: np.ndarray, factor: int) -> float:
    """Mean cosine between the factor× grid average-pooled back to base, and the plain base."""
    g0, g1, d = base.shape
    fine = fine[: g0 * factor, : g1 * factor]  # guard rounding
    pooled = fine.reshape(g0, factor, g1, factor, d).mean(axis=(1, 3))
    pooled /= np.maximum(np.linalg.norm(pooled, axis=2, keepdims=True), 1e-6)
    return float((pooled * base).sum(axis=2).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", default="/disk1/prusek/asg_cache_superres")
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--factors", default="1,2,4")
    ap.add_argument("--jbu", action="store_true")
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    args = ap.parse_args()

    factors = [int(x) for x in args.factors.split(",")]
    support = load_fewshot(args.data, "support", args.n)
    test = load_fewshot(args.data, "test", args.n)
    print(f"data={args.data} res={args.res} n_sup={len(support)} n_test={len(test)} jbu={args.jbu}",
          flush=True)
    print(f"{'factor':>6} {'grid':>10} {'coh_cos':>8} {'fg_iou':>8} {'bf':>8}", flush=True)

    base_feats: dict[int, np.ndarray] = {}
    for factor in factors:
        cfg = RunConfig(
            device="auto", cache_dir=args.cache,
            encoder=EncoderConfig(model_id=args.model, resolution=args.res,
                                  superres_factor=factor, jbu=args.jbu and factor > 1),
        )
        dev = cfg.device_resolved()
        enc = CachedEncoder(cfg, dev, args.cache)
        sup = [LabeledExample(im, enc.extract(im), np.asarray(l)) for im, l in support]
        tst_feat = [enc.extract(im) for im, _ in test]
        be = CorrespondenceBackend(cfg.match, cfg.cluster, device=dev)
        be.fit(sup)

        ious, bfs = [], []
        for (im, lab), feat in zip(test, tst_feat):
            fg = be.foreground(im, feat)
            r = score_prediction("fg_iou", fg, np.asarray(lab), None)
            ious.append(r["fg_iou"])
            bfs.append(r["bf"])

        coh = "-"
        if factor == 1:
            base_feats = {i: f for i, f in enumerate(tst_feat)}
        elif base_feats:
            cos = [pooled_cosine(f, base_feats[i], factor) for i, f in enumerate(tst_feat)]
            coh = f"{np.mean(cos):.3f}"
        g = tst_feat[0].shape[0]
        print(f"{factor:>6} {f'{g}x{g}':>10} {coh:>8} "
              f"{np.mean(ious):>8.3f} {np.mean(bfs):>8.3f}", flush=True)


if __name__ == "__main__":
    main()
