"""Failure-case analysis for the head_fusion segmenter (2026-07-12).

Fits head_fusion on support, predicts test, and for each test image records the score + image/GT
characteristics (GT object count, mean object area, foreground fraction, contrast, edge density),
then reports the WORST cases + Pearson correlation of score vs each characteristic (which property
predicts failure). Saves a montage of the worst cases (image | GT | prediction). GPU script.
"""
from __future__ import annotations

import argparse

import numpy as np


def _stats(image, label):
    from scipy import ndimage

    g = np.asarray(image, np.float32)
    if g.ndim == 3:
        g = g.mean(2)
    lab, n = ndimage.label(np.asarray(label) > 0)
    areas = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1)) if n else np.array([0.0])
    gx, gy = np.gradient(g)
    lo, hi = np.percentile(g, [5, 95])
    return {
        "gt_count": float(n),
        "mean_area": float(areas.mean()),
        "fg_frac": float((np.asarray(label) > 0).mean()),
        "contrast": float((hi - lo) / (abs(g.mean()) + 1e-6)),
        "edge_density": float(np.sqrt(gx ** 2 + gy ** 2).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rozpad")
    ap.add_argument("--backend", default="head_fusion")
    ap.add_argument("--support", type=int, default=10)
    ap.add_argument("--test", type=int, default=16)
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--cache", default="/disk2/prusek/asg_cache_superres2")
    ap.add_argument("--worst", type=int, default=4)
    ap.add_argument("--outfig", default="/disk2/prusek/failure_")
    args = ap.parse_args()

    from active_segmenter.config import EncoderConfig, RunConfig
    from active_segmenter.encoder.cached import CachedEncoder
    from active_segmenter.eval.registry import PANEL, load_dataset
    from active_segmenter.eval.scoring import primary_key, score_prediction
    from active_segmenter.segment.base import LabeledExample
    from scripts.al_testbed import make_backend

    spec = PANEL[args.dataset]
    dev = RunConfig(device="auto").device_resolved()
    cfg = RunConfig(device="auto", cache_dir=args.cache,
                    encoder=EncoderConfig(resolution=args.res, superres_factor=1))
    enc = CachedEncoder(cfg, dev, args.cache)
    support, test = load_dataset(spec, args.support, args.test, seed=0)
    sup = [LabeledExample(im, enc.extract(im), np.asarray(l)) for im, l in support]
    be = make_backend(args.backend, cfg, dev)
    be.fit(sup)
    pk = primary_key(spec.metric)

    rows = []
    for im, lab in test:
        lab = np.asarray(lab)
        feat = enc.extract(im)
        fg = be.foreground(im, feat)
        instances = None
        if spec.metric == "instance_ap":
            try:
                instances = [m.mask for m in be.predict(im, feat)]
            except Exception:
                instances = []
        sc = score_prediction(spec.metric, fg, lab, instances)[pk]
        rows.append((sc, _stats(im, lab), im, lab, fg))

    rows.sort(key=lambda r: r[0])
    scores = np.array([r[0] for r in rows])
    print(f"{args.dataset} / {args.backend}: {pk} mean {scores.mean():.3f} "
          f"min {scores.min():.3f} max {scores.max():.3f} (test={len(rows)})", flush=True)
    print("--- correlation of score vs image property (negative = property predicts FAILURE) ---",
          flush=True)
    keys = list(rows[0][1].keys())
    for k in keys:
        vals = np.array([r[1][k] for r in rows])
        c = np.corrcoef(scores, vals)[0, 1] if vals.std() > 0 else float("nan")
        print(f"  {k:>14}: corr={c:+.3f}   (worst-mean {np.mean([r[1][k] for r in rows[:args.worst]]):.2f} "
              f"vs best-mean {np.mean([r[1][k] for r in rows[-args.worst:]]):.2f})", flush=True)
    print("--- WORST cases ---", flush=True)
    for sc, st, *_ in rows[:args.worst]:
        print(f"  {pk}={sc:.3f}  {st}", flush=True)

    # montage of worst cases: image | GT | prediction
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        w = args.worst
        fig, ax = plt.subplots(w, 3, figsize=(9, 3 * w))
        for i, (sc, st, im, lab, fg) in enumerate(rows[:w]):
            g = np.asarray(im, np.float32)
            g = g.mean(2) if g.ndim == 3 else g
            for j, (img, title) in enumerate([(g, f"img ({pk}={sc:.2f})"),
                                              (np.asarray(lab) > 0, "GT"), (fg, "pred")]):
                a = ax[i, j] if w > 1 else ax[j]
                a.imshow(img, cmap="gray"); a.set_title(title, fontsize=8); a.axis("off")
        plt.tight_layout()
        outp = f"{args.outfig}{args.dataset}_{args.backend}.png"
        plt.savefig(outp, dpi=80)
        print(f"SAVED_FIG {outp}", flush=True)
    except Exception as e:
        print(f"fig skipped: {e}", flush=True)
    print("FAILURE_DONE", flush=True)


if __name__ == "__main__":
    main()
