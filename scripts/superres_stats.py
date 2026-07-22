"""Paired statistics for feature super-resolution (spec 2026-07-12, Task 8 step 4).

For each dataset, computes the per-test-image designated metric for the head backend at a
baseline vs a super-resolved encoder config, then reports the paired Wilcoxon signed-rank
p-value, the mean delta with a bootstrap 95% CI, and Cliff's delta effect size. GPU script.

Head training has run-to-run variance; this is a single-seed paired test over test images
(the standard within-run paired comparison) — the write-up notes the caveat.
"""
from __future__ import annotations

import argparse

import numpy as np

from active_segmenter.config import EncoderConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval.registry import PANEL, load_dataset
from active_segmenter.eval.scoring import primary_key, score_prediction
from active_segmenter.segment.base import LabeledExample


def per_image_scores(spec, enc_cfg, dev, cache, n_sup, n_test):
    from scripts.al_testbed import make_backend

    run_cfg = RunConfig(device="auto", cache_dir=cache, encoder=enc_cfg)
    support, test = load_dataset(spec, n_sup, n_test, seed=0)
    enc = CachedEncoder(run_cfg, dev, cache)
    sup = [LabeledExample(im, enc.extract(im), np.asarray(l)) for im, l in support]
    tst = [LabeledExample(im, enc.extract(im), np.asarray(l)) for im, l in test]
    be = make_backend("head", run_cfg, dev)
    be.fit(sup)
    pk = primary_key(spec.metric)
    vals = []
    for ex in tst:
        fg = be.foreground(ex.image, ex.feat_grid)
        instances = None
        if spec.metric == "instance_ap":
            try:
                instances = [m.mask for m in be.predict(ex.image, ex.feat_grid)]
            except Exception:
                instances = []
        vals.append(score_prediction(spec.metric, fg, ex.label_map, instances)[pk])
    return np.asarray(vals, float), pk


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def bootstrap_ci(delta, iters=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(iters, len(delta)))
    means = delta[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="rozpad,dsb2018,spheroidj")
    ap.add_argument("--base", type=int, default=1, help="baseline superres factor")
    ap.add_argument("--ours", type=int, default=2, help="super-resolved factor")
    ap.add_argument("--jbu", action="store_true")
    ap.add_argument("--support", type=int, default=20)
    ap.add_argument("--test", type=int, default=24)
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--cache", default="/disk1/prusek/asg_cache_superres")
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    args = ap.parse_args()

    from scipy.stats import wilcoxon

    dev = RunConfig(device="auto").device_resolved()
    print(f"paired head-backend stats: sr{args.ours}{'+jbu' if args.jbu else ''} vs sr{args.base} "
          f"| support={args.support} test={args.test} res={args.res} dev={dev}", flush=True)
    print(f"{'dataset':>12} {'metric':>12} {'base':>7} {'ours':>7} {'delta':>7} "
          f"{'wilcoxon_p':>11} {'boot95CI':>18} {'cliff_d':>8}", flush=True)
    for name in args.datasets.split(","):
        spec = PANEL[name]
        base_cfg = EncoderConfig(model_id=args.model, resolution=args.res, superres_factor=args.base)
        ours_cfg = EncoderConfig(model_id=args.model, resolution=args.res,
                                 superres_factor=args.ours, jbu=args.jbu)
        base, pk = per_image_scores(spec, base_cfg, dev, args.cache, args.support, args.test)
        ours, _ = per_image_scores(spec, ours_cfg, dev, args.cache, args.support, args.test)
        delta = ours - base
        try:
            p = wilcoxon(ours, base).pvalue
        except ValueError:
            p = float("nan")
        lo, hi = bootstrap_ci(delta)
        cd = cliffs_delta(ours, base)
        print(f"{name:>12} {pk:>12} {base.mean():>7.3f} {ours.mean():>7.3f} {delta.mean():>+7.3f} "
              f"{p:>11.4f} {f'[{lo:+.3f},{hi:+.3f}]':>18} {cd:>+8.3f}", flush=True)


if __name__ == "__main__":
    main()
