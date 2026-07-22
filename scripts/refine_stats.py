"""Paired statistics for the refine stage (spec 2026-07-12 refine-stage, Task 6 step 3).

Per-test-image dsb2018 instance-AP for a set of (backend, refine) configs, then paired Wilcoxon
signed-rank + bootstrap 95% CI + Cliff's δ for the key contrasts (head+refine vs INSID3, and
head+refine vs head-no-refine). GPU script — run on tulen.
"""
from __future__ import annotations

import argparse

import numpy as np

from active_segmenter.config import EncoderConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval.registry import PANEL
from active_segmenter.eval.registry import load_dataset
from active_segmenter.eval.scoring import score_prediction
from active_segmenter.segment.base import LabeledExample


def per_image_ap(backend_name, refine, superres, dev, cache, n_sup, n_test):
    from scripts.al_testbed import make_backend

    spec = PANEL["dsb2018"]
    enc_cfg = EncoderConfig(resolution=672, superres_factor=superres)
    run_cfg = RunConfig(device="auto", cache_dir=cache, encoder=enc_cfg)
    support, test = load_dataset(spec, n_sup, n_test, seed=0)
    enc = CachedEncoder(run_cfg, dev, cache)
    sup = [LabeledExample(im, enc.extract(im), np.asarray(l)) for im, l in support]
    tst = [LabeledExample(im, enc.extract(im), np.asarray(l)) for im, l in test]
    be = make_backend(backend_name, run_cfg, dev, refine=refine)
    be.fit(sup)
    vals = []
    for ex in tst:
        fg = be.foreground(ex.image, ex.feat_grid)
        try:
            instances = [m.mask for m in be.predict(ex.image, ex.feat_grid)]
        except Exception:
            instances = []
        vals.append(score_prediction("instance_ap", fg, ex.label_map, instances)["ap"])
    return np.asarray(vals, float)


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


def main():
    from scipy.stats import wilcoxon

    ap = argparse.ArgumentParser()
    ap.add_argument("--support", type=int, default=16)
    ap.add_argument("--test", type=int, default=16)
    ap.add_argument("--superres", type=int, default=2)
    ap.add_argument("--cache", default="/disk2/prusek/asg_cache_superres2")
    args = ap.parse_args()

    dev = RunConfig(device="auto").device_resolved()
    print(f"dsb2018 instance-AP paired stats | support={args.support} test={args.test} "
          f"superres={args.superres} dev={dev}", flush=True)
    scores = {
        "head+none": per_image_ap("head", "none", args.superres, dev, args.cache, args.support, args.test),
        "head+amodal": per_image_ap("head", "amodal", args.superres, dev, args.cache, args.support, args.test),
        "insid3": per_image_ap("insid3", "none", 1, dev, args.cache, args.support, args.test),
    }
    print(f"{'config':>14} {'mean AP':>8}", flush=True)
    for k, v in scores.items():
        print(f"{k:>14} {v.mean():>8.3f}", flush=True)

    print(f"\n{'contrast':>26} {'delta':>7} {'wilcoxon_p':>11} {'boot95CI':>18} {'cliff_d':>8}",
          flush=True)
    for a, b in [("head+amodal", "insid3"), ("head+amodal", "head+none")]:
        d = scores[a] - scores[b]
        try:
            p = wilcoxon(scores[a], scores[b]).pvalue
        except ValueError:
            p = float("nan")
        lo, hi = bootstrap_ci(d)
        print(f"{a+' vs '+b:>26} {d.mean():>+7.3f} {p:>11.4f} "
              f"{f'[{lo:+.3f},{hi:+.3f}]':>18} {cliffs_delta(scores[a], scores[b]):>+8.3f}",
              flush=True)


if __name__ == "__main__":
    main()
