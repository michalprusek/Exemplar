"""Wall-clock cost of adapting to a new dataset, which is the number the deployed tool lives on.

For an interactive tool the relevant quantity is not accuracy per label but the LATENCY between a
biologist finishing a mask and seeing an updated prediction. That decides whether human-in-the-loop
annotation is a conversation or a coffee break, and it is the axis on which a frozen-backbone head
differs from a network trained per dataset by orders of magnitude rather than by percent.

Reports, on one GPU with nothing else running:

  fit      -- everything from a support set to a usable model: closed-form rules, prototypes, and
              the head's optimisation. The number a HITL loop pays per added mask.
  encode   -- one image through the frozen backbone. Paid once per image and cacheable, so in a
              loop over a fixed dataset it is amortised to zero. Reported BOTH cold (cache cleared)
              and warm, because the two differ by orders of magnitude and quoting the warm number as
              "the encoder cost" would understate a first run on new data.
  predict  -- one image from cached features to a foreground mask. The per-image cost of showing
              the user an updated result.

`encode` bypasses the cache and times the backbone itself, so it is the cost of a FIRST pass over
new data. Inside a HITL loop the pool is fixed and already encoded, so that cost is paid once and
the per-interaction cost is `fit` plus `predict`.

    ~/dinov3_env/bin/python scripts/timing_bench.py --datasets monuseg,drive
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

POOL = {"ctc_u373": 15, "isbi2012em": 16, "fisbe": 16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="monuseg,drive,spheroidj")
    ap.add_argument("--support", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--cache", default="/disk1/prusek/cache_final10")
    ap.add_argument("--method",
                    default="head_fusion_best_nobank_flat_es_ep500_wd4_do5_mix")
    args = ap.parse_args()

    import torch
    from al_testbed import make_backend
    from active_segmenter.config import RunConfig, EncoderConfig
    from active_segmenter.encoder.cached import CachedEncoder
    from active_segmenter.eval.registry import PANEL, load_dataset
    from active_segmenter.segment.base import LabeledExample, reset_backend_for_new_support

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RunConfig(device=dev, cache_dir=args.cache, encoder=EncoderConfig(resolution=args.res))
    enc = CachedEncoder(cfg, dev, args.cache)

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    print("  method: {}\n".format(args.method))
    print("  {:11s} {:>7s} {:>19s} {:>12s} {:>12s}".format(
        "dataset", "img px", "fit (s) med [min-max]", "encode (ms)", "predict (ms)"))

    ROWS = []
    for ds in args.datasets.split(","):
        pool, test = load_dataset(PANEL[ds], POOL.get(ds, 20), 10000, seed=0)
        side = int(max(np.asarray(pool[0][0]).shape[:2]))

        fits, encs, preds = [], [], []
        for r in range(args.repeats):
            idx = np.random.default_rng(r).choice(len(pool), args.support, replace=False)
            shots = [LabeledExample(pool[i][0], enc.extract(pool[i][0]), np.asarray(pool[i][1]))
                     for i in idx]
            be = make_backend(args.method, cfg, dev, enc=enc, support_k=args.support)
            reset_backend_for_new_support(be)

            sync(); t0 = time.perf_counter()
            be.fit(shots)
            sync(); fits.append(time.perf_counter() - t0)

            im = test[0][0]
            # COLD encode: the campaign already cached every test image on disk, so timing
            # extract() would measure a .npy read. Call the underlying encoder directly to get the
            # real backbone cost -- the first version reported 0.006 s and called it that.
            sync(); t0 = time.perf_counter()
            enc.enc.extract(im)
            sync(); encs.append(time.perf_counter() - t0)
            g = enc.extract(im)

            sync(); t0 = time.perf_counter()
            be.foreground(im, g)
            sync(); preds.append(time.perf_counter() - t0)

            del be
            if dev == "cuda":
                torch.cuda.empty_cache()

        print("  {:11s} {:>7d} {:>8.1f} [{:5.1f}-{:5.1f}] {:>12.0f} {:>12.1f}".format(
            ds, side, float(np.median(fits)), float(min(fits)), float(max(fits)),
            1000 * float(np.median(encs)), 1000 * float(np.median(preds))))
        ROWS.append((ds, side, float(np.median(fits)), float(min(fits)), float(max(fits)),
                     1000 * float(np.median(encs)), 1000 * float(np.median(preds))))

    _summary(ROWS)


def _summary(rows):
    """Print the range the paper sentence quotes, so the claim is copied rather than recalled."""
    if not rows:
        return
    lo, hi = min(r[3] for r in rows), max(r[4] for r in rows)
    e_lo, e_hi = min(r[5] for r in rows), max(r[5] for r in rows)
    p_lo, p_hi = min(r[6] for r in rows), max(r[6] for r in rows)
    print("\n  ACROSS {} DATASETS AND ALL DRAWS".format(len(rows)))
    print("    fit      {:.0f} to {:.0f} s".format(lo, hi))
    print("    encode   {:.0f} to {:.0f} ms per image".format(e_lo, e_hi))
    print("    predict  {:.1f} to {:.1f} ms per image".format(p_lo, p_hi))


if __name__ == "__main__":
    main()
