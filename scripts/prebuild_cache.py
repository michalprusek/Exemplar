#!/usr/bin/env python
"""Pre-extract DINOv3 features for every campaign image into ONE shared cache.

Features depend only on (image bytes, encoder config) -- not on K, seed or method -- so every
DINOv3-based run in the campaign can read this cache instead of recomputing. Building it once up
front also removes the documented failure mode of many concurrent jobs WRITING a shared cache
(that race previously produced irreproducible numbers and a fake "hrf regression").

Resolutions must cover every resolution the campaign actually launches, because ``cache_tag``
encodes resolution: the in-context methods run at 672 and INSID3 runs at its documented best
config of 1024. A resolution missing here is not an error, it is a silent 100% miss rate that
puts the concurrent writers back.
"""
from __future__ import annotations

import argparse
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--test", type=int, default=10000)
    ap.add_argument("--res", default="672,1024",
                    help="comma-separated encoder resolutions to prebuild (must cover every "
                         "resolution the campaign launches; 672 = in-context, 1024 = INSID3)")
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--pool_override", default="",
                    help="comma-separated name=pool overrides, e.g. ctc_u373=15,fisbe=16")
    args = ap.parse_args()

    from active_segmenter.config import EncoderConfig, RunConfig
    from active_segmenter.encoder.cached import CachedEncoder
    from active_segmenter.eval.registry import PANEL, load_dataset

    names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [n for n in names if n not in PANEL]
    if unknown:
        raise SystemExit(f"unknown dataset(s) {unknown}; known: {sorted(PANEL)}")
    resolutions = [int(r) for r in args.res.split(",") if r.strip()]

    overrides = {}
    for item in [x for x in args.pool_override.split(",") if x.strip()]:
        k, _, v = item.partition("=")
        overrides[k.strip()] = int(v)

    total = 0
    for res in resolutions:
        cfg = RunConfig(device="auto", cache_dir=args.cache,
                        encoder=EncoderConfig(model_id=args.model, resolution=res))
        dev = cfg.device_resolved()
        enc = CachedEncoder(cfg, dev, args.cache)
        for name in names:
            spec = PANEL[name]
            pool_n = overrides.get(name, args.pool)
            pool, test = load_dataset(spec, pool_n, args.test, seed=0)
            t0 = time.time()
            for im, _ in list(pool) + list(test):
                enc.extract(im)
                total += 1
            print(f"  [res{res}][{name}] cached {len(pool) + len(test)} images "
                  f"({len(pool)} pool + {len(test)} test) in {time.time() - t0:.0f}s", flush=True)
    if total == 0:
        raise SystemExit("prebuilt nothing — refusing to report a cache that would be 100% miss")
    print(f"PREBUILD_DONE {total} extractions", flush=True)


if __name__ == "__main__":
    sys.exit(main())
