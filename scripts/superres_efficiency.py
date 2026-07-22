"""Efficiency measurement for feature super-resolution (spec 2026-07-12, Task 8 step 3).

Measures per-image wall-clock and peak GPU memory of ``encoder.extract`` for the finer-grid
strategies, so the "finer AND cheaper than tiling" claim is proven or retracted, not asserted.
Uses the bare (uncached) encoder to time raw compute. GPU script — run on tulen.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from active_segmenter.config import EncoderConfig, RunConfig
from active_segmenter.encoder.factory import make_encoder
from active_segmenter.eval.datasets import load_fewshot

CONFIGS = {
    "sr1_res672": dict(resolution=672, superres_factor=1),
    "sr2_res672": dict(resolution=672, superres_factor=2),
    "sr4_res672": dict(resolution=672, superres_factor=4),
    "sr2jbu_res672": dict(resolution=672, superres_factor=2, jbu=True),
    "tile_res672": dict(resolution=672, tile=True),
    "single_res1024": dict(resolution=1024, superres_factor=1),  # ~INSID3 input resolution
}


def bench(enc, images):
    import torch

    for im in images[:2]:  # warmup
        enc.extract(im)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    grids = [enc.extract(im) for im in images]
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / len(images)
    peak = torch.cuda.max_memory_allocated() / 1e9
    return dt, peak, grids[0].shape


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    args = ap.parse_args()

    images = [im for im, _ in load_fewshot(args.data, "test", args.n)]
    dev = RunConfig(device="auto").device_resolved()
    print(f"data={args.data} n={len(images)} dev={dev} img0={np.asarray(images[0]).shape}",
          flush=True)
    print(f"{'config':>16} {'ms/img':>9} {'peak_GB':>9} {'grid':>16}", flush=True)
    for name, kw in CONFIGS.items():
        cfg = EncoderConfig(model_id=args.model, **kw)
        enc = make_encoder(cfg, dev)
        dt, peak, shape = bench(enc, images)
        print(f"{name:>16} {dt * 1e3:>9.1f} {peak:>9.2f} {str(shape):>16}", flush=True)
        del enc
        import torch

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
