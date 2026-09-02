#!/usr/bin/env python
"""Native-resolution TRAINING study — accuracy-preserving memory optimization, then reinvest the freed
headroom into resolution.

The classical guided-upsample study (CLASSICAL-UPSAMPLE-FINDINGS) found that changing the classical-bank
resolution only at INFERENCE regresses (tiled 0.471 < cap-1536 0.576 on HRF), because the head is
co-adapted to the resolution it TRAINED at. So the real lever is training at higher resolution — blocked
only by the O(out_hw²) activation memory that forced max_side=1536. Gradient checkpointing removes that
term EXACTLY (recompute native activations in backward → identical gradients, global dice intact; verified
bit-exact locally), so we can train — and match inference — at native resolution.

This script: (1) proves ckpt is accuracy-parity + memory-cheaper at max_side=1536; (2) sweeps max_side
{1536,2048,2560,native} with ckpt on, TRAIN and INFER matched at each, measuring accuracy and TRAIN PEAK
memory. Hypothesis: matched higher-res training co-adapts the head to native thin-vessel detail and
recovers/exceeds the 1536 baseline — the win the inference-only uncap could not deliver.

Run on tulen (GPU1 = dedicated A5000 24 GB; other job holds the A100):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. ~/dinov3_env/bin/python scripts/native_res_train_eval.py \
    --datasets hrf,rozpad --support 8 --pool 20 --test 24 --seeds 3 --cache /disk1/prusek/asg_cache_panel
"""
import argparse

import numpy as np

from active_segmenter.config import EncoderConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval.registry import PANEL, load_dataset
from active_segmenter.eval.scoring import primary_key, score_prediction
from active_segmenter.segment.base import LabeledExample


def _variants(max_sides, fine_grids):
    """(name, overrides). Two axes. Default = TRAIN+INFER resolution (max_side drives both classical
    paths): ms1536 (no ckpt) baseline, ms1536_ck parity/memory proof, then ckpt-on at each higher
    max_side. If --fine_grids given: sweep the DINOv3 fine-grid cap at a FIXED good out_hw (max_side=1536)
    → isolates SEMANTIC detail (less upsampling blur) from the out_hw penalty that dragged the max_side
    sweep down. fine_max_grid only downsamples (m>cap), so a cap above the native patch grid (HRF 219,
    rozpad 128) just yields native."""
    if fine_grids:
        return [(f"fg{g}", {"max_side": 1536, "grad_checkpoint": True, "fine_max_grid": g})
                for g in fine_grids]
    v = [("ms1536", {"max_side": 1536, "grad_checkpoint": False}),
         ("ms1536_ck", {"max_side": 1536, "grad_checkpoint": True})]
    for ms in max_sides:
        if ms != 1536:
            v.append((f"ms{ms}_ck", {"max_side": ms, "grad_checkpoint": True}))
    return v


def run_dataset(spec, args, dev):
    import torch
    from active_segmenter.segment.head_fusion_backend import HeadFusionBackend

    support_pool, test = load_dataset(spec, args.pool, args.test, seed=0)
    if len(support_pool) < args.support or len(test) < 2:
        raise RuntimeError(f"too few images (pool={len(support_pool)} test={len(test)})")
    cfg = RunConfig(device="auto", cache_dir=args.cache,
                    encoder=EncoderConfig(model_id=args.model, resolution=args.res))
    enc = CachedEncoder(cfg, dev, args.cache)
    pool = [LabeledExample(im, enc.extract(im), np.asarray(l)) for im, l in support_pool]
    tst = [LabeledExample(im, enc.extract(im), np.asarray(l)) for im, l in test]
    pk = primary_key(spec.metric)
    cuda = str(dev).startswith("cuda")

    # batch_size=1: item-wise gradient accumulation is mathematically identical to full-batch (one
    # optimizer step/epoch), so it changes ONLY memory, not accuracy — the tightest footprint per item.
    base = dict(device=dev, epochs=args.epochs, instance_mode="blob",
                scale_fusion=True, upsampler="guided", encoder=enc, batch_size=1)
    if args.uni:  # re-test the resolution axis under the UNIVERSAL recipe (topology+boundary loss, CLAHE,
        base.update(cldice=True, boundary_head=True, contrast_norm=True, fine_scales=True)  # finer scales
    out = {}
    for name, over in _variants(args.max_sides, args.fine_grids):
        try:
            be = HeadFusionBackend(**base, **over)
            prims, peak_gb = [], 0.0
            for s in range(args.seeds):
                be.head = None                                 # fresh head per seed; keep feature caches
                rng = np.random.default_rng(s)
                idx = list(rng.choice(len(pool), args.support, replace=False))
                torch.manual_seed(s)                           # paired init across variants per seed
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                be.fit([pool[i] for i in idx])
                if cuda:
                    peak_gb = max(peak_gb, torch.cuda.max_memory_allocated() / 2**30)
                rows = [score_prediction(spec.metric, be.foreground(ex.image, ex.feat_grid), ex.label_map)
                        for ex in tst]
                prims.append(float(np.mean([r[pk] for r in rows])))
            out[name] = (float(np.mean(prims)), float(np.std(prims)), peak_gb)
            print(f"  {name:>11}: {pk} {out[name][0]:.3f}±{out[name][1]:.3f}   train peak {peak_gb:5.1f} GB",
                  flush=True)
        except Exception as e:
            out[name] = (float("nan"), float("nan"), float("nan"))
            print(f"  {name:>11}: FAILED — {repr(e)[:110]}", flush=True)
        finally:
            torch.cuda.empty_cache() if cuda else None
    return out, pk, args.support, len(tst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="hrf,rozpad")
    ap.add_argument("--max_sides", default="1536,2048,2560,4096",
                    help="TRAIN+INFER resolution caps to sweep (≥native → full native; 4096 covers HRF 3504)")
    ap.add_argument("--fine_grids", default="",
                    help="if set (e.g. 96,128,160,219): sweep the DINOv3 fine-grid cap at fixed max_side=1536 "
                         "instead of the max_side sweep — isolates semantic detail from the out_hw penalty")
    ap.add_argument("--support", type=int, default=8)
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--test", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--uni", action="store_true",
                    help="use the UNIVERSAL recipe (clDice+boundary loss, CLAHE, finer scales) for the "
                         "resolution sweep — does a topology-aware loss change the native verdict?")
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--cache", default="/tmp/asg_cache_panel")
    args = ap.parse_args()
    args.max_sides = [int(c) for c in args.max_sides.split(",") if c]
    args.fine_grids = [int(c) for c in args.fine_grids.split(",") if c]

    dev = RunConfig(device="auto").device_resolved()
    names = [n for n in args.datasets.split(",") if n in PANEL]
    print(f"device={dev} datasets={names} max_sides={args.max_sides} support={args.support} "
          f"pool={args.pool} test={args.test} seeds={args.seeds}", flush=True)

    results = {}
    for name in names:
        print(f"\n[{name}] ({PANEL[name].note})", flush=True)
        try:
            results[name] = run_dataset(PANEL[name], args, dev)
        except Exception as e:
            print(f"  SKIP — {repr(e)[:140]}", flush=True)

    print("\n===== NATIVE-RES-TRAIN SUMMARY (accuracy + train peak mem; train=infer res matched) =====",
          flush=True)
    for name, (out, pk, ns, nt) in results.items():
        base = out.get("ms1536", (float("nan"),))[0]
        print(f"\n[{name}] metric={pk} support={ns} test={nt}  (baseline ms1536={base:.3f})", flush=True)
        print(f"  {'variant':>11} {'acc':>14} {'Δ vs 1536':>10} {'train GB':>9}")
        for vname, (mu, sd, gb) in out.items():
            print(f"  {vname:>11} {mu:>7.3f}±{sd:<5.3f} {mu - base:>+10.3f} {gb:>9.1f}", flush=True)
    print("NATIVE_RES_TRAIN_DONE")


if __name__ == "__main__":
    main()
