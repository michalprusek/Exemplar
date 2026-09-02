#!/usr/bin/env python
"""Classical-bank guided-upsample study — is a CHEAP low-res classical bank + image-guided upsample as
accurate as the exact full-native tiled bank, at a fraction of the compute?

The winning `head_fusion_sf_gid` computes its 35-ch classical hyperbank capped at max_side=1536 → on
large images (HRF 3504) it loses the finest structures (thin vessels). `tile_classical=True` fixes this
EXACTLY (full-native tile-by-tile) but recomputes the expensive Frangi/LoG on ~25 overlapping tiles.
This study measures a third path: compute the bank once at a LOW cap, then lift it to native with a
parameter-free Fast Guided Filter guided by the native image (fast_guided_upsample). We sweep the cap and
report BOTH accuracy (vs the tiled ceiling and the cap-1536 floor) AND wall-clock split into the
classical-bank stage (where the variants differ) and the full foreground.

Scale-specificity is the risk: low-res Frangi cannot invent a sub-pixel vessel; image guidance only
sharpens edges of what already sampled. The cap→accuracy curve locates where that breaks.

Run on tulen (one dataset per process keeps native-res memory bounded):
  ~/dinov3_env/bin/python scripts/classical_upsample_eval.py --datasets hrf,rozpad \
    --support 8 --pool 20 --test 24 --seeds 3 --caps 384,512,768,1024 \
    --cache /disk1/prusek/asg_cache_panel
"""
import argparse
import time

import numpy as np

from active_segmenter.config import EncoderConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval.registry import PANEL, load_dataset
from active_segmenter.eval.scoring import primary_key, score_prediction
from active_segmenter.segment.base import LabeledExample


def _variants(caps):
    """Ordered (name, backend-kwarg-overrides). Common config = head_fusion_sf_gid; only the classical
    path differs. cap1536 = current floor, tiled = exact ceiling, gupN = low-res bank + guided upsample."""
    v = [("cap1536", {}), ("tiled", {"tile_classical": True})]
    for c in caps:
        v.append((f"gup{c}", {"classical_up": True, "classical_lowres_side": c}))
    return v


def _sync(dev):
    if str(dev).startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def _time_bank_and_total(be, tst, dev, n):
    """Wall-clock, seed-independent. Warm the fine-DINOv3 cache once, then per image measure (a) the COLD
    classical-bank stage in isolation and (b) the full foreground with the bank cold too (fine warm).
    _ccache is cleared before each measured call so the bank is genuinely recomputed."""
    for ex in tst[:n]:  # warm _fcache (fine branch) — constant across variants, must not skew the delta
        be.foreground(ex.image, ex.feat_grid)
    bank_ms, total_ms = [], []
    for ex in tst[:n]:
        be._ccache = {}
        _sync(dev); t0 = time.perf_counter()
        be._classical(ex.image, inference=True)
        _sync(dev); bank_ms.append((time.perf_counter() - t0) * 1e3)
        be._ccache = {}
        _sync(dev); t0 = time.perf_counter()
        be.foreground(ex.image, ex.feat_grid)
        _sync(dev); total_ms.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(bank_ms)), float(np.mean(total_ms))


def run_dataset(spec, args, dev):
    from active_segmenter.segment.head_fusion_backend import HeadFusionBackend

    pool_n = args.pool if args.pool > 0 else args.support * 2
    support_pool, test = load_dataset(spec, pool_n, args.test, seed=0)
    if len(support_pool) < args.support or len(test) < 2:
        raise RuntimeError(f"too few images (pool={len(support_pool)} test={len(test)})")
    cfg = RunConfig(device="auto", cache_dir=args.cache,
                    encoder=EncoderConfig(model_id=args.model, resolution=args.res))
    enc = CachedEncoder(cfg, dev, args.cache)
    pool = [LabeledExample(im, enc.extract(im), np.asarray(l)) for im, l in support_pool]
    tst = [LabeledExample(im, enc.extract(im), np.asarray(l)) for im, l in test]
    pk = primary_key(spec.metric)

    import torch
    base = dict(device=dev, epochs=args.epochs, max_side=1536, instance_mode="blob",
                scale_fusion=True, upsampler="guided", encoder=enc)
    out = {}
    for name, over in _variants(args.caps):
        try:  # one variant's OOM (native-res inference on a 24 GB card) must not sink the others
            be = HeadFusionBackend(**base, **over)
            prims = []
            for s in range(args.seeds):
                be.head = None                                 # fresh head per seed; keep feature caches
                rng = np.random.default_rng(s)
                idx = list(rng.choice(len(pool), args.support, replace=False))
                torch.manual_seed(s)                           # PAIR the head init across variants per
                be.fit([pool[i] for i in idx])                 # seed → controlled variant comparison
                rows = [score_prediction(spec.metric, be.foreground(ex.image, ex.feat_grid), ex.label_map)
                        for ex in tst]
                prims.append(float(np.mean([r[pk] for r in rows])))
            try:
                bank_ms, total_ms = _time_bank_and_total(be, tst, dev, min(args.timing_n, len(tst)))
            except Exception as e:  # timing must never sink the accuracy result
                bank_ms, total_ms = float("nan"), float("nan")
                print(f"    [{name}] timing failed: {repr(e)[:100]}", flush=True)
            out[name] = (float(np.mean(prims)), float(np.std(prims)), bank_ms, total_ms)
            print(f"  {name:>9}: {pk} {out[name][0]:.3f}±{out[name][1]:.3f}   "
                  f"bank {bank_ms:7.1f} ms   total {total_ms:7.1f} ms", flush=True)
        except Exception as e:
            out[name] = (float("nan"), float("nan"), float("nan"), float("nan"))
            print(f"  {name:>9}: FAILED — {repr(e)[:110]}", flush=True)
        finally:
            torch.cuda.empty_cache() if str(dev).startswith("cuda") else None
    return out, pk, args.support, len(tst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="hrf,rozpad")
    ap.add_argument("--caps", default="384,512,768,1024",
                    help="low-res classical-bank caps to sweep for the guided-upsample variant")
    ap.add_argument("--support", type=int, default=8)
    ap.add_argument("--pool", type=int, default=20, help="FIXED support pool (same for every seed)")
    ap.add_argument("--test", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--timing_n", type=int, default=6, help="#test images for the wall-clock pass")
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--cache", default="/tmp/asg_cache_panel")
    args = ap.parse_args()
    args.caps = [int(c) for c in args.caps.split(",") if c]

    dev = RunConfig(device="auto").device_resolved()
    names = [n for n in args.datasets.split(",") if n in PANEL]
    print(f"device={dev} datasets={names} caps={args.caps} support={args.support} "
          f"pool={args.pool} test={args.test} seeds={args.seeds}", flush=True)

    results = {}
    for name in names:
        print(f"\n[{name}] ({PANEL[name].note})", flush=True)
        try:
            results[name] = run_dataset(PANEL[name], args, dev)
        except Exception as e:
            print(f"  SKIP — {repr(e)[:140]}", flush=True)

    print("\n===== CLASSICAL-UPSAMPLE SUMMARY (accuracy vs the tiled ceiling; bank/total ms) =====",
          flush=True)
    for name, (out, pk, ns, nt) in results.items():
        ceil = out.get("tiled", (float("nan"),))[0]
        print(f"\n[{name}] metric={pk} support={ns} test={nt}  (tiled ceiling={ceil:.3f})", flush=True)
        print(f"  {'variant':>9} {'acc':>14} {'Δ vs tiled':>11} {'bank ms':>9} {'total ms':>9}")
        for vname, (mu, sd, bms, tms) in out.items():
            print(f"  {vname:>9} {mu:>7.3f}±{sd:<5.3f} {mu - ceil:>+11.3f} {bms:>9.1f} {tms:>9.1f}",
                  flush=True)
    print("CLASSICAL_UPSAMPLE_DONE")


if __name__ == "__main__":
    main()
