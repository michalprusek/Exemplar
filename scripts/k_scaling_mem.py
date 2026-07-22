#!/usr/bin/env python
"""K-scaling memory diagnostic — does gradient checkpointing let K=50/100 fit?

Claim under test: checkpointing shrinks the ACTIVATION term (K-independent, already bounded by the
batch_size=1 gradient accumulation), NOT the K-linear term — which is the SUPPORT INPUTS held resident on
GPU for speed (`fit` keeps all K items: the classical bank C0 ≈ 165 MB/large-image @1536² dominates). So
checkpointing should NOT change the K at which we OOM. This measures peak train memory vs K, ckpt off/on,
to confirm — and to locate the OOM point on the current GPU.

Images are replicated (distinct array copies → distinct id → the backend holds K distinct banks, the real
footprint) so we can reach K beyond the few-shot pool. Coarse features are content-cached by the encoder
(fast); the per-image classical/fine tensors are id-keyed (genuinely K distinct on GPU).

Run (GPU1 = 24 GB A5000 → directly tests the deployment card; GPU0 = 80 GB for true peaks):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. ~/dinov3_env/bin/python scripts/k_scaling_mem.py \
    --dataset rozpad --ks 8,25,50,100 --cache /disk1/prusek/asg_cache_panel
"""
import argparse

import numpy as np

from active_segmenter.config import EncoderConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval.registry import PANEL, load_dataset
from active_segmenter.segment.base import LabeledExample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rozpad", help="large-image dataset = worst case for the K term")
    ap.add_argument("--ks", default="8,25,50,100")
    ap.add_argument("--epochs", type=int, default=3, help="peak is reached in epoch 1; keep small")
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--cache", default="/tmp/asg_cache_panel")
    args = ap.parse_args()
    ks = [int(c) for c in args.ks.split(",") if c]

    import torch
    from active_segmenter.segment.head_fusion_backend import HeadFusionBackend
    dev = RunConfig(device="auto").device_resolved()
    spec = PANEL[args.dataset]
    base, _ = load_dataset(spec, 20, 4, seed=0)
    if not base:
        raise RuntimeError("no images")
    cfg = RunConfig(device="auto", cache_dir=args.cache,
                    encoder=EncoderConfig(model_id=args.model, resolution=args.res))
    enc = CachedEncoder(cfg, dev, args.cache)
    h, w = np.asarray(base[0][0]).shape[:2]
    print(f"device={dev} dataset={args.dataset} image={h}x{w} ks={ks}", flush=True)

    kmax = max(ks)
    imgs = [np.array(base[i % len(base)][0]).copy() for i in range(kmax)]  # distinct ids
    labs = [np.asarray(base[i % len(base)][1]) for i in range(kmax)]
    feats = [enc.extract(im) for im in imgs]                               # content-cached → fast
    examples = [LabeledExample(imgs[i], feats[i], labs[i]) for i in range(kmax)]

    print(f"\n  {'K':>5} {'ckpt':>5} {'train peak':>12}", flush=True)
    for K in ks:
        for ck in (False, True):
            try:
                be = HeadFusionBackend(device=dev, epochs=args.epochs, batch_size=1, max_side=1536,
                                       instance_mode="blob", scale_fusion=True, upsampler="guided",
                                       encoder=enc, grad_checkpoint=ck)
                torch.manual_seed(0)
                if str(dev).startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                be.fit(examples[:K])
                peak = torch.cuda.max_memory_allocated() / 2**30 if str(dev).startswith("cuda") else 0.0
                print(f"  {K:>5} {str(ck):>5} {peak:>10.1f} GB", flush=True)
            except RuntimeError as e:
                oom = "out of memory" in str(e).lower()
                print(f"  {K:>5} {str(ck):>5} {'OOM' if oom else repr(e)[:40]:>12}", flush=True)
            finally:
                del be
                torch.cuda.empty_cache() if str(dev).startswith("cuda") else None
    print("K_SCALING_DONE")


if __name__ == "__main__":
    main()
