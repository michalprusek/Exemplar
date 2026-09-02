#!/usr/bin/env python
"""Universal-recipe ABLATION — the always-on superset (`head_fusion_uni`) regresses on clean blobs
(spheroid −0.030) while helping filaments/touching. Which added term is responsible? Ablate each of the
four uni additions (clDice loss, CLAHE contrast, finer classical scales, boundary head) one at a time on a
BLOB dataset (where uni regresses) and a FILAMENT dataset (where uni helps), so we down-weight only the
culprit — and keep the terms that carry the wins.

sf_gid = the strong baseline (none of the four). uni = all four. Each `uni_no_*` drops exactly one.

Run:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. ~/dinov3_env/bin/python scripts/uni_ablation.py \
    --datasets spheroid,hrf --support 8 --pool 20 --test 24 --seeds 3 --cache /disk1/prusek/asg_cache_panel
"""
import argparse

import numpy as np

from active_segmenter.config import EncoderConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval.registry import PANEL, load_dataset
from active_segmenter.eval.scoring import primary_key, score_prediction
from active_segmenter.segment.base import LabeledExample

# uni's four additions over sf_gid; each ablation flips ONE off.
UNI = dict(cldice=True, boundary_head=True, contrast_norm=True, fine_scales=True)
VARIANTS = [
    ("sf_gid", {k: False for k in UNI}),
    ("uni", dict(UNI)),
    ("uni_no_cldice", {**UNI, "cldice": False}),
    ("uni_no_clahe", {**UNI, "contrast_norm": False}),
    ("uni_no_finescale", {**UNI, "fine_scales": False}),
    ("uni_no_boundary", {**UNI, "boundary_head": False}),
]


def run_dataset(spec, args, dev):
    import torch
    from active_segmenter.segment.head_fusion_backend import HeadFusionBackend

    pool, test = load_dataset(spec, args.pool, args.test, seed=0)
    if len(pool) < args.support or len(test) < 2:
        raise RuntimeError(f"too few images (pool={len(pool)} test={len(test)})")
    cfg = RunConfig(device="auto", cache_dir=args.cache,
                    encoder=EncoderConfig(model_id=args.model, resolution=args.res))
    enc = CachedEncoder(cfg, dev, args.cache)
    P = [LabeledExample(im, enc.extract(im), np.asarray(l)) for im, l in pool]
    T = [LabeledExample(im, enc.extract(im), np.asarray(l)) for im, l in test]
    pk = primary_key(spec.metric)
    base = dict(device=dev, epochs=args.epochs, max_side=1536, instance_mode="blob",
                scale_fusion=True, upsampler="guided", encoder=enc, batch_size=1)
    out = {}
    for name, over in VARIANTS:
        be = HeadFusionBackend(**base, **over)
        prims = []
        for s in range(args.seeds):
            be.head = None
            rng = np.random.default_rng(s)
            idx = list(rng.choice(len(P), args.support, replace=False))
            torch.manual_seed(s)
            be.fit([P[i] for i in idx])
            rows = []
            for ex in T:
                fg = be.foreground(ex.image, ex.feat_grid)
                inst = [m.mask for m in be.predict(ex.image, ex.feat_grid)] if spec.metric == "instance_ap" else None
                rows.append(score_prediction(spec.metric, fg, ex.label_map, inst))
            prims.append(float(np.mean([r[pk] for r in rows])))
        out[name] = (float(np.mean(prims)), float(np.std(prims)))
        print(f"  {name:>16}: {pk} {out[name][0]:.3f}±{out[name][1]:.3f}", flush=True)
        torch.cuda.empty_cache() if str(dev).startswith("cuda") else None
    return out, pk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="spheroid,hrf")
    ap.add_argument("--support", type=int, default=8)
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--test", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--cache", default="/tmp/asg_cache_panel")
    args = ap.parse_args()
    dev = RunConfig(device="auto").device_resolved()
    names = [n for n in args.datasets.split(",") if n in PANEL]
    print(f"device={dev} datasets={names} ablation of uni's 4 additions", flush=True)
    for name in names:
        print(f"\n[{name}] ({PANEL[name].note})", flush=True)
        try:
            run_dataset(PANEL[name], args, dev)
        except Exception as e:
            print(f"  SKIP — {repr(e)[:140]}", flush=True)
    print("UNI_ABLATION_DONE")


if __name__ == "__main__":
    main()
