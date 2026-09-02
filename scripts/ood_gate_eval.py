#!/usr/bin/env python
"""Validate the W3 OOD confidence gate: does the model-free support-distance score (CLS space)
separate the catastrophic per-image failures from the successes?

For each dataset: fit head_fusion on the support, score every test image's fg-IoU, compute its
OOD z-score (`active_segmenter.acquire.ood.ood_scores`, nearest-support cosine distance / support
spread), and report (a) Spearman corr between OOD-z and fg-IoU (expect NEGATIVE: more OOD → worse),
(b) a gate analysis — flag test images with z above a threshold as low-confidence and measure
whether that catches the failures (recall of images below a fg-IoU floor), and (c) mean fg-IoU of
the flagged vs kept sets (the gate is useful iff flagged ≪ kept).

Run on tulen (GPU1):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/disk1/prusek/active-segmenter \
    ~/dinov3_env/bin/python scripts/ood_gate_eval.py --datasets kvasir,spheroid \
    --support 12 --test 30 --cache /disk2/prusek/asg_cache_lowk
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from active_segmenter.acquire.ood import ood_scores
from active_segmenter.config import EncoderConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval.metrics import foreground_iou
from active_segmenter.eval.registry import PANEL, load_dataset


def run_dataset(name, spec, args, dev):
    from scripts.al_testbed import make_backend
    from active_segmenter.segment.base import LabeledExample

    support, test = load_dataset(spec, args.support, args.test, seed=0)
    if len(support) < 3 or len(test) < 4:
        raise RuntimeError(f"too few images (support={len(support)} test={len(test)})")
    cfg = RunConfig(device="auto", cache_dir=args.cache,
                    encoder=EncoderConfig(model_id=args.model, resolution=args.res))
    enc = CachedEncoder(cfg, dev, args.cache)
    cls_sup = np.stack([enc.extract_cls(im) for im, _ in support])
    cls_tst = np.stack([enc.extract_cls(im) for im, _ in test])
    raw, z = ood_scores(cls_sup, cls_tst)

    be = make_backend(args.segmenter, cfg, dev)
    be.fit([LabeledExample(im, enc.extract(im), (np.asarray(l) > 0).astype(int))
            for im, l in support])
    ious = np.array([foreground_iou(be.foreground(im, enc.extract(im)), np.asarray(l) > 0)
                     for im, l in test])

    rho, p = spearmanr(z, ious)
    # gate at the support-spread multiple `args.zgate`: flag z > zgate as low-confidence
    flagged = z > args.zgate
    floor = args.iou_floor
    fails = ious < floor
    recall = float(flagged[fails].mean()) if fails.any() else float("nan")   # failures caught
    prec = float(fails[flagged].mean()) if flagged.any() else float("nan")   # flagged that are fails
    kept_iou = float(ious[~flagged].mean()) if (~flagged).any() else float("nan")
    flag_iou = float(ious[flagged].mean()) if flagged.any() else float("nan")

    print(f"\n[{name}] ({spec.note}; support={len(support)} test={len(test)})", flush=True)
    print(f"  fg-IoU: mean {ious.mean():.3f}  min {ious.min():.3f}  max {ious.max():.3f}", flush=True)
    print(f"  OOD-z : mean {z.mean():.2f}  min {z.min():.2f}  max {z.max():.2f}", flush=True)
    print(f"  Spearman(OOD-z, fg-IoU) = {rho:+.3f} (p={p:.3f})  [want NEGATIVE]", flush=True)
    print(f"  gate z>{args.zgate}: flagged {int(flagged.sum())}/{len(test)}  "
          f"| catches {recall*100:.0f}% of fg-IoU<{floor} failures (precision {prec*100:.0f}%)",
          flush=True)
    print(f"  mean fg-IoU  flagged {flag_iou:.3f}  vs kept {kept_iou:.3f}  "
          f"(gate useful iff flagged << kept)", flush=True)
    return {"dataset": name, "iou": ious.tolist(), "ood_z": z.tolist(), "rho": float(rho),
            "p": float(p), "recall": recall, "precision": prec,
            "kept_iou": kept_iou, "flag_iou": flag_iou}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="kvasir,spheroid,dsb2018")
    ap.add_argument("--support", type=int, default=12)
    ap.add_argument("--test", type=int, default=30)
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--segmenter", default="head_fusion")
    ap.add_argument("--zgate", type=float, default=2.0, help="flag test images with OOD-z above this")
    ap.add_argument("--iou_floor", type=float, default=0.4, help="fg-IoU below this = a failure")
    ap.add_argument("--cache", default="/disk2/prusek/asg_cache_lowk")
    ap.add_argument("--out", default="/disk2/prusek/ood_gate_results.json")
    args = ap.parse_args()

    dev = RunConfig(device="auto").device_resolved()
    names = [n for n in args.datasets.split(",") if n in PANEL]
    print(f"device={dev} datasets={names} segmenter={args.segmenter} "
          f"zgate={args.zgate} iou_floor={args.iou_floor}", flush=True)
    out = []
    for name in names:
        try:
            out.append(run_dataset(name, PANEL[name], args, dev))
        except Exception as e:
            print(f"\n[{name}] SKIP — {repr(e)[:160]}", flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}\nOOD_DONE", flush=True)


if __name__ == "__main__":
    main()
