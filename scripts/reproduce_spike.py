#!/usr/bin/env python
"""Reproduce the P1 spike numbers on DSB2018 using the library encoder.

This is the M1 regression gate: a deliberately simple averaged fg/bg prototype
(the spike's own baseline, NOT the library's kNN correspondence) driven by the
GT-as-oracle AL loop. It must reproduce the spike's ~0.47 cold-start IoU and the
AL-vs-random curve, proving the library encoder matches the throwaway spike.

Run on tulen:
  ~/dinov3_env/bin/python scripts/reproduce_spike.py \
      --cache /disk1/prusek/asg_cache --data /disk1/prusek/dsb2018 \
      --pool 120 --test 50 --rounds 10 [--knn --res 672]
"""
import argparse
import csv
import os
import time

import numpy as np
from scipy.cluster.vq import kmeans2

from active_segmenter.config import EncoderConfig, RunConfig
from active_segmenter.encoder.cache import EmbeddingCache
from active_segmenter.encoder.dinov3 import Dinov3Encoder
from active_segmenter.eval import metrics
from active_segmenter.eval.datasets import load_dsb2018


def fg_grid(lbl, g):
    import torch

    m = torch.from_numpy((np.asarray(lbl) > 0).astype(np.float32))[None, None]
    r = torch.nn.functional.interpolate(m, size=(g, g), mode="bilinear", align_corners=False)
    return (r[0, 0] > 0.5).numpy()


def build_prototype(bank_idx, feats, data, g):
    fg, bg = [], []
    for i in bank_idx:
        f = feats[i].reshape(-1, feats[i].shape[-1]).astype(np.float32)
        m = fg_grid(data[i][1], g).reshape(-1)
        if m.any():
            fg.append(f[m])
        if (~m).any():
            bg.append(f[~m])
    fgp = np.concatenate(fg).mean(0)
    bgp = np.concatenate(bg).mean(0)
    fgp /= np.linalg.norm(fgp) + 1e-8
    bgp /= np.linalg.norm(bgp) + 1e-8
    return fgp, bgp


def predict_proto(feat, shape, proto):
    import torch

    fgp, bgp = proto
    f = feat.astype(np.float32)
    sfg = f @ fgp
    sbg = f @ bgp
    predg = (sfg > sbg).astype(np.float32)
    pf = torch.nn.functional.interpolate(
        torch.from_numpy(predg)[None, None], size=tuple(shape), mode="nearest"
    )[0, 0].numpy() > 0.5
    return pf, (sfg - sbg)


def mean_iou(feats, data, proto):
    ious = []
    for feat, (im, lbl) in zip(feats, data):
        pf, _ = predict_proto(feat, np.asarray(lbl).shape, proto)
        ious.append(metrics.foreground_iou(pf, lbl))
    return float(np.mean(ious))


def uncertainty(feat, proto, g):
    _, margin = predict_proto(feat, (g, g), proto)
    return float(np.mean(np.abs(margin) < 0.03))


def cold_start(cls, pool, k=3, seed=0):
    cent, label = kmeans2(cls, k, seed=seed, minit="++", missing="raise")
    picks = []
    for c in range(k):
        idx = [i for i in pool if label[i] == c]
        if idx:
            d = np.linalg.norm(cls[idx] - cent[c], axis=1)
            picks.append(idx[int(np.argmin(d))])
    return picks


def run(strategy, train, test, train_feat, test_feat, cls, g, rounds, seed=0):
    rng = np.random.default_rng(seed)
    pool = list(range(len(train)))
    bank = cold_start(cls, pool, seed=seed)
    for p in bank:
        pool.remove(p)
    curve = []
    for r in range(rounds):
        proto = build_prototype(bank, train_feat, train, g)
        curve.append((len(bank), mean_iou(test_feat, test, proto)))
        if not pool:
            break
        if strategy == "random":
            pick = int(rng.choice(pool))
        else:
            pick = max((uncertainty(train_feat[i], proto, g), i) for i in pool)[1]
        bank.append(pick)
        pool.remove(pick)
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/asg_cache")
    ap.add_argument("--data", default="/tmp/dsb2018")
    ap.add_argument("--pool", type=int, default=120)
    ap.add_argument("--test", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--res", type=int, default=448)
    ap.add_argument("--out", default="results/spike_repro.csv")
    args = ap.parse_args()

    cfg = RunConfig(device="auto", cache_dir=args.cache, encoder=EncoderConfig(resolution=args.res))
    dev = cfg.device_resolved()
    print(f"device={dev} res={args.res}", flush=True)
    g = args.res // 16

    train = load_dsb2018(args.data, "train", args.pool)
    test = load_dsb2018(args.data, "test", args.test)
    print(f"pool={len(train)} test={len(test)}", flush=True)

    enc = Dinov3Encoder(cfg.encoder, dev)
    cache = EmbeddingCache(args.cache)
    extra = f"res{args.res}-{cfg.encoder.model_id}"

    t0 = time.time()
    train_feat = [cache.get_or_compute(im, extra, lambda im=im: enc.extract(im)) for im, _ in train]
    test_feat = [cache.get_or_compute(im, extra, lambda im=im: enc.extract(im)) for im, _ in test]
    cls = np.stack([cache.get_or_compute(im, "cls-" + extra, lambda im=im: enc.extract_cls(im)[None]) for im, _ in train])[:, 0]
    print(f"features in {time.time()-t0:.0f}s", flush=True)

    al = run("uncertainty", train, test, train_feat, test_feat, cls, g, args.rounds)
    rnd = run("random", train, test, train_feat, test_feat, cls, g, args.rounds)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        w = csv.writer(fh)
        w.writerow(["n_annotated", "al_iou", "random_iou"])
        for (n, a), (_, b) in zip(al, rnd):
            w.writerow([n, round(a, 4), round(b, 4)])

    print("\n=== RESULT (test foreground IoU vs #annotated) ===")
    print(f"{'n':>4} {'AL':>7} {'random':>7}")
    for (n, a), (_, b) in zip(al, rnd):
        print(f"{n:>4} {a:>7.3f} {b:>7.3f}")
    print(f"\ncold-start(3-img) IoU={al[0][1]:.3f} | AL final={al[-1][1]:.3f} | random final={rnd[-1][1]:.3f}")
    print("DONE")


if __name__ == "__main__":
    main()
