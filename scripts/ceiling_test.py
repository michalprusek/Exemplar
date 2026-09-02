#!/usr/bin/env python
"""M2 gate — the in-context ceiling via the LIBRARY propose path.

Compares an averaged fg/bg prototype vs the library's per-patch kNN correspondence
(MemoryBank + correspondence.score_map) at two resolutions, on DSB2018. Mirrors the
spike ceiling test (which hit knn@672 = 0.624). Still training-free.

Run on tulen:
  ~/dinov3_env/bin/python scripts/ceiling_test.py --cache /disk1/prusek/asg_cache \
      --data /disk1/prusek/dsb2018 --pool 60 --test 50 --bank 10
"""
import argparse
import time

import numpy as np

from active_segmenter.config import EncoderConfig, MatchConfig, RunConfig
from active_segmenter.encoder.cache import EmbeddingCache
from active_segmenter.encoder.dinov3 import Dinov3Encoder
from active_segmenter.eval import metrics
from active_segmenter.eval.datasets import load_dsb2018
from active_segmenter.membank.bank import MemoryBank
from active_segmenter.propose import correspondence as corr


def predict_up(score_grid, shape):
    from skimage.transform import resize

    pf = resize(score_grid.astype(np.float32), tuple(shape), order=0, mode="edge",
                anti_aliasing=False)
    return pf > 0


def proto_predict(feat, bank, shape):
    fg = bank.fg(1).mean(0); fg /= np.linalg.norm(fg) + 1e-8
    bg = bank.bg(1).mean(0); bg /= np.linalg.norm(bg) + 1e-8
    f = feat.reshape(-1, feat.shape[-1])
    s = (f @ fg - f @ bg).reshape(feat.shape[0], feat.shape[1])
    return predict_up(s, shape)


def run_res(res, train, test, bank_idx, cache, extra_prefix, cfg):
    enc = Dinov3Encoder(EncoderConfig(resolution=res), cfg.device_resolved())
    extra = f"{extra_prefix}-res{res}"
    tr = [cache.get_or_compute(im, extra, lambda im=im: enc.extract(im)) for im, _ in train]
    tf = [cache.get_or_compute(im, extra, lambda im=im: enc.extract(im)) for im, _ in test]
    bank = MemoryBank()
    for i in bank_idx:
        binlabel = (np.asarray(train[i][1]) > 0).astype(int)
        bank.add_from_annotation(tr[i], binlabel, {1: 1}, round=0)
    mc = MatchConfig(topk=5, bidirectional=False)
    proto_ious, knn_ious = [], []
    for feat, (im, lbl) in zip(tf, test):
        proto_ious.append(metrics.foreground_iou(proto_predict(feat, bank, np.asarray(lbl).shape), lbl))
        s = corr.score_map(feat, bank, 1, mc)
        knn_ious.append(metrics.foreground_iou(predict_up(s, np.asarray(lbl).shape), lbl))
    return float(np.mean(proto_ious)), float(np.mean(knn_ious))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/asg_cache")
    ap.add_argument("--data", default="/tmp/dsb2018")
    ap.add_argument("--pool", type=int, default=60)
    ap.add_argument("--test", type=int, default=50)
    ap.add_argument("--bank", type=int, default=10)
    args = ap.parse_args()

    cfg = RunConfig(device="auto", cache_dir=args.cache)
    train = load_dsb2018(args.data, "train", args.pool)
    test = load_dsb2018(args.data, "test", args.test)
    cache = EmbeddingCache(args.cache)
    bank_idx = list(range(args.bank))
    print(f"device={cfg.device_resolved()} pool={len(train)} test={len(test)} bank={args.bank}", flush=True)
    for res in (448, 672):
        t0 = time.time()
        p, k = run_res(res, train, test, bank_idx, cache, cfg.encoder.model_id.split("/")[-1], cfg)
        print(f"res={res} grid={res//16}x{res//16}  proto_IoU={p:.3f}  knn_IoU={k:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
