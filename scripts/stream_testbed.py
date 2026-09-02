#!/usr/bin/env python
"""Streaming (single-pass, per-frame) active-learning test-bed.

The online biologist loop: frames of the uploaded dataset arrive one at a time, and for
each one we decide AT ENCOUNTER TIME whether to ask for an annotation — without seeing
the future frames — under a fixed budget (``StreamingSelector``). Every annotated frame
enters the memory bank, so the frozen in-context segmenter updates after every frame.

Compares, at the same budget and on the same shuffled arrival order:
- ``stream_smec``   : value = committee-disagreement(frame | support) x novelty(frame | support)
- ``stream_random`` : value = random  (the streaming baseline)
and prints the pool-based SMEC final IoU as a non-streaming reference (it sees the whole
pool, so it is an easier upper-ish bound at the same label count).

Run on tulen:
  CUDA_VISIBLE_DEVICES=0 ~/dinov3_env/bin/python scripts/stream_testbed.py \
      --cache /disk1/prusek/asg_cache --data /disk1/prusek/dsb2018 \
      --pool 40 --test 30 --budget 8 --seeds 3
"""
import argparse

import numpy as np
from skimage.transform import resize

from active_segmenter.config import EncoderConfig, MatchConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval import metrics
from active_segmenter.eval.datasets import load_dsb2018
from active_segmenter.membank.bank import MemoryBank
from active_segmenter.propose import correspondence as corr
from active_segmenter.acquire import smec
from active_segmenter.acquire.streaming import StreamingSelector

MC = MatchConfig(topk=5, bidirectional=False)


def bank_of(trf, train, idxs):
    b = MemoryBank()
    for i in idxs:
        b.add_from_annotation(trf[i], (np.asarray(train[i][1]) > 0).astype(int), {1: 1}, 0)
    return b


def test_iou(tef, test, bank, dev):
    if not bank.classes():
        return 0.0
    out = []
    for f, (_, l) in zip(tef, test):
        s = corr.score_map(f, bank, 1, MC, device=dev)
        pf = resize((s > 0).astype(np.float32), np.asarray(l).shape, order=0,
                    mode="edge", anti_aliasing=False) > 0.5
        out.append(metrics.foreground_iou(pf, l))
    return float(np.mean(out))


def smec_value(x, labeled, trf, train, cls, dev, rng, n_committee=4, subset_frac=0.6, min_support=2):
    """Value of an arriving frame = how much the current support set DISAGREES on it
    (committee over support subsets) times how NOVEL it is vs the support."""
    nov = max(0.0, 1.0 - float(np.max(cls[x] @ cls[labeled].T))) if labeled else 1.0
    if len(labeled) < min_support:
        return nov                                   # cold start: pure coverage
    size = max(1, int(round(subset_frac * len(labeled))))
    subsets = [sorted(int(i) for i in rng.choice(labeled, size=size, replace=False))
               for _ in range(n_committee)]
    masks = [corr.score_map(trf[x], bank_of(trf, train, s), 1, MC, device=dev) > 0 for s in subsets]
    return smec.mask_disagreement(masks) * nov


def run_stream(value_fn, order, trf, train, tef, test, cls, dev, budget, warmup, seed):
    sel = StreamingSelector(budget=budget, warmup=warmup, target_frames=len(order), seed=seed)
    rng = np.random.default_rng(seed)
    labeled, curve = [], []
    for x in order:
        val = value_fn(x, labeled, trf, train, cls, dev, rng)
        if sel.should_annotate(val):
            labeled.append(int(x))
            curve.append((len(labeled), test_iou(tef, test, bank_of(trf, train, labeled), dev)))
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/asg_cache")
    ap.add_argument("--data", default="/tmp/dsb2018")
    ap.add_argument("--pool", type=int, default=40)
    ap.add_argument("--test", type=int, default=30)
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    cfg = RunConfig(device="auto", cache_dir=args.cache, encoder=EncoderConfig(resolution=args.res))
    dev = cfg.device_resolved()
    train = load_dsb2018(args.data, "train", args.pool)
    test = load_dsb2018(args.data, "test", args.test)
    enc = CachedEncoder(cfg, dev, args.cache)
    trf = [enc.extract(im) for im, _ in train]
    tef = [enc.extract(im) for im, _ in test]
    cls = np.stack([enc.extract_cls(im) for im, _ in train])
    print(f"device={dev} pool={len(train)} test={len(test)} budget={args.budget} "
          f"warmup={args.warmup} seeds={args.seeds}", flush=True)

    def rand_value(x, labeled, trf, train, cls, dev, rng):
        return float(rng.random())

    finals = {"stream_smec": [], "stream_random": []}
    for seed in range(args.seeds):
        order = list(np.random.default_rng(seed).permutation(len(train)))
        for name, vfn in (("stream_smec", smec_value), ("stream_random", rand_value)):
            curve = run_stream(vfn, order, trf, train, tef, test, cls, dev,
                               args.budget, args.warmup, seed)
            finals[name].append(curve[-1][1] if curve else 0.0)

    print("\n# streaming final test fg IoU (mean +/- std over seeds)")
    for name in ("stream_random", "stream_smec"):
        v = np.array(finals[name])
        print(f"{name:>14}  {v.mean():.3f} +/-{v.std():.3f}")
    print("DONE")


if __name__ == "__main__":
    main()
