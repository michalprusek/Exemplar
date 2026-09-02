#!/usr/bin/env python
"""Experiment C — is the active-learning selection optimal?

Two direct tests of "does the image the model picks actually maximise the accuracy
gain?", using fg IoU through kNN correspondence (fast, GPU) as the model quality:

1. GREEDY ORACLE upper bound. Each round, the clairvoyant oracle tries EVERY candidate,
   measures the real test-IoU gain, and picks the true best. Compared against EPIG,
   uncertainty, and random. The oracle is the best any greedy acquisition could do;
   the gap to it is the acquisition's regret.

2. RANK CORRELATION. At a fixed bank, Spearman correlation between each candidate's
   acquisition score and its ACTUAL test-IoU gain if labelled. High correlation ⇒ the
   score predicts informativeness (AL works); ~0 ⇒ the score is no better than random.

Run on tulen:
  ~/dinov3_env/bin/python scripts/analyze_al_optimality.py --cache /disk1/prusek/asg_cache \
      --data /disk1/prusek/dsb2018 --pool 30 --test 30 --res 672 --rounds 6
"""
import argparse

import numpy as np
from scipy.stats import spearmanr
from skimage.transform import resize

from active_segmenter.config import EncoderConfig, MatchConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval import metrics
from active_segmenter.eval.datasets import load_dsb2018
from active_segmenter.membank.bank import MemoryBank
from active_segmenter.propose import correspondence as corr
from active_segmenter.acquire import coldstart
from active_segmenter.acquire.uncertainty import ambiguous_fraction

MC = MatchConfig(topk=5, bidirectional=False)


def bank_of(trf, train, idxs):
    b = MemoryBank()
    for i in idxs:
        b.add_from_annotation(trf[i], (np.asarray(train[i][1]) > 0).astype(int), {1: 1}, 0)
    return b


def test_iou(tef, test, bank, dev):
    out = []
    for f, (im, l) in zip(tef, test):
        s = corr.score_map(f, bank, 1, MC, device=dev)
        pf = resize((s > 0).astype(np.float32), np.asarray(l).shape, order=0,
                    mode="edge", anti_aliasing=False) > 0.5
        out.append(metrics.foreground_iou(pf, l))
    return float(np.mean(out))


def uncertainty(trf, i, bank, dev):
    return ambiguous_fraction(corr.score_map(trf[i], bank, 1, MC, device=dev), MC.fg_bg_margin_eps)


def epig_score(i, unc_i, cls, pool):
    rep = float(np.mean(cls[i] @ cls[pool].T))
    return unc_i * rep


def run_arm(strategy, trf, train, tef, test, cold, pool0, rounds, cls, dev, rng):
    idxs = list(cold)
    pool = [p for p in pool0 if p not in idxs]
    curve = []
    for r in range(rounds):
        bank = bank_of(trf, train, idxs)
        base = test_iou(tef, test, bank, dev)
        curve.append((len(idxs), base))
        if r == rounds - 1 or not pool:
            break
        if strategy == "oracle":
            gains = []
            for i in pool:
                gi = test_iou(tef, test, bank_of(trf, train, idxs + [i]), dev) - base
                gains.append((gi, i))
            pick = max(gains)[1]
        elif strategy == "random":
            pick = int(rng.choice(pool))
        else:
            unc = {i: uncertainty(trf, i, bank, dev) for i in pool}
            if strategy == "uncertainty":
                pick = max(pool, key=lambda i: unc[i])
            else:  # epig
                pick = max(pool, key=lambda i: epig_score(i, unc[i], cls, pool))
        idxs.append(pick)
        pool.remove(pick)
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/asg_cache")
    ap.add_argument("--data", default="/tmp/dsb2018")
    ap.add_argument("--pool", type=int, default=30)
    ap.add_argument("--test", type=int, default=30)
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--rounds", type=int, default=6)
    args = ap.parse_args()

    cfg = RunConfig(device="auto", cache_dir=args.cache, encoder=EncoderConfig(resolution=args.res))
    dev = cfg.device_resolved()
    train = load_dsb2018(args.data, "train", args.pool)
    test = load_dsb2018(args.data, "test", args.test)
    enc = CachedEncoder(cfg, dev, args.cache)
    trf = [enc.extract(im) for im, _ in train]
    tef = [enc.extract(im) for im, _ in test]
    cls = np.stack([enc.extract_cls(im) for im, _ in train])
    pool0 = list(range(len(train)))
    cold = coldstart.typiclust(cls, 3, seed=0)
    print(f"device={dev} res={args.res} pool={len(train)} test={len(test)} rounds={args.rounds}")

    # --- 1. oracle vs strategies ---
    print("\n# greedy oracle upper bound vs strategies (test fg IoU)")
    rng = np.random.default_rng(0)
    arms = {s: run_arm(s, trf, train, tef, test, cold, pool0, args.rounds, cls, dev,
                       np.random.default_rng(0))
            for s in ["oracle", "epig", "uncertainty", "random"]}
    ns = [n for n, _ in arms["oracle"]]
    print(f"{'n':>4} " + " ".join(f"{s:>10}" for s in arms))
    for r in range(len(ns)):
        print(f"{ns[r]:>4} " + " ".join(f"{arms[s][r][1]:>10.3f}" for s in arms))
    # regret of EPIG vs oracle (area between curves)
    reg = np.mean([arms["oracle"][r][1] - arms["epig"][r][1] for r in range(len(ns))])
    reg_rand = np.mean([arms["oracle"][r][1] - arms["random"][r][1] for r in range(len(ns))])
    print(f"\nmean gap to oracle:  EPIG={reg:.3f}   random={reg_rand:.3f}")
    print(f"EPIG closes {100*(1 - reg/reg_rand) if reg_rand>1e-6 else float('nan'):.0f}% of the random->oracle gap")

    # --- 2. rank correlation at the cold-start bank ---
    print("\n# rank correlation: acquisition score vs ACTUAL test-IoU gain (cold bank)")
    bank = bank_of(trf, train, cold)
    base = test_iou(tef, test, bank, dev)
    cand = [i for i in pool0 if i not in cold]
    actual_gain = np.array([test_iou(tef, test, bank_of(trf, train, list(cold) + [i]), dev) - base
                            for i in cand])
    unc = np.array([uncertainty(trf, i, bank, dev) for i in cand])
    epig = np.array([epig_score(i, unc[k], cls, cand) for k, i in enumerate(cand)])
    for name, sc in [("uncertainty", unc), ("EPIG", epig)]:
        rho, p = spearmanr(sc, actual_gain)
        print(f"  Spearman({name}, actual_gain) = {rho:+.3f}  (p={p:.3f})")
    # best-possible single pick vs what EPIG picked
    oracle_gain = actual_gain.max()
    epig_pick_gain = actual_gain[int(np.argmax(epig))]
    rand_gain = float(np.mean(actual_gain))
    print(f"  best single-pick gain={oracle_gain:.3f}  EPIG-pick gain={epig_pick_gain:.3f}  "
          f"avg(random)={rand_gain:.3f}")
    print("DONE")


if __name__ == "__main__":
    main()
