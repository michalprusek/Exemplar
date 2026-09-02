"""Transductive feasibility test: does exploiting the UNLABELED POOL (the in-context setting's key asset)
break the K=8 support-only fg-separability ceiling on monuseg?

Baseline = a classifier trained ONLY on the K=8 support grid patches (the current inductive regime).
Transductive = SELF-TRAINING: seed on the support, predict the whole pool, add high-confidence pool patches
as pseudo-labels, retrain — a few rounds. Both scored on the test GT (grid resolution). If transductive AUROC /
best-threshold fg-IoU beats support-only, the pool carries signal the K=8 masks alone cannot, and a transductive
mechanism is the way to close the data-asymmetry gap the specialists exploit with thousands of LABELED nuclei.

Uses a plain logistic regression on the coarse DINOv3 grid (isolates the TRANSDUCTION axis from the head); the
RELATIVE gain is the signal. Run on tulen: PYTHONPATH=. python scripts/transductive_test.py --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from active_segmenter.config import EncoderConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval.registry import PANEL, load_dataset
from active_segmenter.membank.bank import _mask_to_grid


def _grid_fg(lm, gh, gw):
    return _mask_to_grid(np.asarray(lm) > 0, gh, gw).reshape(-1)


def _best_iou(score, fg):
    best = 0.0
    for t in np.quantile(score, np.linspace(0.05, 0.95, 19)):
        p = score >= t
        best = max(best, (p & fg).sum() / max((p | fg).sum(), 1))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="monuseg")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.9, help="pseudo-label confidence threshold")
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--cache", default="/disk1/prusek/asg_cache_transd")
    a = ap.parse_args()

    dev = RunConfig(device="auto").device_resolved()
    cfg = RunConfig(device="auto", cache_dir=a.cache, encoder=EncoderConfig(model_id=a.model, resolution=a.res))
    enc = CachedEncoder(cfg, dev, a.cache)
    pool_imgs, test = load_dataset(PANEL[a.dataset], 20, 24, seed=0)
    sub = list(np.random.default_rng(a.seed).choice(len(pool_imgs), a.k, replace=False))

    # support seed patches (K masks)
    Xs, ys = [], []
    for i in sub:
        g = np.asarray(enc.extract(pool_imgs[i][0]), np.float32)
        gh, gw, d = g.shape
        fg = _grid_fg(pool_imgs[i][1], gh, gw)
        f = g.reshape(-1, d)
        Xs.append(f); ys.append(fg.astype(int))
    Xs = np.concatenate(Xs); ys = np.concatenate(ys)

    # test features + grid GT (scoring) — also the UNLABELED pool for transduction
    Xt, yt, shapes = [], [], []
    for im, lm in test:
        g = np.asarray(enc.extract(im), np.float32)
        gh, gw, d = g.shape
        Xt.append(g.reshape(-1, d)); yt.append(_grid_fg(lm, gh, gw)); shapes.append((gh, gw))
    Xt_all = np.concatenate(Xt); yt_all = np.concatenate(yt)

    def _mk(kind):
        return (LogisticRegression(max_iter=200, C=1.0, class_weight="balanced") if kind == "LR"
                else MLPClassifier(hidden_layer_sizes=(256,), max_iter=120, early_stopping=True))

    def fit_eval(kind, X, y, tag):
        clf = _mk(kind).fit(X, y)
        s = clf.predict_proba(Xt_all)[:, 1]
        au = roc_auc_score(yt_all, s); iou = _best_iou(s, yt_all.astype(bool))
        print(f"  {tag:26} AUROC {au:.3f}  best-thr fg-IoU {iou:.3f}  (train n={len(y)})")
        return clf, au, iou

    print(f"[{a.dataset}] transductive test (seed {a.seed}, K={a.k}, {len(test)} test imgs):")
    for kind in ("LR", "MLP"):                          # linear + NONLINEAR adapter
        clf, au0, iou0 = fit_eval(kind, Xs, ys, f"{kind} support-only (K=8)")
        au, iou = au0, iou0
        for r in range(a.rounds):                       # SELF-TRAINING over the unlabeled pool
            p = clf.predict_proba(Xt_all)[:, 1]
            hi = p >= a.conf; lo = p <= (1 - a.conf)
            Xp = np.concatenate([Xs, Xt_all[hi], Xt_all[lo]])
            yp = np.concatenate([ys, np.ones(hi.sum(), int), np.zeros(lo.sum(), int)])
            clf, au, iou = fit_eval(kind, Xp, yp, f"{kind} transductive rnd{r + 1}")
        print(f"  → {kind} VERDICT: AUROC {au0:.3f}→{au:.3f} ({au - au0:+.3f}), fg-IoU {iou0:.3f}→{iou:.3f} "
              f"({iou - iou0:+.3f}) — {'HELPS' if (au - au0 > 0.01 or iou - iou0 > 0.01) else 'no gain'}\n")


if __name__ == "__main__":
    main()
