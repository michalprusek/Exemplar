"""Test edge-aware SPATIAL refinement of best_v2's monuseg fg probability — a single, training-free,
feature-orthogonal mechanism targeting the diagnosed diffuse interior bleed + speckle by snapping fg to
H&E nucleus edges. Compares raw vs guided-filter vs dense-CRF, at each method's best threshold, reporting
precision / recall / fg-IoU + merge/speckle proxy (predicted-CC-count / true-count). Guide = the H&E
hematoxylin channel (nuclei-high), a far better edge guide than gray for this modality.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from active_segmenter.config import EncoderConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval.registry import PANEL, load_dataset
from active_segmenter.segment.base import LabeledExample
from active_segmenter.segment.crf import guided_upsample
from scripts.al_testbed import make_backend


def _hematoxylin(img):
    a = np.asarray(img, np.float32)
    if a.ndim != 3 or a.shape[2] < 3:
        g = a if a.ndim == 2 else a.mean(-1)
        return (g - g.min()) / (np.ptp(g) + 1e-6)
    try:
        from skimage.color import rgb2hed
        h = rgb2hed(a[..., :3] / (float(a[..., :3].max()) + 1e-6))[..., 0]
        return (h - h.min()) / (np.ptp(h) + 1e-6)
    except Exception:
        g = a[..., :3].mean(-1)
        return (g - g.min()) / (np.ptp(g) + 1e-6)


def _metrics(prob, gt, thr):
    from skimage.measure import label as sklabel, regionprops
    pred = prob > thr
    gfg = gt > 0
    tp = int((pred & gfg).sum()); fp = int((pred & ~gfg).sum()); fn = int((~pred & gfg).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    iou = tp / max(int((pred | gfg).sum()), 1)
    comps = gt if int(gt.max()) > 1 else sklabel(gfg)
    truen = int(comps.max())
    ccr = int(sklabel(pred).max()) / max(truen, 1)
    det = sum(1 for p in regionprops(comps)
              if p.area >= 4 and int((pred & (comps == p.label)).sum()) / max(int((comps == p.label).sum()), 1) >= 0.5)
    return prec, rec, iou, ccr, det / max(truen, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="monuseg")
    ap.add_argument("--method", default="head_fusion_best_cgate_film_nobank")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--radius", type=int, default=8)
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--cache", default="/disk1/prusek/asg_cache_crf")
    a = ap.parse_args()

    dev = RunConfig(device="auto").device_resolved()
    cfg = RunConfig(device="auto", cache_dir=a.cache, encoder=EncoderConfig(model_id=a.model, resolution=a.res))
    enc = CachedEncoder(cfg, dev, a.cache)
    pool, test = load_dataset(PANEL[a.dataset], 20, 24, seed=0)
    be = make_backend(a.method, cfg, dev, enc=enc)
    sub = list(np.random.default_rng(a.seed).choice(len(pool), 8, replace=False))
    be.fit([LabeledExample(pool[i][0], enc.extract(pool[i][0]), np.asarray(pool[i][1])) for i in sub])

    items = [(np.asarray(be.foreground_prob(im, enc.extract(im))), np.asarray(im), np.asarray(lm)) for im, lm in test]

    def eval_refiner(name, refine):
        best = None
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            M = np.array([_metrics(refine(pr, im), gt, thr) for pr, im, gt in items])
            m = M.mean(0)
            if best is None or m[2] > best[1][2]:
                best = (thr, m)
        thr, m = best
        print(f"  {name:16} @thr {thr:.1f}: prec {m[0]:.3f} rec {m[1]:.3f} fg-IoU {m[2]:.3f} "
              f"CCratio {m[3]:.3f} detect {m[4]:.3f}")
        return m[2]

    print(f"[{a.dataset}] {a.method} SPATIAL-refine test (seed {a.seed}, {len(test)} imgs, guide=hematoxylin, radius {a.radius}):")
    iou_raw = eval_refiner("raw", lambda pr, im: pr)
    iou_g = eval_refiner("guided(hema)", lambda pr, im: guided_upsample(_hematoxylin(im), pr, radius=a.radius))
    print(f"\nCRF/guided VERDICT: best-thr fg-IoU raw {iou_raw:.3f} → guided {iou_g:.3f} ({iou_g - iou_raw:+.3f}) — "
          f"{'HELPS' if iou_g - iou_raw > 0.01 else 'no gain'}")


if __name__ == "__main__":
    main()
