"""Training-free operating-point sweep on the base model's fg PROBABILITY (monuseg). Fits best_v2 once,
gets foreground_prob per test image, and sweeps the threshold — reporting precision / recall / fg-IoU and a
MERGE PROXY (predicted connected-component count / true nucleus count; <1 ⇒ merges) + detection at each. A
higher threshold that pushes the CC-ratio toward 1 while keeping detection high ⇒ operating-point calibration
un-merges touching nuclei ⇒ would lift instance-AP (candidate lever direction 3)."""
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
from scripts.al_testbed import make_backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="monuseg")
    ap.add_argument("--method", default="head_fusion_best_cgate_film_nobank")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--cache", default="/disk1/prusek/asg_cache_thsweep")
    a = ap.parse_args()
    from skimage.measure import label as sklabel, regionprops

    spec = PANEL[a.dataset]
    dev = RunConfig(device="auto").device_resolved()
    cfg = RunConfig(device="auto", cache_dir=a.cache,
                    encoder=EncoderConfig(model_id=a.model, resolution=a.res))
    enc = CachedEncoder(cfg, dev, a.cache)
    pool, test = load_dataset(spec, 20, 24, seed=0)
    be = make_backend(a.method, cfg, dev, enc=enc)
    sub = list(np.random.default_rng(a.seed).choice(len(pool), 8, replace=False))
    P = [LabeledExample(pool[i][0], enc.extract(pool[i][0]), np.asarray(pool[i][1])) for i in sub]
    be.fit(P)

    probs = [np.asarray(be.foreground_prob(im, enc.extract(im))) for im, _ in test]
    gts = [np.asarray(lm) for _, lm in test]
    print(f"[{a.dataset}] {a.method} threshold sweep (seed {a.seed}, {len(test)} imgs):")
    print(f"{'thr':>5} {'prec':>6} {'rec':>6} {'fgIoU':>6} {'CCratio':>8} {'detect':>7}")
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        precs, recs, ious, ccr, dets = [], [], [], [], []
        for prob, gt in zip(probs, gts):
            pred = prob > thr
            gfg = gt > 0
            tp = int((pred & gfg).sum()); fp = int((pred & ~gfg).sum()); fn = int((~pred & gfg).sum())
            precs.append(tp / max(tp + fp, 1)); recs.append(tp / max(tp + fn, 1))
            ious.append(tp / max(int((pred | gfg).sum()), 1))
            comps = gt if int(gt.max()) > 1 else sklabel(gfg)
            truen = int(comps.max())
            ccr.append(int(sklabel(pred).max()) / max(truen, 1))
            det = sum(1 for p in regionprops(comps)
                      if p.area >= 4 and int((pred & (comps == p.label)).sum()) / max(int((comps == p.label).sum()), 1) >= 0.5)
            dets.append(det / max(truen, 1))
        print(f"{thr:>5.1f} {np.mean(precs):>6.3f} {np.mean(recs):>6.3f} {np.mean(ious):>6.3f} "
              f"{np.mean(ccr):>8.3f} {np.mean(dets):>7.3f}")


if __name__ == "__main__":
    main()
