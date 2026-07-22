"""Diagnose WHAT kills monuseg FOREGROUND quality for best_v2 — the reframe from C15 says the gap is
downstream (boundary / resolution / decoding), not coarse correspondence. Fits best_v2 on the K=8 support,
predicts fg on the fixed test set, and DECOMPOSES the fg error:

  * precision vs recall (over-prediction / false positives  vs  missed nuclei),
  * boundary-band error fraction (what share of wrong pixels sit within 3px of a GT boundary — HIGH ⇒ the
    error is boundary imprecision → Boundary DoU is well-targeted; LOW ⇒ whole-object miss/FP → different lever),
  * per-nucleus detection rate (GT instances recovered at ≥50% pixel-recall),
  * size-stratified per-nucleus recall (small <200px vs large — small≪large ⇒ a RESOLUTION problem).

Run on tulen: PYTHONPATH=. python scripts/monuseg_fg_diag.py --dataset monuseg --seed 0
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
from scripts.al_testbed import make_backend


def _boundary_band(mask, w=3):
    from scipy import ndimage
    m = np.asarray(mask) > 0
    return ndimage.binary_dilation(m, iterations=w) & ~ndimage.binary_erosion(m, iterations=w)


def diag(dataset, res, seed, method, model, cache, feat_upsampler="none", upsampler_factor=2):
    from skimage.measure import label as sklabel, regionprops

    spec = PANEL[dataset]
    dev = RunConfig(device="auto").device_resolved()
    cfg = RunConfig(device="auto", cache_dir=cache,
                    encoder=EncoderConfig(model_id=model, resolution=res,
                                          feat_upsampler=feat_upsampler,
                                          feat_upsample_factor=upsampler_factor))
    enc = CachedEncoder(cfg, dev, cache)
    support_pool, test = load_dataset(spec, 20, 24, seed=0)
    be = make_backend(method, cfg, dev, enc=enc)
    sub = list(np.random.default_rng(seed).choice(len(support_pool), 8, replace=False))
    P = [LabeledExample(support_pool[i][0], enc.extract(support_pool[i][0]),
                        np.asarray(support_pool[i][1])) for i in sub]
    be.fit(P)

    ious, precs, recs, bnd_fracs, det_rates, small_rec, large_rec = [], [], [], [], [], [], []
    for im, lm in test:
        gt = np.asarray(lm)
        gtfg = gt > 0
        pred = np.asarray(be.foreground(im, enc.extract(im))) > 0
        if pred.shape != gtfg.shape:
            continue
        inter = int((pred & gtfg).sum()); union = int((pred | gtfg).sum())
        tp = inter; fp = int((pred & ~gtfg).sum()); fn = int((~pred & gtfg).sum())
        ious.append(inter / max(union, 1))
        precs.append(tp / max(tp + fp, 1)); recs.append(tp / max(tp + fn, 1))
        err = pred ^ gtfg
        band = _boundary_band(gtfg, 3)
        bnd_fracs.append(int(err[band].sum()) / max(int(err.sum()), 1))
        comps = gt if int(gt.max()) > 1 else sklabel(gtfg)
        det = ninst = 0
        for p in regionprops(comps):
            if p.area < 4:
                continue
            ninst += 1
            m = comps == p.label
            r = int((pred & m).sum()) / max(int(m.sum()), 1)
            det += int(r >= 0.5)
            (small_rec if p.area < 200 else large_rec).append(r)
        det_rates.append(det / max(ninst, 1))

    def mm(x):
        return f"{np.mean(x):.3f}" if x else "n/a"

    print(f"\n[{dataset}] best_v2={method} fg diagnostic (seed {seed}, {len(ious)} test imgs):")
    print(f"  fg-IoU {mm(ious)} | precision {mm(precs)} | recall {mm(recs)}")
    print(f"  boundary-band(3px) error fraction {mm(bnd_fracs)}  "
          f"(HIGH→boundary imprecision=Boundary-DoU target; LOW→whole-object miss/FP)")
    print(f"  per-nucleus detection rate (recall>=0.5) {mm(det_rates)}")
    print(f"  per-nucleus recall: small(<200px) {mm(small_rec)} | large {mm(large_rec)}  "
          f"(small<<large→resolution problem)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="monuseg")
    ap.add_argument("--method", default="head_fusion_best_cgate_film_nobank")
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--cache", default="/disk1/prusek/asg_cache_fgdiag")
    ap.add_argument("--feat_upsampler", default="none")
    ap.add_argument("--upsampler_factor", type=int, default=2)
    a = ap.parse_args()
    diag(a.dataset, a.res, a.seed, a.method, a.model, a.cache, a.feat_upsampler, a.upsampler_factor)


if __name__ == "__main__":
    main()
