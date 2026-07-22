"""Operating-point sweep scored on INSTANCE-AP (not fg-IoU) — the screen C16 asked for and nobody ran.

C16 recorded the lesson verbatim: "fg-IoU is the WRONG screen metric for over-prediction levers -- it is
blind to the bled pixels that MERGE touching nuclei ... Next levers targeting precision MUST be screened on
instance-AP." Every probe since (C17 prec, C18 anyup, C22 backbones, C26/C29 readout, C30 layer fusion) was
nevertheless screened on fg-IoU or the fg diagnostic. So the lesson was written down and then not applied.

C17 also named the cheapest untested consequence as its next direction (a): the foreground threshold is
HARD-WIRED at prob 0.5 (`foreground_from_score(logits, hw, thresh=0.0)`), and MoNuSeg sits at precision 0.683
/ recall 0.885 / detection 0.970 -- i.e. there is EXCESS RECALL to trade, and nearly every nucleus is already
detected. Trading some of that recall for precision should un-merge touching nuclei and raise AP even while
fg-IoU stays flat or falls, which is exactly the move fg-IoU cannot see.

This sweeps the threshold and reports BOTH metrics per step, so the two curves can be compared directly.

HONEST SCOPE: the best-tau column here is chosen ON THE TEST SET. It is an UPPER BOUND that measures whether
headroom exists at all -- it is NOT a usable method. Only if a clear peak appears away from 0.5 is it worth
building a support-only (leave-one-out) selector and paying for the honest version.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from active_segmenter.config import EncoderConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval.registry import PANEL, load_dataset
from active_segmenter.eval.scoring import primary_key, score_prediction
from active_segmenter.segment.base import LabeledExample
from active_segmenter.segment.head_fusion_backend import _affinity_watershed_instances, _ridge_map
from scripts.al_testbed import make_backend

MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
CACHE = os.environ.get("ASG_THR_CACHE", "/disk1/prusek/asg_cache_thrap")
TAUS = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]
SEEDS = int(os.environ.get("ASG_THR_SEEDS", "2"))
# targets = the three instance sets where we lose to fine-tuned specialists
DATASETS = os.environ.get("ASG_THR_DATASETS", "monuseg,dsb2018,ctc_u373").split(",")


def fg_iou(pred, gt_fg):
    inter = np.logical_and(pred, gt_fg).sum()
    union = np.logical_or(pred, gt_fg).sum()
    return float(inter / union) if union else 1.0


def main():
    dev = RunConfig(device="auto").device_resolved()
    cfg = RunConfig(device="auto", cache_dir=CACHE,
                    encoder=EncoderConfig(model_id=MODEL, resolution=672))
    enc = CachedEncoder(cfg, dev, CACHE)
    pk = primary_key("instance_ap")

    print(f"# threshold sweep scored on BOTH fg-IoU and instance-AP | {SEEDS} seeds, K=8, res 672")
    print("# tau=0.50 is the current hard-wired operating point.")
    print("# best-tau is chosen ON TEST => an UPPER BOUND on headroom, not a method.\n")

    for name in DATASETS:
        try:
            pool, test = load_dataset(PANEL[name], 20, 24, seed=0)
        except Exception as e:
            print(f"{name}: LOAD FAILED {type(e).__name__}: {e}", flush=True)
            continue

        ious = {t: [] for t in TAUS}
        aps = {t: [] for t in TAUS}

        for seed in range(SEEDS):
            be = make_backend("head_fusion_best_cgate_film_nobank", cfg, dev, enc=enc)
            sub = list(np.random.default_rng(seed).choice(len(pool), 8, replace=False))
            be.fit([LabeledExample(pool[i][0], enc.extract(pool[i][0]), np.asarray(pool[i][1]))
                    for i in sub])

            for im, gt in test:
                gt = np.asarray(gt)
                gt_fg = gt > 0
                fgrid = enc.extract(im)
                prob = be.foreground_prob(im, fgrid)          # native-res sigmoid, pre-threshold
                ridge = _ridge_map(be._channel(im))
                # Semantic datasets (binary GT) leave the instance decoder uncalibrated (_inst_r is None).
                # Score fg-IoU only there rather than crashing or, worse, inventing an r*.
                has_inst = be._inst_r is not None and be._inst_merge_cos is not None
                for t in TAUS:
                    fg = prob >= t
                    ious[t].append(fg_iou(fg, gt_fg))
                    if has_inst:
                        inst = [m.mask for m in _affinity_watershed_instances(
                            fg, ridge, fgrid, 1, r_star=be._inst_r, merge_cos=be._inst_merge_cos)]
                        aps[t].append(float(score_prediction("instance_ap", fg, gt, inst)[pk]))

        print(f"=== {name} ===")
        if not aps[TAUS[0]]:
            # Semantic dataset: the instance decoder never ran, so every aps[t] is EMPTY. np.mean([]) is
            # NaN, and max() over NaNs just returns the first tau -- which would print a confident
            # "best-tau = 0.30: AP nan -> nan" conclusion that no measurement supports. Report the
            # fg-IoU sweep only and say why the AP column is absent.
            print("  (semantic dataset: instance decoder inactive — AP column and best-tau SKIPPED)")
            print(f"{'tau':>6} {'fg_IoU':>8}")
            for t in TAUS:
                mark = "  <- current" if abs(t - 0.5) < 1e-9 else ""
                print(f"{t:6.2f} {np.mean(ious[t]):8.3f}{mark}")
            print(flush=True)
            continue
        print(f"{'tau':>6} {'fg_IoU':>8} {'AP':>8}")
        for t in TAUS:
            mark = "  <- current" if abs(t - 0.5) < 1e-9 else ""
            print(f"{t:6.2f} {np.mean(ious[t]):8.3f} {np.mean(aps[t]):8.3f}{mark}")
        best = max(TAUS, key=lambda t: np.mean(aps[t]))
        d_ap = np.mean(aps[best]) - np.mean(aps[0.50])
        d_iou = np.mean(ious[best]) - np.mean(ious[0.50])
        print(f"  best-tau (TEST-CHOSEN, upper bound) = {best:.2f}: "
              f"AP {np.mean(aps[0.50]):.3f} -> {np.mean(aps[best]):.3f} ({d_ap:+.3f}), "
              f"fg-IoU {np.mean(ious[0.50]):.3f} -> {np.mean(ious[best]):.3f} ({d_iou:+.3f})\n", flush=True)

    print("READ IT LIKE THIS")
    print("  If AP peaks well away from tau=0.50 while fg-IoU peaks AT 0.50 (or falls), then the")
    print("  operating point is real headroom that every fg-IoU-screened probe was blind to, and the")
    print("  next step is an honest support-LOO threshold selector.")
    print("  If the AP curve is flat, the operating point is exhausted and the gap is elsewhere.")


if __name__ == "__main__":
    main()
