#!/usr/bin/env python
"""Tyche on the AutoSeg panel, scored BOTH ways, so the choice is measured rather than argued.

Tyche (Rakic et al., CVPR 2024) emits a SET of stochastic candidates. Two ways to turn that set
into one number:

  mean  -- average the candidate probability maps, threshold at 0.5. Uses no ground truth. This is
           what the paper currently reports, and it is what our backend has always done.
  best  -- THEIR OWN metric. The paper scores `L_seg({y_k}, y) = min_k L_Dice(y_k, y)`, i.e. the
           candidate closest to the ground truth, justified by the use case of proposing a set to a
           human rater who picks one.

`best` selects using the TEST label, so it is an oracle upper bound and must be labelled as one.
But scoring a baseline by anything other than its authors' own metric understates it, and an error
in our own favour is the one a reviewer reads as deliberate. So: measure both, report theirs, say
what it is.

Same fixed split, seeds and K draw as every other arm.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.environ.get("ASG_REPO", "/disk1/prusek/active-segmenter"))
from active_segmenter.eval.registry import PANEL, load_dataset
from active_segmenter.eval.scoring import score_prediction, primary_key
from active_segmenter.eval.score_record import write_score_record, split_fingerprint
from active_segmenter.segment.tyche_backend import TycheBackend
from active_segmenter.segment.base import LabeledExample

POOL_OVERRIDE = {"ctc_u373": 15, "isbi2012em": 16, "fisbe": 16}


def candidates(be, image):
    """The full [n_pred, H, W] candidate stack at the image's own resolution."""
    import torch
    from skimage.transform import resize
    q = torch.from_numpy(be._img128(image))[None, None].to(be.device)
    with torch.no_grad():
        yhat = be._model.pred_ged_stats(
            {"x": q, "sx": be._sup[0], "sy": be._sup[1], "target_size": be.n_pred}, sigmoid=True)
    hw = np.asarray(image).shape[:2]
    return np.stack([resize(p, hw, order=1, mode="edge", anti_aliasing=True)
                     for p in yhat[0].float().cpu().numpy()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--support", type=int, default=8)
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--test", type=int, default=10000)
    ap.add_argument("--score_dir_mean", required=True)
    ap.add_argument("--score_dir_best", required=True)
    args = ap.parse_args()

    be = TycheBackend(device="cuda")
    seeds = list(range(args.seeds))
    for name in args.datasets.split(","):
        spec = PANEL[name]
        metric = spec.metric if spec.metric != "instance_ap" else "iou"
        pk = primary_key(metric)
        pool_n = POOL_OVERRIDE.get(name, args.pool)
        support_pool, test_pairs = load_dataset(spec, pool_n, args.test, seed=0)
        fp = split_fingerprint(test_pairs)
        print(f"[{name}] pool={len(support_pool)} test={len(test_pairs)} metric={pk} fp={fp}", flush=True)
        mean_seeds, best_seeds = [], []
        for seed in seeds:
            sub = list(np.random.default_rng(seed).choice(len(support_pool), args.support, replace=False))
            be.fit([LabeledExample(support_pool[i][0], None, np.asarray(support_pool[i][1])) for i in sub])
            mv, bv = [], []
            for img, gt in test_pairs:
                cs = candidates(be, img)
                gt = np.asarray(gt)
                mv.append(score_prediction(metric, cs.mean(0) > 0.5, gt)[pk])
                # their metric: the single candidate that scores best against this image's label
                bv.append(max(score_prediction(metric, c > 0.5, gt)[pk] for c in cs))
            mean_seeds.append(mv); best_seeds.append(bv)
            print(f"  seed {seed}: mean={np.mean(mv):.4f}  best-of-{be.n_pred}={np.mean(bv):.4f}", flush=True)
        for d, per, meth in ((args.score_dir_mean, mean_seeds, "tyche"),
                             (args.score_dir_best, best_seeds, "tyche_bestof")):
            write_score_record(d, method=meth, dataset=name, metric=pk, per_seed_images=per,
                               seeds=seeds, split_fp=fp,
                               protocol=dict(pool=pool_n, test=args.test, support=args.support,
                                             split_seed=0, res=128))
        print(f"  {name}: mean={np.mean([np.mean(v) for v in mean_seeds]):.4f}  "
              f"best={np.mean([np.mean(v) for v in best_seeds]):.4f}", flush=True)


if __name__ == "__main__":
    main()
