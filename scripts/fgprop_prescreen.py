"""Training-free pre-screen: can the K support masks predict the QUERY foreground PROPORTION?

Motivation (C16/C28). The measured MoNuSeg failure mode is OVER-PREDICTION, not missed objects:
precision 0.683 / recall 0.885 / detection 0.97 => predicted fg AREA is ~R/P = 1.30x the true area,
and those bled pixels MERGE touching nuclei, which is what kills instance-AP.

C28 RANK 2 (RePRI) says a foreground-PROPORTION constraint is "the ONE signal our earlier
self-training lacked", and that an ORACLE proportion buys +11-14 mIoU. RePRI uses an oracle.
We cannot -- but we have K support masks. THIS SCRIPT ASKS THE ONLY QUESTION THAT MATTERS FIRST:

    is the support-derived fg proportion a good enough estimate of the query fg proportion
    to be worth constraining on, WITHOUT any oracle?

No GPU, no training, no model -- pure ground-truth mask statistics. If the support estimate is not
materially better than our current 1.30x over-prediction, the lever is dead before any GPU-hour.

Protocol per CLAUDE.md: load_dataset(spec, pool, test, seed=0) ONCE, subsample K per seed.
"""
import sys
import numpy as np

from active_segmenter.eval.registry import PANEL, load_dataset

POOL, TEST, K, SEEDS = 20, 24, 8, 10
# monuseg = the wall. dsb2018/ctc_u373 = the other two instance sets (same metric).
# spheroidj (compact blobs) + drive (thin vessels) = morphology controls: a proportion prior must
# not be misleading there either, or it cannot go into a one-pipeline method.
DATASETS = ["monuseg", "dsb2018", "ctc_u373", "spheroidj", "drive"]

# our measured MoNuSeg operating point (CONFIG-REGISTRY C16, best_v2 base)
OURS_PRECISION, OURS_RECALL = 0.683, 0.885


def frac(label_map):
    a = np.asarray(label_map)
    return float((a > 0).mean())


def main():
    over = OURS_RECALL / OURS_PRECISION
    print(f"# reference: our MoNuSeg predicted-area / true-area = R/P = {over:.3f} "
          f"(we over-predict fg area by {100 * (over - 1):.0f}%)\n")
    hdr = (f"{'dataset':<12} {'test_fg':>8} {'per-img':>8} {'sup_est':>8} {'seed_sd':>8} "
           f"{'bias':>8} {'best_MAE':>9} {'img_MAE':>9}")
    print(hdr)
    print("-" * len(hdr))

    for name in DATASETS:
        spec = PANEL.get(name)
        if spec is None:
            print(f"{name:<12} MISSING from PANEL")
            continue
        try:
            pool_pairs, test_pairs = load_dataset(spec, POOL, TEST, seed=0)
        except Exception as e:                                    # fail loud, do not substitute
            print(f"{name:<12} LOAD FAILED: {type(e).__name__}: {e}")
            continue

        pool_fracs = np.array([frac(m) for _, m in pool_pairs], float)
        test_fracs = np.array([frac(m) for _, m in test_pairs], float)
        if len(pool_fracs) < K or len(test_fracs) == 0:
            print(f"{name:<12} TOO SMALL: pool={len(pool_fracs)} test={len(test_fracs)}")
            continue

        t_mean = test_fracs.mean()
        # per-image heterogeneity: the irreducible error of ANY single global proportion prior
        per_img_spread = np.abs(test_fracs - t_mean).mean() / t_mean

        ests = []
        for s in range(SEEDS):
            rng = np.random.default_rng(s)
            idx = rng.choice(len(pool_fracs), size=K, replace=False)
            ests.append(pool_fracs[idx].mean())
        ests = np.array(ests, float)

        bias = (ests.mean() - t_mean) / t_mean                    # systematic error of the estimate
        # the achievable error of the support prior, per seed, against the true dataset proportion
        est_rel_err = np.abs(ests - t_mean) / t_mean
        # and against each individual test image (what a per-image constraint would actually face)
        img_mae = np.mean([np.abs(ests[s] - test_fracs).mean() / t_mean for s in range(SEEDS)])

        print(f"{name:<12} {t_mean:8.4f} {per_img_spread:8.3f} {ests.mean():8.4f} "
              f"{ests.std():8.4f} {bias:+8.3f} {est_rel_err.mean():9.3f} {img_mae:9.3f}")

    print("\nCOLUMNS")
    print("  test_fg   mean ground-truth fg fraction over the 24 test images")
    print("  per-img   mean |img_fg - dataset_fg| / dataset_fg  = irreducible error of ANY global prior")
    print("  sup_est   mean over seeds of the K=8 support fg fraction (the estimate we could actually use)")
    print("  seed_sd   sd of that estimate across the 10 seeds (how support-draw-sensitive it is)")
    print("  bias      systematic relative error of the support estimate vs the true dataset proportion")
    print("  best_MAE  mean |support_est - dataset_fg| / dataset_fg  = DATASET-level prior error")
    print("  img_MAE   mean |support_est - img_fg| / dataset_fg      = PER-IMAGE prior error")
    print("\nGO/NO-GO: the lever is worth GPU time only if best_MAE (and ideally img_MAE) is")
    print(f"          materially below our current {100 * (over - 1):.0f}% over-prediction on monuseg.")


if __name__ == "__main__":
    sys.exit(main())
