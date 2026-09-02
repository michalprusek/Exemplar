#!/usr/bin/env python
"""INSID3 (Cuttano et al., CVPR 2026 Oral) on the AutoSeg panel, at the MATCHED protocol.

RUNS THE AUTHORS' OWN CODE. This script is plumbing only: it hands INSID3 the same fixed test
split, the same seeds and the same K support masks our method receives, then scores its output
with the dataset's designated metric. Nothing of the method is reimplemented -- `build_insid3`,
`set_reference` and `segment` are called exactly as the authors' own `inference_segmentation.py`
calls them.

WHY CRF IS ON BY DEFAULT HERE. Their code defaults to `mask_refiner="bilinear"` and their README
presents CRF as optional, but the published Chest X-ray one-shot number (78.8 mIoU) reproduces
only WITH it: measured on their own script and their own data, bilinear gives 77.3/76.4/77.4 over
three seeds and CRF gives 78.7. Running the bilinear default would therefore benchmark a
configuration the authors do not report, and understate them by about 1.5 points.

  ASG_DATA_ROOT=/disk1/prusek PANEL_DL_ROOT=/disk1/prusek/panel_datasets \
  /scratch/prusek/envs/insid3/bin/python scripts/insid3_bench.py \
      --datasets monuseg --seeds 10 --support 8 --score_dir results/insid3_k8
"""
import argparse, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.environ.get("ASG_REPO", "/disk1/prusek/active-segmenter"))
sys.path.insert(0, "/disk1/prusek/INSID3")
from active_segmenter.eval.registry import PANEL, load_dataset
from active_segmenter.eval.scoring import score_prediction, primary_key
from active_segmenter.eval.score_record import write_score_record, split_fingerprint

# Pool sizes that differ from the default, exactly as sota_final.py applies them: these are all the
# annotation each dataset allows beside a disjoint test split. Getting isbi2012em wrong here would
# shift its TEST slice, not just its pool, because it is a download-kind dataset.
POOL_OVERRIDE = {"ctc_u373": 15, "isbi2012em": 16, "fisbe": 16}


def _pil(a):
    a = np.asarray(a)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.dtype != np.uint8:
        a = (255 * (a.astype(np.float32) - a.min()) / max(1e-6, float(a.max() - a.min()))).astype(np.uint8)
    return Image.fromarray(a[..., :3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--support", type=int, default=8)
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--test", type=int, default=10000)
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--score_dir", required=True)
    ap.add_argument("--bilinear", action="store_true",
                    help="use the code default instead of the configuration the paper reports")
    args = ap.parse_args()

    from models import build_insid3
    os.chdir("/disk1/prusek/INSID3")
    model = build_insid3(image_size=args.image_size,
                        mask_refiner="bilinear" if args.bilinear else "crf")
    seeds = list(range(args.seeds))

    for name in args.datasets.split(","):
        spec = PANEL[name]
        pool_n = POOL_OVERRIDE.get(name, args.pool)
        support_pool, test_pairs = load_dataset(spec, pool_n, args.test, seed=0)
        if args.support > len(support_pool):
            # CTC-U373's pool holds fifteen, so K=16 cannot be drawn -- the same reason the paper's
            # K-scaling figure averages ten datasets rather than eleven. Skip rather than crash, and
            # say so, because a silent skip is how an arm ends up short a cell without anyone noticing.
            print(f"[{name}] SKIP: pool of {len(support_pool)} cannot supply K={args.support}", flush=True)
            continue
        pk = primary_key(spec.metric if spec.metric != "instance_ap" else "iou")
        fp = split_fingerprint(test_pairs)
        print(f"[{name}] pool={len(support_pool)} test={len(test_pairs)} metric={pk} fp={fp}", flush=True)
        per_seed_images = []
        for seed in seeds:
            # THE SAME DRAW our method gets: sota_final.py line 319, verbatim.
            sub = list(np.random.default_rng(seed).choice(len(support_pool), args.support, replace=False))
            refs = [(_pil(support_pool[i][0]), (np.asarray(support_pool[i][1]) > 0).astype(np.uint8) * 255)
                    for i in sub]
            vals = []
            for img, gt in test_pairs:
                for ri, rm in refs:
                    model.set_reference(ri, Image.fromarray(rm))
                model.set_target(_pil(img))
                pred = model.segment()
                fg = np.asarray(pred.detach().cpu() if hasattr(pred, "detach") else pred).astype(bool)
                if fg.shape != np.asarray(gt).shape[:2]:
                    fg = np.asarray(Image.fromarray(fg.astype(np.uint8) * 255)
                                    .resize((np.asarray(gt).shape[1], np.asarray(gt).shape[0]),
                                            Image.NEAREST)) > 127
                vals.append(score_prediction(spec.metric if spec.metric != "instance_ap" else "iou",
                                             fg, np.asarray(gt))[pk])
            per_seed_images.append(vals)
            print(f"  seed {seed}: {pk}={np.mean(vals):.4f}", flush=True)
        p = write_score_record(args.score_dir, method="insid3_official", dataset=name, metric=pk,
                               per_seed_images=per_seed_images, seeds=seeds, split_fp=fp,
                               protocol=dict(pool=pool_n, test=args.test, support=args.support,
                                             split_seed=0, res=args.image_size))
        print(f"  wrote {p}  mean={np.mean([np.mean(v) for v in per_seed_images]):.4f}", flush=True)


if __name__ == "__main__":
    main()
