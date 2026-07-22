"""nnU-Net as an annotation-efficiency rival: trained on the SAME K support masks we get.

Why this benchmark exists. Our paper calls itself "self-configuring", which is nnU-Net's term
(Isensee et al., Nature Methods 2021), so the first question any reviewer asks is "why not just run
nnU-Net?". Citing it and describing the contrast is an argument; running it on the identical support
draw is evidence. The comparison is fair in the only way that matters here -- nnU-Net sees exactly
the images we see, drawn by the same generator, and is scored on the identical fixed test split with
the identical metric.

What is deliberately NOT done, and why:

  * We do not give nnU-Net instance labels. It is a semantic segmenter, and the paper's metric is
    semantic foreground, so the target is binary foreground. Asking it for instances would handicap
    it on a task neither method is being scored on.
  * We do not run 5-fold cross-validation. With K=8 training images a five-way split is not a
    meaningful validation protocol, it is five models trained on ~6 images each. `-f all` trains one
    model on everything the method is allowed to see, which is the strongest configuration.
  * We do not use the 1000-epoch default blindly. nnU-Net defines an epoch as a fixed 250 iterations
    regardless of dataset size, so its wall-clock is independent of having 8 images or 800, and
    1000 epochs over 8 images is far past convergence. `--epochs` selects one of nnU-Net's OWN
    documented trainer variants, and `--epochs 1000` runs the stock default so the budget can be
    shown not to be the thing holding it back. Whatever budget is used MUST be reported.

Each (dataset, seed) gets a fresh nnU-Net dataset id and a fresh model. Reusing either would let one
seed's training leak into the next and collapse the across-seed variance the error bars exist to
show -- the same trap the specialist fine-tuning bench documents.

Run from the nnU-Net environment (it needs nnunetv2, not our DINOv3 stack):

    ~/nnunet_env/bin/python scripts/nnunet_bench.py \
        --datasets monuseg,drive,spheroidj,dsb2018 --support 8 --seeds 3 \
        --epochs 250 --work_dir /disk2/prusek/nnunet_work --score_dir results/final10/nnunet_k8
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VALID_EPOCHS = (5, 10, 20, 50, 100, 250, 500, 750, 1000, 2000, 4000, 8000)


def draw_support(pool, k, seed):
    """EXACTLY the draw sota_final.py uses, so nnU-Net and Exemplar see identical support shots."""
    idx = np.random.default_rng(seed).choice(len(pool), k, replace=False)
    return [pool[i] for i in idx]


def _as_rgb_u8(im):
    """nnU-Net's natural-image path wants uint8 with a fixed channel count.

    Datasets in the panel arrive as uint8 RGB, uint16 grayscale, or float. Normalising per image
    here (rather than letting a cast clip) keeps a 16-bit fluorescence image from collapsing to
    near-black, which would silently starve the model on exactly the faint-signal datasets where
    the comparison is most interesting.
    """
    a = np.asarray(im)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    a = a[..., :3]
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        lo, hi = float(a.min()), float(a.max())
        a = np.zeros_like(a) if hi <= lo else (a - lo) / (hi - lo) * 255.0
        a = a.astype(np.uint8)
    return np.ascontiguousarray(a)


def _write_raw(shots, raw_root, did, name):
    """Write the K shots as an nnU-Net raw dataset (2D natural-image layout, binary foreground)."""
    import skimage.io
    d = os.path.join(raw_root, f"Dataset{did:03d}_{name}")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "imagesTr"))
    os.makedirs(os.path.join(d, "labelsTr"))
    for i, (im, lb) in enumerate(shots):
        rgb = _as_rgb_u8(im)
        for c in range(3):
            skimage.io.imsave(os.path.join(d, "imagesTr", f"{name}_{i:03d}_{c:04d}.png"),
                              rgb[..., c], check_contrast=False)
        skimage.io.imsave(os.path.join(d, "labelsTr", f"{name}_{i:03d}.png"),
                          (np.asarray(lb) > 0).astype(np.uint8), check_contrast=False)
    json.dump({"channel_names": {"0": "R", "1": "G", "2": "B"},
               "labels": {"background": 0, "foreground": 1},
               "numTraining": len(shots), "file_ending": ".png"},
              open(os.path.join(d, "dataset.json"), "w"), indent=2)
    return d


def _run(cmd, env):
    """Fail loud. A failed plan/train step leaves an empty results folder, and the predictor would
    then either raise something unrelated or -- worse -- load a stale checkpoint from a previous
    seed and report it as this seed's number."""
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"nnU-Net step failed: {' '.join(cmd)}\n"
                         f"--- stdout ---\n{p.stdout[-2500:]}\n--- stderr ---\n{p.stderr[-2500:]}")
    return p.stdout


def train_nnunet(shots, name, work, did, epochs, gpu):
    """Train nnU-Net on the K shots; return predict(image) -> binary foreground label map."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    raw = os.path.join(work, "raw")
    pre = os.path.join(work, "preprocessed")
    res = os.path.join(work, "results")
    for p in (raw, pre, res):
        os.makedirs(p, exist_ok=True)
    env = dict(os.environ, nnUNet_raw=raw, nnUNet_preprocessed=pre, nnUNet_results=res,
               CUDA_VISIBLE_DEVICES=str(gpu), nnUNet_n_proc_DA="4",
               # nnU-Net v2.8 torch.compile()s the network by default; on this host triton's
               # generated kernel wants a newer GLIBC than the system provides and training dies
               # inside a dataloader worker, surfacing only as "background workers are no longer
               # alive". Eager mode costs some speed and changes no result.
               nnUNet_compile="f")
    _write_raw(shots, raw, did, name)

    _run([os.path.join(sys.prefix, "bin", "nnUNetv2_plan_and_preprocess"),
          "-d", str(did), "-c", "2d", "--verify_dataset_integrity"], env)
    trainer = "nnUNetTrainer" if epochs == 1000 else f"nnUNetTrainer_{epochs}epochs"
    _run([os.path.join(sys.prefix, "bin", "nnUNetv2_train"),
          str(did), "2d", "all", "-tr", trainer], env)

    model_dir = os.path.join(res, f"Dataset{did:03d}_{name}",
                             f"{trainer}__nnUNetPlans__2d")
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                           device=torch.device("cuda", 0), allow_tqdm=False)
    pred.initialize_from_trained_model_folder(model_dir, use_folds=("all",),
                                              checkpoint_name="checkpoint_final.pth")

    def predict(img):
        rgb = _as_rgb_u8(img)
        # nnU-Net's 2D predictor takes (C, 1, H, W); properties carry the spacing it expects.
        arr = np.ascontiguousarray(rgb.transpose(2, 0, 1)[:, None]).astype(np.float32)
        out = pred.predict_single_npy_array(arr, {"spacing": [999.0, 1.0, 1.0]}, None, None, False)
        return (np.asarray(out).squeeze() > 0).astype(np.int32)

    return predict


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--support", type=int, default=8)
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--test", type=int, default=10000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=250, choices=VALID_EPOCHS,
                    help="one of nnU-Net's OWN trainer variants; 1000 is its stock default")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--work_dir", default="/disk2/prusek/nnunet_work")
    ap.add_argument("--score_dir", required=True)
    ap.add_argument("--keep_work", action="store_true",
                    help="keep the per-seed nnU-Net trees (they are ~GB each; off by default)")
    args = ap.parse_args()

    from active_segmenter.eval.registry import PANEL, load_dataset
    from active_segmenter.eval.scoring import PRIMARY, score_prediction
    from active_segmenter.eval.score_record import split_fingerprint, write_score_record

    for di, name in enumerate(args.datasets.split(",")):
        spec = PANEL[name]
        # The campaign scores instance datasets as foreground IoU; nnU-Net is a semantic segmenter
        # and the paper's claim is semantic, so the effective metric is the semantic one throughout.
        eff = "cldice" if spec.metric == "cldice" else "iou"
        pk = PRIMARY[eff]
        pool, test = load_dataset(spec, args.pool, args.test, seed=0)
        per_seed_images, seeds_used = [], []
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            shots = draw_support(pool, args.support, seed)
            did = 500 + di * 20 + seed          # fresh dataset id per (dataset, seed)
            work = os.path.join(args.work_dir, f"{name}_s{seed}")
            shutil.rmtree(work, ignore_errors=True)
            predict = train_nnunet(shots, name, work, did, args.epochs, args.gpu)
            scores = []
            for im, lb in test:
                fg = predict(im) > 0
                scores.append(float(score_prediction(eff, fg, np.asarray(lb), None)[pk]))
            per_seed_images.append(scores)
            seeds_used.append(seed)
            print(f"    [{name}] K={args.support} seed{seed} {eff}={np.mean(scores):.4f} "
                  f"(n={len(scores)}, {args.epochs} epochs)", flush=True)
            if not args.keep_work:
                shutil.rmtree(work, ignore_errors=True)
        path = write_score_record(
            args.score_dir, method="nnunet", dataset=name, metric=pk,
            per_seed_images=per_seed_images, seeds=seeds_used,
            split_fp=split_fingerprint(test),
            protocol={"support": args.support, "pool": args.pool, "test": len(test),
                      "config": "2d", "folds": "all", "epochs": args.epochs,
                      "trainer": "nnUNetTrainer" if args.epochs == 1000
                                 else f"nnUNetTrainer_{args.epochs}epochs",
                      "target": "binary foreground"},
            note=f"nnU-Net v2 trained from scratch on the same {args.support} support masks per seed")
        print(f"  wrote {path}", flush=True)


if __name__ == "__main__":
    main()
