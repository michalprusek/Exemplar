"""SAM 3 text-PCS benchmark on dsb2018 instance-AP (spec 2026-07-12 refine-stage, SAM3 stretch).

Runs in the DINOv3 env: loads dsb2018 test via the shared registry (same GT + instance_ap as every
other benchmark), batches the query images to the isolated SAM 3 worker (``~/sam3_env`` subprocess),
scores each concept. Lets us compare zero-shot text-concept SAM 3 head-to-head against the
in-context head + amodal-SAM refine (0.445) and INSID3 (0.271).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile

import numpy as np

from active_segmenter.eval.registry import PANEL, load_dataset
from active_segmenter.eval.scoring import primary_key, score_prediction

_SAM3_PY = os.environ.get("SAM3_PYTHON", "/home/prusek/sam3_env/bin/python")
_WORKER = "scripts/sam3_worker.py"


def head_proposal_boxes(spec, images, k, superres, cache, dev):
    """k XYXY-absolute boxes from the trainable head's OWN top-k predicted instances per image —
    the realistic (non-oracle) exemplar seed: head propose → SAM3 exemplar-PCS → all instances."""
    from active_segmenter.config import EncoderConfig, RunConfig
    from active_segmenter.encoder.cached import CachedEncoder
    from active_segmenter.segment.base import LabeledExample
    from scripts.al_testbed import make_backend

    support, _ = load_dataset(spec, 16, max(4, len(images)), seed=0)
    cfg = RunConfig(device="auto", cache_dir=cache,
                    encoder=EncoderConfig(resolution=672, superres_factor=superres))
    enc = CachedEncoder(cfg, dev, cache)
    sup = [LabeledExample(im, enc.extract(im), np.asarray(l)) for im, l in support]
    be = make_backend("head", cfg, dev)
    be.fit(sup)
    boxes_per = []
    for im in images:
        insts = sorted(be.predict(im, enc.extract(im)), key=lambda m: int(m.mask.sum()),
                       reverse=True)[:k]
        bs = []
        for m in insts:
            ys, xs = np.where(m.mask)
            if len(xs):
                bs.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
        boxes_per.append(np.asarray(bs, np.float32))
    return boxes_per


def gt_exemplar_boxes(label_map, k, seed=0):
    """k XYXY-absolute exemplar boxes from the LARGEST connected components of the binary GT
    foreground — a few worked examples of the target morphology (oracle-seeded upper bound)."""
    from scipy import ndimage

    lab, n = ndimage.label(np.asarray(label_map) > 0)
    if n == 0:
        return np.zeros((0, 4), np.float32)
    areas = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
    order = np.argsort(areas)[::-1][:k]
    boxes = []
    for idx in order:
        ys, xs = np.where(lab == idx + 1)
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
    return np.asarray(boxes, np.float32)


def sam3_masks_batch(images, threshold, mode="text", concept=None, box_lists=None):
    with tempfile.TemporaryDirectory() as td:
        inp, outp = os.path.join(td, "in.npz"), os.path.join(td, "out.npz")
        payload = {"n": len(images), "threshold": threshold, "mode": mode}
        if mode == "text":
            payload["text"] = concept
        for i, im in enumerate(images):
            payload[f"img_{i}"] = np.asarray(im)
            if mode == "exemplar":
                payload[f"boxes_{i}"] = np.asarray(box_lists[i], np.float32)
        np.savez(inp, **payload)
        r = subprocess.run([_SAM3_PY, _WORKER, inp, outp], capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(outp):
            raise RuntimeError(f"SAM3 worker failed: {r.stderr[-600:]}")
        d = np.load(outp, allow_pickle=True)
        return [list(d[f"masks_{i}"]) for i in range(len(images))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dsb2018")
    ap.add_argument("--test", type=int, default=16)
    ap.add_argument("--mode", default="text", choices=["text", "exemplar"])
    ap.add_argument("--concepts", default="cell,spot,blob", help="text mode: comma list of concepts")
    ap.add_argument("--k", type=int, default=3, help="exemplar mode: # example boxes")
    ap.add_argument("--seed_boxes", default="gt", choices=["gt", "head"],
                    help="exemplar seed: gt (oracle) or head (realistic: head's own proposals)")
    ap.add_argument("--superres", type=int, default=2)
    ap.add_argument("--cache", default="/disk2/prusek/asg_cache_superres2")
    ap.add_argument("--threshold", type=float, default=0.3)
    args = ap.parse_args()

    spec = PANEL[args.dataset]
    _, test = load_dataset(spec, 4, args.test, seed=0)
    images = [im for im, _ in test]
    labels = [np.asarray(lab) for _, lab in test]
    pk = primary_key(spec.metric)
    print(f"SAM3 {args.mode}-PCS on {args.dataset}: test={len(images)} metric={spec.metric} "
          f"thr={args.threshold} k={args.k}", flush=True)

    def report(name, masks_per_img):
        scores, ninst = [], []
        for masks, lab in zip(masks_per_img, labels):
            fg = np.any(masks, axis=0) if len(masks) else np.zeros(lab.shape, bool)
            scores.append(score_prediction(spec.metric, fg, lab, masks)[pk])
            ninst.append(len(masks))
        print(f"{name:>18} {np.mean(scores):>8.3f} {np.mean(ninst):>11.1f}", flush=True)

    print(f"{'prompt':>18} {pk:>8} {'mean #inst':>11}", flush=True)
    if args.mode == "exemplar":
        if args.seed_boxes == "head":
            from active_segmenter.config import RunConfig
            dev = RunConfig(device="auto").device_resolved()
            box_lists = head_proposal_boxes(spec, images, args.k, args.superres, args.cache, dev)
        else:
            box_lists = [gt_exemplar_boxes(lab, args.k) for lab in labels]
        report(f"exemplar(k={args.k},{args.seed_boxes})", sam3_masks_batch(
            images, args.threshold, mode="exemplar", box_lists=box_lists))
    else:
        for concept in args.concepts.split(","):
            report(concept, sam3_masks_batch(images, args.threshold, mode="text", concept=concept))
    print("SAM3_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
