#!/usr/bin/env python
"""PerSeg in-context (K=1 personalized) segmentation evaluator.

Loops the 40 PerSeg objects. For each object it conditions a pluggable method on the SINGLE
reference (index ``00`` image + mask) and predicts a binary foreground mask for every query
image (all indices except ``00``). Metric = **mIoU exactly as PerSAM's ``eval_miou.py``**:
per-object IoU is the ratio of accumulated-over-queries intersection / union, averaged over the
40 objects. Also reports mean-per-image IoU and boundary-IoU (bIoU) as diagnostics.

This is the first GENERAL (natural-object) in-context signal for our biomed few-shot segmenter,
directly comparable to the published PerSAM / PerSAM-F / Matcher / SegGPT PerSeg numbers.

Methods (``--method``):
  best_v2      : ``make_backend("head_fusion_best_cgate_film")`` — our best segmenter. Fit the light
                 head on the K=1 reference (image, mask), then ``foreground()`` each query.
  dinov2_corr  : training-free dense-correspondence baseline on the SAME frozen DINOv3 features —
                 nearest-neighbour fg-vs-bg matching: label each query patch fg iff its max cosine
                 similarity to any reference-FG patch exceeds its max cosine to any reference-BG patch.
                 Bounds "pure feature matching, no training" on identical features (isolates what the
                 trained head adds).

Run on tulen (use the GPU with >=15GB free; do NOT disrupt the K-scaling on GPU0):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/disk1/prusek/active-segmenter HF_HOME=/disk1/prusek/.cache/huggingface \
    ~/dinov3_env/bin/python scripts/perseg_eval.py --method best_v2 --res 672 --superres 2 \
      --data "/disk1/prusek/incontext/PerSeg_data/data 3" \
      --cache /disk1/prusek/perseg_cache --out results/perseg_best_v2.json
"""
import argparse
import json
import os

import numpy as np
from PIL import Image
from skimage.transform import resize


# ─────────────────────────────── data ───────────────────────────────
def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path):
    """Binary foreground, matching eval_miou.py's ``cv2.cvtColor(...GRAY) > 0``."""
    return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8)


def object_list(data_dir):
    idir = os.path.join(data_dir, "Images")
    objs = sorted(o for o in os.listdir(idir)
                  if os.path.isdir(os.path.join(idir, o)) and "DS" not in o)
    return objs


def query_files(data_dir, obj, ref_idx="00"):
    idir = os.path.join(data_dir, "Images", obj)
    fs = sorted(f for f in os.listdir(idir) if f.endswith(".jpg"))
    return [f for f in fs if not f.startswith(ref_idx)]   # eval_miou skips the reference


# ─────────────────────────────── metrics ───────────────────────────────
def boundary_region(mask, d):
    from scipy.ndimage import binary_erosion
    if mask.sum() == 0:
        return np.zeros_like(mask, bool)
    er = binary_erosion(mask, iterations=d, border_value=1)   # border_value=1 keeps image-edge fg
    return mask & (~er)


def boundary_iou(pred, gt):
    """Boundary IoU (Cheng et al. 2021): band width d = 2% of the image diagonal."""
    h, w = gt.shape
    d = max(1, int(round(0.02 * float(np.hypot(h, w)))))
    pb = boundary_region(pred.astype(bool), d)
    gb = boundary_region(gt.astype(bool), d)
    inter = np.logical_and(pb, gb).sum()
    union = np.logical_or(pb, gb).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


# ─────────────────────────────── methods ───────────────────────────────
class BestV2:
    """Our best segmenter, fit on the K=1 reference; predict = semantic foreground."""

    def __init__(self, enc, dev, method_name="head_fusion_best_cgate_film"):
        import torch
        from scripts.al_testbed import make_backend
        from active_segmenter.config import RunConfig
        self.enc = enc
        self.dev = dev
        self.torch = torch
        self.cfg = RunConfig(device="auto")
        self.be = make_backend(method_name, self.cfg, dev, enc=enc)

    def fit(self, ref_img, ref_mask):
        from active_segmenter.segment.base import LabeledExample
        support = [LabeledExample(ref_img, self.enc.extract(ref_img), ref_mask.astype(int))]
        if hasattr(self.be, "head"):
            self.be.head = None                 # reset per object (no cross-object leakage)
        self.torch.manual_seed(0)
        self.be.fit(support)

    def predict(self, query_img):
        return self.be.foreground(query_img, self.enc.extract(query_img)).astype(np.uint8)


class DinoCorr:
    """Training-free dense-correspondence baseline on the frozen DINOv3 features (nearest-neighbour
    fg-vs-bg matching). No head, no training — bounds pure feature matching."""

    def __init__(self, enc, dev):
        import torch
        self.enc = enc
        self.dev = dev
        self.torch = torch
        self.fg = None
        self.bg = None

    @staticmethod
    def _l2(x):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

    def fit(self, ref_img, ref_mask):
        feat = np.asarray(self.enc.extract(ref_img), np.float32)     # [Gh, Gw, D]
        gh, gw = feat.shape[:2]
        gm = resize(ref_mask.astype(np.float32), (gh, gw), order=0,
                    mode="edge", anti_aliasing=False) > 0.5
        flat = feat.reshape(-1, feat.shape[-1])
        m = gm.reshape(-1)
        fg = flat[m]
        bg = flat[~m]
        if fg.shape[0] == 0:                    # degenerate ref mask → fall back to all-fg proto
            fg = flat
        if bg.shape[0] == 0:
            bg = flat
        self.fg = self.torch.from_numpy(self._l2(fg)).to(self.dev)   # [Nf, D]
        self.bg = self.torch.from_numpy(self._l2(bg)).to(self.dev)   # [Nb, D]

    def predict(self, query_img):
        feat = np.asarray(self.enc.extract(query_img), np.float32)   # [gh, gw, D]
        gh, gw = feat.shape[:2]
        q = self.torch.from_numpy(self._l2(feat.reshape(-1, feat.shape[-1]))).to(self.dev)  # [P, D]
        with self.torch.no_grad():
            s_fg = (q @ self.fg.T).max(dim=1).values      # [P] max cosine to any ref-fg patch
            s_bg = (q @ self.bg.T).max(dim=1).values
            lab = (s_fg > s_bg).cpu().numpy().reshape(gh, gw).astype(np.float32)
        up = resize(lab, query_img.shape[:2], order=0, mode="edge", anti_aliasing=False)
        return (up > 0.5).astype(np.uint8)


def build_method(name, res, superres, model_id, cache, dev):
    from active_segmenter.config import EncoderConfig, RunConfig
    from active_segmenter.encoder.cached import CachedEncoder
    cfg = RunConfig(device="auto", cache_dir=cache,
                    encoder=EncoderConfig(model_id=model_id, resolution=res,
                                          superres_factor=superres))
    enc = CachedEncoder(cfg, dev, cache)
    if name == "best_v2":
        return BestV2(enc, dev), enc
    if name == "dinov2_corr":
        return DinoCorr(enc, dev), enc
    raise ValueError(f"unknown method {name}")


# ─────────────────────────────── main ───────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["best_v2", "dinov2_corr"])
    ap.add_argument("--data", default="/disk1/prusek/incontext/PerSeg_data/data 3")
    ap.add_argument("--cache", default="/disk1/prusek/perseg_cache")
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--superres", type=int, default=2)
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--out", default="results/perseg_result.json")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N objects")
    args = ap.parse_args()

    from active_segmenter.config import RunConfig
    dev = RunConfig(device="auto").device_resolved()
    method, enc = build_method(args.method, args.res, args.superres, args.model, args.cache, dev)
    print(f"method={args.method} device={dev} res={args.res} superres={args.superres} "
          f"model={args.model}", flush=True)

    objs = object_list(args.data)
    if args.limit:
        objs = objs[:args.limit]

    per_obj = {}
    all_img_iou, all_img_biou = [], []
    for oi, obj in enumerate(objs):
        ref_img = load_rgb(os.path.join(args.data, "Images", obj, "00.jpg"))
        ref_mask = load_mask(os.path.join(args.data, "Annotations", obj, "00.png"))
        method.fit(ref_img, ref_mask)

        inter_sum = union_sum = target_sum = 0
        obj_img_iou, obj_img_biou = [], []
        for f in query_files(args.data, obj):
            q_img = load_rgb(os.path.join(args.data, "Images", obj, f))
            gt = load_mask(os.path.join(args.data, "Annotations", obj, f.replace(".jpg", ".png")))
            pred = method.predict(q_img)
            if pred.shape != gt.shape:          # safety: align to GT (should already match)
                pred = (resize(pred.astype(np.float32), gt.shape, order=0,
                               mode="edge", anti_aliasing=False) > 0.5).astype(np.uint8)
            inter = int(np.logical_and(pred, gt).sum())
            union = int(np.logical_or(pred, gt).sum())
            tgt = int(gt.sum())
            inter_sum += inter
            union_sum += union
            target_sum += tgt
            iou = 1.0 if union == 0 else inter / union
            biou = boundary_iou(pred, gt)
            obj_img_iou.append(iou)
            obj_img_biou.append(biou)
            all_img_iou.append(iou)
            all_img_biou.append(biou)

        iou_obj = inter_sum / (union_sum + 1e-10)        # eval_miou.py per-class IoU
        acc_obj = inter_sum / (target_sum + 1e-10)
        per_obj[obj] = dict(iou_accum=iou_obj, acc=acc_obj,
                            iou_meanimg=float(np.mean(obj_img_iou)),
                            biou_meanimg=float(np.mean(obj_img_biou)),
                            n_query=len(obj_img_iou))
        print(f"  [{oi+1:2d}/{len(objs)}] {obj:20s} "
              f"IoU(accum)={100*iou_obj:5.1f}  IoU(meanimg)={100*np.mean(obj_img_iou):5.1f}  "
              f"bIoU={100*np.mean(obj_img_biou):5.1f}  nq={len(obj_img_iou)}", flush=True)

    miou_accum = float(np.mean([v["iou_accum"] for v in per_obj.values()]))     # HEADLINE (eval_miou.py)
    macc = float(np.mean([v["acc"] for v in per_obj.values()]))
    miou_meanobj = float(np.mean([v["iou_meanimg"] for v in per_obj.values()]))
    mbiou_obj = float(np.mean([v["biou_meanimg"] for v in per_obj.values()]))

    summary = dict(method=args.method, res=args.res, superres=args.superres, model=args.model,
                   n_obj=len(objs), n_query_total=len(all_img_iou),
                   mIoU_accum_pct=100 * miou_accum, mAcc_pct=100 * macc,
                   mIoU_meanobj_pct=100 * miou_meanobj,
                   mIoU_meanimg_pct=100 * float(np.mean(all_img_iou)),
                   bIoU_meanobj_pct=100 * mbiou_obj,
                   bIoU_meanimg_pct=100 * float(np.mean(all_img_biou)),
                   per_obj=per_obj)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n================ PerSeg SUMMARY ================", flush=True)
    print(f"method                : {args.method}", flush=True)
    print(f"objects / queries     : {len(objs)} / {len(all_img_iou)}", flush=True)
    print(f"mIoU  (accum, eval_miou.py headline) : {100*miou_accum:.2f}", flush=True)
    print(f"mAcc  (accum)                        : {100*macc:.2f}", flush=True)
    print(f"mIoU  (mean-per-object of mean-img)  : {100*miou_meanobj:.2f}", flush=True)
    print(f"mIoU  (mean-per-image, flat)         : {100*float(np.mean(all_img_iou)):.2f}", flush=True)
    print(f"bIoU  (mean-per-object)              : {100*mbiou_obj:.2f}", flush=True)
    print(f"bIoU  (mean-per-image, flat)         : {100*float(np.mean(all_img_biou)):.2f}", flush=True)
    print(f"saved                 : {args.out}", flush=True)
    print("PERSEG_DONE", flush=True)


if __name__ == "__main__":
    main()
