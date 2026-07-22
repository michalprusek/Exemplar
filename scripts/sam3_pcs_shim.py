"""SAM 3 text-prompted Promptable-Concept-Segmentation shim.

Runs ONLY in the isolated ``~/sam3_env`` (transformers 5.13.x with SAM 3). ``segment_pcs`` takes a
query image and a concept string and returns every instance of that concept as a boolean mask —
SAM 3's flagship zero-shot capability, used here as a producer to race against the in-context
head + amodal-SAM refine. The DINOv3 (transformers 4.57) env can NOT import this — it is called
across the subprocess boundary by ``scripts/sam3_worker.py``.
"""
from __future__ import annotations

import numpy as np

_MODEL = None
_PROC = None


def _load(device: str):
    global _MODEL, _PROC
    if _MODEL is None:
        from transformers import Sam3Model, Sam3Processor

        _PROC = Sam3Processor.from_pretrained("facebook/sam3")
        _MODEL = Sam3Model.from_pretrained("facebook/sam3").eval().to(device)
    return _MODEL, _PROC


def _rgb_uint8(query_image):
    a = np.asarray(query_image)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    return a[..., :3].astype(np.uint8)


def _masks_from_result(res, h, w):
    masks = res["masks"] if "masks" in res else res.get("segmentation")
    if masks is None or len(masks) == 0:
        return np.zeros((0, h, w), bool)
    if hasattr(masks, "detach"):  # a [N,H,W] CUDA tensor
        masks = masks.detach().cpu().numpy()
    else:  # a list of tensors/arrays
        masks = np.asarray([np.asarray(m.detach().cpu() if hasattr(m, "detach") else m)
                            for m in masks])
    if masks.ndim == 2:
        masks = masks[None]
    return (masks > 0.5).astype(bool) if masks.dtype != bool else masks


def segment_pcs(query_image, text: str, device: str = "cuda", threshold: float = 0.3):
    """Segment every instance of concept ``text`` in ``query_image`` → ``[N, H, W]`` bool masks."""
    import torch
    from PIL import Image

    a = _rgb_uint8(query_image)
    h, w = a.shape[:2]
    model, proc = _load(device)
    inputs = proc(images=Image.fromarray(a), text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    res = proc.post_process_instance_segmentation(
        outputs, threshold=threshold, target_sizes=[(h, w)]
    )[0]
    return _masks_from_result(res, h, w)


def segment_pcs_boxes(query_image, boxes, labels=None, device: str = "cuda",
                      threshold: float = 0.3):
    """VISUAL exemplar PCS: ``boxes`` (**XYXY absolute pixel** ``[x0,y0,x1,y1]``) define the concept
    by example on the query image; SAM 3 segments every matching instance → ``[N, H, W]`` bool masks.
    Box-only works (no text needed). Empirically XYXY-absolute is required (normalized/cxcywh → 0
    detections). Tests whether visual exemplars unlock niche morphology where TEXT concepts fail."""
    import torch
    from PIL import Image

    a = _rgb_uint8(query_image)
    h, w = a.shape[:2]
    if len(boxes) == 0:
        return np.zeros((0, h, w), bool)
    if labels is None:
        labels = [1] * len(boxes)
    model, proc = _load(device)
    inputs = proc(
        images=Image.fromarray(a),
        input_boxes=[[[float(x) for x in b] for b in boxes]],
        input_boxes_labels=[[int(l) for l in labels]],
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    res = proc.post_process_instance_segmentation(
        outputs, threshold=threshold, target_sizes=[(h, w)]
    )[0]
    return _masks_from_result(res, h, w)
