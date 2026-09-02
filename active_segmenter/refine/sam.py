"""SAM2 promptable refiner.

Each coarse instance mask is turned into a crisp native-resolution mask by prompting
SAM2 with a POSITIVE point at the instance's most-interior pixel (distance-transform
peak) plus NEGATIVE points at nearby sibling instances (so SAM separates touching
objects). The image backbone runs once per image (``get_image_embeddings``) and is
reused across all instance prompts. Each instance stays its own mask — overlap safe.

transformers 4.57 exposes ``Sam2Model``/``Sam2Processor`` for
``facebook/sam2.1-hiera-large`` (there is no ``Sam3`` class in this version).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage
from skimage.transform import resize

from active_segmenter.config import RefineConfig
from active_segmenter.types import InstanceMask

_MAX_NEG = 4  # nearest sibling peaks used as negative prompts


class SamRefiner:
    def __init__(self, cfg: RefineConfig, device: str):
        from transformers import Sam2Model, Sam2Processor

        self.cfg = cfg
        self.device = device
        self.processor = Sam2Processor.from_pretrained(cfg.sam_id)
        self.model = Sam2Model.from_pretrained(cfg.sam_id).eval().to(device)

    def _to_native(self, m: InstanceMask, hw) -> np.ndarray:
        if m.mask.shape == tuple(hw):
            return m.mask
        return resize(m.mask.astype(np.float32), tuple(hw), order=0, mode="edge",
                      anti_aliasing=False) > 0.5

    def refine(self, image, instance_masks, feat_grid: Optional[np.ndarray] = None):
        import torch

        img = np.asarray(image)
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        img = img[..., :3].astype(np.uint8)
        h, w = img.shape[:2]
        if not instance_masks:
            return []

        native = [self._to_native(m, (h, w)) for m in instance_masks]
        peaks = [_peak_xy(mask) for mask in native]  # (x, y) or None

        base = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model.get_image_embeddings(base["pixel_values"])

        out = []
        for i, m in enumerate(instance_masks):
            if peaks[i] is None:
                out.append(InstanceMask(mask=native[i], points=None, class_id=m.class_id,
                                        instance_id=m.instance_id, score=m.score))
                continue
            pts = [list(peaks[i])]
            lbls = [1]
            use_neg = self.cfg.sam_negatives and not self.cfg.amodal  # amodal keeps overlaps
            if use_neg:  # negatives separate touching instances
                for j in _nearest_siblings(i, peaks, _MAX_NEG):
                    pts.append(list(peaks[j]))
                    lbls.append(0)
            proc_kw = dict(images=img, input_points=[[pts]], input_labels=[[lbls]],
                           return_tensors="pt")
            if self.cfg.prompt_mode == "mask_box":
                bb = mask_bbox_xyxy(native[i])
                if bb is not None:
                    proc_kw["input_boxes"] = [[bb]]
            pinp = self.processor(**proc_kw).to(self.device)
            model_kw = dict(image_embeddings=emb, input_points=pinp["input_points"],
                            input_labels=pinp["input_labels"], multimask_output=False)
            if "input_boxes" in pinp:
                model_kw["input_boxes"] = pinp["input_boxes"]
            if self.cfg.prompt_mode in ("mask", "mask_box"):
                lo = mask_to_lowres_logits(native[i])  # [256, 256] shape prompt
                # SAM2 mask encoder wants [batch, 1, H, W]
                model_kw["input_masks"] = torch.as_tensor(lo)[None, None].to(self.device)
            with torch.no_grad():
                res = self.model(**model_kw)
            masks = self.processor.post_process_masks(res.pred_masks.cpu(), pinp["original_sizes"])
            m0 = np.asarray(masks[0]).reshape(-1, h, w)[0].astype(bool)
            score = float(res.iou_scores.reshape(-1)[0])
            out.append(InstanceMask(mask=m0, points=None, class_id=m.class_id,
                                    instance_id=m.instance_id, score=score))
        return out


def mask_to_lowres_logits(mask, size: int = 256, pos: float = 8.0, neg: float = -8.0):
    """Coarse instance mask -> SAM low-res mask-prompt logits ``[size, size]`` (inside=pos).

    A mask prompt carries the whole proposal shape (not one pixel), so SAM returns tighter
    instance boundaries — the direct lever on instance-AP.
    """
    from skimage.transform import resize as _resize

    m = _resize(np.asarray(mask, np.float32), (size, size), order=0, mode="edge",
                anti_aliasing=False) > 0.5
    return np.where(m, pos, neg).astype(np.float32)


def mask_bbox_xyxy(mask):
    """Tight ``[x0, y0, x1, y1]`` of the mask, or None if empty."""
    ys, xs = np.where(np.asarray(mask))
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _peak_xy(mask: np.ndarray):
    """Most-interior pixel of ``mask`` as ``(x, y)`` (distance-transform argmax)."""
    if not mask.any():
        return None
    dt = ndimage.distance_transform_edt(mask)
    yy, xx = np.unravel_index(int(np.argmax(dt)), dt.shape)
    return int(xx), int(yy)


def _nearest_siblings(i: int, peaks, k: int) -> list[int]:
    pi = peaks[i]
    others = [(j, p) for j, p in enumerate(peaks) if j != i and p is not None]
    others.sort(key=lambda jp: (jp[1][0] - pi[0]) ** 2 + (jp[1][1] - pi[1]) ** 2)
    return [j for j, _ in others[:k]]
