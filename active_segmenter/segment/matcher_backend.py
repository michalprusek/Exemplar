"""Matcher one-shot exemplar segmenter (Liu et al., ICLR 2024) as a segmenter backend for the panel.

Matcher ("Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching", aim-uofa/Matcher)
is a TRAINING-FREE one-shot segmenter: a frozen DINOv2 does patch-level feature matching between ONE
annotated support (image, mask) pair and the query, the matched points prompt a frozen SAM automatic
mask generator, and the candidate masks are scored/merged into the query foreground. It is the closest
EXTERNAL architectural analog to our frozen-backbone method (frozen features + light matching, no
training), and it has never been evaluated on microscopy — this backend runs it on our panel.

Native regime is one-shot (K=1); it also supports K-shot by stacking multiple references
(``set_reference`` accepts nshot>1), so we run K=1,4,8 exactly as Matcher's own K-shot scripts do.
Semantic foreground only → instances via connected components (its honest limitation on touching
objects, same as any semantic-only method without a separation stage).

FAIRNESS: we use Matcher's OWN published inference config — the FSS-1000 one-shot config
(``scripts/fss.sh``: sample-range (4,6), max_sample_iterations 30, multimask_output 0,
alpha 0.8 beta 0.2 exp 1.0, num_merging_mask 10, DINOv2 ViT-L, SAM ViT-H, points_per_side 64), which
is the canonical one-shot SEMANTIC-segmentation setting and the closest of Matcher's configs to our
microscopy foreground task. For K>1 we follow ``scripts/fss_5shot.sh`` (same config, num_merging_mask 5).
NO per-dataset tuning, NO GT/oracle mask selection — Matcher's default scoring picks and merges the
masks. Images are resized to Matcher's native 518×518 (bilinear, min-max to [0,1] like our other
baselines), masks nearest; the prediction is resized back to native resolution and thresholded.

Point the backend at the cloned repo via ``MATCHER_SRC`` (default: the tulen checkout) and the local
DINOv2/SAM weights via ``MATCHER_DINOV2_WEIGHTS`` / ``MATCHER_SAM_WEIGHTS``."""
from __future__ import annotations

import os
import sys
import types

import numpy as np

from active_segmenter.types import InstanceMask

MATCHER_SRC = os.environ.get("MATCHER_SRC", "/disk1/prusek/incontext/Matcher")
DINOV2_WEIGHTS = os.environ.get(
    "MATCHER_DINOV2_WEIGHTS", os.path.join(MATCHER_SRC, "models/dinov2_vitl14_pretrain.pth"))
SAM_WEIGHTS = os.environ.get(
    "MATCHER_SAM_WEIGHTS", os.path.join(MATCHER_SRC, "models/sam_vit_h_4b8939.pth"))


def _rgb01(image, size: int) -> np.ndarray:
    """Native image → 3-channel float [0,1] at size×size (bilinear). Min-max normalized per image,
    the same contrast handling our UniverSeg/Tyche baselines use (robust to 8/16-bit microscopy)."""
    from skimage.transform import resize
    a = np.asarray(image).astype(np.float32)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=2)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    elif a.ndim == 3 and a.shape[2] >= 3:
        a = a[:, :, :3]
    a = (a - a.min()) / (np.ptp(a) + 1e-6)
    return resize(a, (size, size, 3), order=1, mode="edge", anti_aliasing=True).astype(np.float32)


def _mask_at(label, size: int) -> np.ndarray:
    """GT/support mask → binary float {0,1} at size×size (nearest, no smoothing across the boundary)."""
    from skimage.transform import resize
    m = (np.asarray(label) > 0).astype(np.float32)
    return (resize(m, (size, size), order=0, mode="edge") > 0.5).astype(np.float32)


def _matcher_args(device, num_merging_mask: int):
    """Matcher's FSS one-shot config as an argparse-like namespace (see scripts/fss.sh). ``eval``-ed
    sample_range is passed as a tuple; num_merging_mask is 10 (1-shot) / 5 (K-shot)."""
    return types.SimpleNamespace(
        dinov2_size="vit_large", sam_size="vit_h",
        dinov2_weights=DINOV2_WEIGHTS, sam_weights=SAM_WEIGHTS, device=device,
        # --- FSS one-shot config (scripts/fss.sh) ---
        max_sample_iterations=30, sample_range=(4, 6), multimask_output=0,
        alpha=0.8, beta=0.2, exp=1.0, num_merging_mask=num_merging_mask,
        # --- argparse defaults (unchanged from Matcher) ---
        points_per_side=64, pred_iou_thresh=0.88, stability_score_thresh=0.95,
        sel_stability_score_thresh=0.0, iou_filter=0.0, box_nms_thresh=1.0, output_layer=3,
        dense_multimask_output=0, use_dense_mask=0, num_centers=8, use_box=False,
        use_points_or_centers=False, emd_filter=0.0, purity_filter=0.0, coverage_filter=0.0,
        use_score_filter=False, deep_score_norm_filter=0.1, deep_score_filter=0.33,
        topk_scores_threshold=0.7,
    )


class MatcherBackend:
    """fit(support) stores the support set; foreground/predict run Matcher per query. Matches the
    SegmenterBackend interface (fit / foreground / predict) so it drops into the panel harness exactly
    like UniverSegBackend. The heavy models (DINOv2 + SAM) are built once, lazily, on first fit."""

    # `fit` REASSIGNS `_sup` wholesale and re-derives `num_merging_mask` from the new draw's size, so
    # nothing carries over. The lazily built models are frozen weights, not support state.
    stateless_support = True

    def __init__(self, device=None, input_size: int = 518, max_support: int = 16):
        self.device = device or "cpu"
        self.input_size = input_size
        self.max_support = max_support        # K-shot cap (Matcher stacks references; keep bounded)
        self._matcher = None
        self._sup = None                      # (imgs [1,ns,3,S,S], masks [1,ns,S,S]) tensors on device

    def _build(self):
        if self._matcher is None:
            if MATCHER_SRC not in sys.path:
                sys.path.insert(0, MATCHER_SRC)     # vendored dinov2 + Matcher-modified segment_anything
            # numpy-2.0 compat: Matcher's 2023 code (Matcher.py:368) uses the removed alias ``np.int``.
            # Restore the deprecated builtin aliases (identical to their <2.0 meaning) so the frozen
            # Matcher inference runs unchanged under this env's numpy 2.x. Process-local (isolated run).
            for _alias, _typ in (("int", int), ("float", float), ("bool", bool), ("object", object),
                                 ("str", str), ("complex", complex), ("long", int), ("unicode", str)):
                if not hasattr(np, _alias):
                    setattr(np, _alias, _typ)
            import torch  # noqa: F401
            from matcher.Matcher import build_matcher_oss
            import torch as _t
            dev = _t.device(self.device) if not isinstance(self.device, _t.device) else self.device
            self._matcher = build_matcher_oss(_matcher_args(dev, num_merging_mask=10))
        return self._matcher

    def fit(self, support) -> None:
        import torch
        self._build()
        sup = support[: self.max_support]
        S = self.input_size
        imgs = np.stack([_rgb01(ex.image, S).transpose(2, 0, 1) for ex in sup])   # [ns,3,S,S]
        masks = np.stack([_mask_at(ex.label_map, S) for ex in sup])               # [ns,S,S]
        dev = self._matcher.device
        self._sup = (torch.from_numpy(imgs)[None].float().to(dev),                # [1,ns,3,S,S]
                     torch.from_numpy(masks)[None].float().to(dev))               # [1,ns,S,S]
        # K-shot follows fss_5shot.sh (num_merging_mask 5); 1-shot follows fss.sh (10).
        self._matcher.num_merging_mask = 5 if len(sup) > 1 else 10

    def _prob_mask(self, image) -> np.ndarray:
        import torch
        from skimage.transform import resize
        S = self.input_size
        q = torch.from_numpy(_rgb01(image, S).transpose(2, 0, 1))[None].float().to(self._matcher.device)
        self._matcher.set_reference(self._sup[0], self._sup[1])
        self._matcher.set_target(q)                                               # [1,3,S,S]
        with torch.no_grad():
            pred = self._matcher.predict()                                        # [num_masks or 1, S, S]
        self._matcher.clear()
        pm = pred.detach().cpu().numpy()
        if pm.ndim == 3:                                                          # merge any candidate dim
            pm = (pm > 0.5).any(axis=0).astype(np.float32)
        pm = (np.asarray(pm) > 0.5).astype(np.float32)
        return resize(pm, np.asarray(image).shape[:2], order=0, mode="edge") > 0.5

    def foreground(self, image, feat_grid, class_id: int = 1) -> np.ndarray:
        if self._sup is None:
            return np.zeros(np.asarray(image).shape[:2], bool)
        # FAIL-LOUD: a legitimate "no mask" comes through _prob_mask's normal path (empty prediction ->
        # all-false); the only thing a try/except would catch here is a real crash (CUDA OOM, model/code
        # error). Silently scoring that as an empty mask would deflate Matcher's numbers, so we let it
        # propagate and abort the run rather than average a crash in as a 0.
        return np.asarray(self._prob_mask(image), dtype=bool)

    def predict(self, image, feat_grid, class_id: int = 1) -> list[InstanceMask]:
        from skimage.measure import label
        fg = self.foreground(image, feat_grid, class_id)
        lab = label(fg)                                                           # semantic → CC instances
        out = []
        for i in range(1, int(lab.max()) + 1):
            mm = lab == i
            if mm.any():
                out.append(InstanceMask(mask=mm, points=None, class_id=class_id,
                                        instance_id=i, score=1.0))
        return out
