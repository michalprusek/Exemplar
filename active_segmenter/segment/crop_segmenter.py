"""Native-resolution CROP segmentation pipeline.

Wraps a base foreground segmenter (head_fusion) to work at the encoder's native resolution on
overlapping crops instead of on downscaled whole images. Motivation (this session's #1 constraint):
whole-image inference downscales a 2048²/3504² image to the encoder res, so small crumbs and 1–2 px
vessels vanish. Cropping at native res preserves every pixel.

The design decouples the two sub-problems (see the design discussion 2026-07-12):
- **Resolution → per crop.** Train AND infer the base segmenter's SEMANTIC foreground on native-res
  crops (matched train/infer res — unlike the tiled-classical attempt that trained capped/inferred
  native and regressed). Blend overlapping crop probabilities with a feather window → one seamless
  full-res foreground. Foreground unions trivially across seams (a pixel is fg/bg regardless of crop).
- **Identity → global.** Run instance separation (blob-marker / watershed) ONCE on the stitched
  full-res foreground. Instances are defined globally, so an object cut by a crop seam is NOT
  fragmented or double-counted — the failure mode of naive per-crop instance segmentation.

Uses the RAW (uncached) encoder for the many transient crops, so it does not bloat the disk cache.
"""
from __future__ import annotations

import numpy as np

from active_segmenter.acquire.crop_tiles import tile_grid
from active_segmenter.segment.base import LabeledExample
from active_segmenter.segment.head_fusion_backend import (
    _blob_markers,
    _gray01,
    _watershed_instances,
)


def _feather(crop: int, floor: float = 0.08) -> np.ndarray:
    """2-D raised-cosine window (≈1 centre, tapering to ``floor`` at the edge) for seam-free blend."""
    w = np.hanning(crop)
    w = np.maximum(w, 0.0)
    win = np.outer(w, w).astype(np.float32)
    win = win / win.max()
    return floor + (1.0 - floor) * win


def _as_array(image) -> np.ndarray:
    return np.asarray(image)


class CropSegmenter:
    def __init__(self, base, encoder, crop: int = 672, overlap: float = 0.25,
                 max_crops_fit: int = 12, instance_mode: str = "blob",
                 min_ratio: float = 4.0, bg_frac: float = 0.35):
        self.base = base                 # HeadFusionBackend — trains on crop feat+classical+label
        self.enc = getattr(encoder, "enc", encoder)  # RAW encoder (bypass disk cache for crops)
        self.crop = crop
        self.overlap = overlap
        self.max_crops_fit = max_crops_fit    # cap labeled crops per image (tractable training)
        self.instance_mode = instance_mode
        # GATE: cropping only pays off when whole-image inference downscales heavily relative to the
        # crop. If the support's median max_side/crop < min_ratio, auto-fall back to whole-image
        # (train+infer on whole images) so wrapping in CropSegmenter is always safe (measured: crop
        # helps HRF at 5.2× but HURTS rozpad at 3.0× and dsb small).
        self.min_ratio = min_ratio
        self.bg_frac = bg_frac                # BALANCED fit: fraction of kept crops that are background
        self.passthrough = False              # set in fit() by the gate
        self._win = _feather(crop)

    # -- geometry --------------------------------------------------------------------------------
    def _crop_at(self, arr: np.ndarray, y: int, x: int):
        c = arr[y:y + self.crop, x:x + self.crop]
        ph, pw = self.crop - c.shape[0], self.crop - c.shape[1]
        if ph or pw:                                   # pad border crops to a full encoder tile
            pad = [(0, ph), (0, pw)] + ([(0, 0)] if arr.ndim == 3 else [])
            c = np.pad(c, pad, mode="reflect")
        return c

    def _gate(self, support) -> bool:
        """Crop only if the support's median image is large enough relative to the crop."""
        sides = [max(np.asarray(ex.image).shape[:2]) for ex in support]
        return bool(sides and float(np.median(sides)) / self.crop >= self.min_ratio)

    def _balanced(self, scored):
        """Pick up to max_crops_fit crops per image, BALANCED: mostly foreground-bearing crops but a
        ``bg_frac`` share of background crops so the head learns representative background too (the
        fg-only selection over-predicted foreground). ``scored`` = [(fg_frac, y, x, cl)] desc."""
        fg = [s for s in scored if s[0] > 0.005]
        bg = [s for s in scored if s[0] <= 0.005]
        n_bg = min(len(bg), int(round(self.max_crops_fit * self.bg_frac)))
        n_fg = min(len(fg), self.max_crops_fit - n_bg)
        keep = fg[:n_fg]
        if n_bg and bg:                                 # spread the background picks across the image
            keep += [bg[i] for i in np.linspace(0, len(bg) - 1, n_bg).round().astype(int)]
        return keep

    # -- fit: label crops, train the base head at native res -------------------------------------
    def fit(self, support: list[LabeledExample]) -> None:
        self.passthrough = not self._gate(support)
        if self.passthrough:                            # near-native / small imgs → whole-image path
            self.base.fit(support)
            return
        exs: list[LabeledExample] = []
        for ex in support:
            img = _as_array(ex.image)
            lab = (np.asarray(ex.label_map) > 0).astype(np.uint8)
            h, w = img.shape[:2]
            scored = sorted(((float(self._crop_at(lab, y, x).mean()), y, x)
                             for (y, x) in tile_grid(h, w, self.crop, self.overlap)),
                            key=lambda t: -t[0])
            for _fg, y, x in self._balanced([(f, yy, xx) for f, yy, xx in scored]):
                ci = self._crop_at(img, y, x)
                cl = self._crop_at(lab, y, x)
                exs.append(LabeledExample(ci, self.enc.extract(ci), cl.astype(int)))
        if exs:
            self.base.fit(exs)

    # -- inference: stitch a seamless full-res foreground ----------------------------------------
    def foreground(self, image, feat_grid=None, class_id: int = 1) -> np.ndarray:
        if self.passthrough:                            # gate said whole-image
            return self.base.foreground(image, feat_grid, class_id)
        img = _as_array(image)
        h, w = img.shape[:2]
        acc = np.zeros((h, w), np.float32)
        wsum = np.zeros((h, w), np.float32)
        for (y, x) in tile_grid(h, w, self.crop, self.overlap):
            ci = self._crop_at(img, y, x)
            prob = self.base.foreground_prob(ci, self.enc.extract(ci))   # [crop, crop] native
            ph = min(self.crop, h - y)
            pw = min(self.crop, w - x)
            acc[y:y + ph, x:x + pw] += (prob * self._win)[:ph, :pw]
            wsum[y:y + ph, x:x + pw] += self._win[:ph, :pw]
        return (acc / np.maximum(wsum, 1e-6)) > 0.5

    # -- inference: instance separation ONCE, globally, on the stitched foreground ---------------
    def predict(self, image, feat_grid=None, class_id: int = 1):
        if self.passthrough:
            return self.base.predict(image, feat_grid, class_id)
        fg = self.foreground(image, feat_grid, class_id)
        markers = _blob_markers(fg, _gray01(image)) if self.instance_mode == "blob" else None
        return _watershed_instances(fg, class_id, markers=markers)

    def score_map(self, image, feat_grid=None, class_id: int = 1) -> np.ndarray:
        g = 32
        return np.zeros((g, g), np.float32)
