"""Frozen DINOv3-ConvNeXt dense feature extractor — the convolutional alternative.

The ViT backbone's patch stride 16 caps the feature grid at ``resolution//16`` (e.g.
42x42 at res 672), so sub-patch objects (the small "crumbs" on decay spheroids) fall
between grid cells. A ConvNeXt backbone is hierarchical: its stage feature maps have
strides 4/8/16/32, so at input 1024 a stride-8 map is 128x128 — a ~9x finer grid than
the ViT — recovering small structures. Same DINOv3 SSL pretraining, convolutional trunk.

Duck-typed like :class:`Dinov3Encoder` (``extract`` -> ``[G0, G1, D]`` L2-normalised,
``extract_cls`` -> ``[D]``), so it drops into the same cache + backends unchanged.
``convnext_stage`` indexes HF ``hidden_states``: 1=stride4, 2=stride8, 3=stride16, 4=stride32.
"""
from __future__ import annotations

import numpy as np

from active_segmenter.config import EncoderConfig
from active_segmenter.encoder.dinov3 import _IMAGENET_MEAN, _IMAGENET_STD, _to_rgb01


class Dinov3ConvNeXtEncoder:
    def __init__(self, cfg: EncoderConfig, device: str):
        import torch
        from transformers import AutoModel

        self.cfg = cfg
        self.device = device
        self._torch = torch
        self.stage = cfg.convnext_stage
        self.model = AutoModel.from_pretrained(cfg.model_id).eval().to(device)
        self._mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
        self._std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1).to(device)

    _NATIVE_CAP = 2048  # guard: cap the long side so a huge image doesn't OOM ConvNeXt-L

    def _target_hw(self, image: np.ndarray):
        """Target (H, W) fed to ConvNeXt. ``resolution > 0`` = fixed square (downsample);
        ``resolution <= 0`` = NATIVE — the image's own size snapped to a multiple of 32
        (ConvNeXt total stride) and capped, so a fully-convolutional pass segments at native
        resolution with a non-square feature grid."""
        r = self.cfg.resolution
        if r > 0:
            return r, r
        h, w = np.asarray(image).shape[:2]

        def snap(x):
            return int(max(32, min(self._NATIVE_CAP, round(x / 32) * 32)))

        return snap(h), snap(w)

    def _preprocess(self, image: np.ndarray):
        torch = self._torch
        a = _to_rgb01(image)
        t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(self.device)
        t = torch.nn.functional.interpolate(
            t, size=self._target_hw(image), mode="bilinear", align_corners=False
        )
        return (t - self._mean) / self._std

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Dense L2-normalised ``[G0, G1, D]`` features from ConvNeXt stage ``convnext_stage``
        (non-square when native)."""
        torch = self._torch
        with torch.no_grad():
            out = self.model(pixel_values=self._preprocess(image), output_hidden_states=True)
        fmap = out.hidden_states[self.stage][0]           # [C, G0, G1]
        fmap = torch.nn.functional.normalize(fmap, dim=0)  # L2 over channels -> cosine = dot
        return fmap.permute(1, 2, 0).float().cpu().numpy()  # [G0, G1, C]

    def extract_cls(self, image: np.ndarray) -> np.ndarray:
        """Global descriptor for cold start: L2-normalised pooled embedding ``[D]``."""
        torch = self._torch
        with torch.no_grad():
            out = self.model(pixel_values=self._preprocess(image))
        pooled = out.pooler_output[0] if out.pooler_output is not None \
            else out.last_hidden_state[0].mean(0)
        return torch.nn.functional.normalize(pooled, dim=0).float().cpu().numpy()
