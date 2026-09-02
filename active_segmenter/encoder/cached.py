"""Disk-cached encoder adapter.

Wraps :class:`Dinov3Encoder` with the on-disk :class:`EmbeddingCache` so features
are computed once per (image, resolution, model) and reused across arms, refiners,
and separate benchmark invocations. Duck-typed like the bare encoder
(``.extract`` / ``.extract_cls``).
"""
from __future__ import annotations

import numpy as np

from active_segmenter.encoder.cache import EmbeddingCache
from active_segmenter.encoder.factory import cache_tag, make_encoder


class CachedEncoder:
    def __init__(self, cfg, device: str, cache_dir: str):
        self.enc = make_encoder(cfg.encoder, device)
        self.cache = EmbeddingCache(cache_dir)
        self.extra = cache_tag(cfg.encoder)

    def extract(self, image) -> np.ndarray:
        return self.cache.get_or_compute(image, self.extra, lambda: self.enc.extract(image))

    def extract_cls(self, image) -> np.ndarray:
        arr = self.cache.get_or_compute(
            image, "cls-" + self.extra, lambda: self.enc.extract_cls(image)[None]
        )
        return arr[0]
