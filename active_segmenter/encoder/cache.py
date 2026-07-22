"""Disk-backed embedding cache.

DINOv3 feature extraction is the expensive step; the AL loop re-prelabels the
whole pool every round, so features must be computed once per (image, resolution,
model) and reused. Keyed by a content hash of the image bytes plus an ``extra``
string (carries resolution / model id), stored as ``.npy``.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from typing import Callable

import numpy as np


class EmbeddingCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _key(self, image: np.ndarray, extra: str) -> str:
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(image).tobytes())
        h.update(str(image.shape).encode())
        h.update(str(image.dtype).encode())
        h.update(extra.encode())
        return h.hexdigest()

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, key + ".npy")

    def get_or_compute(
        self, image: np.ndarray, extra: str, fn: Callable[[], np.ndarray]
    ) -> np.ndarray:
        path = self._path(self._key(image, extra))
        if os.path.exists(path):
            return np.load(path)
        arr = np.asarray(fn(), dtype=np.float32)
        # The temp file must be unique per WRITER, not per key. Two concurrent writers that miss the
        # same key would otherwise open, truncate and write one shared ``.tmp`` at once, and the loser's
        # ``os.replace`` then dies with FileNotFoundError because the winner already renamed it away.
        # ``mkstemp`` is unique across both processes and threads; ``os.replace`` still publishes it
        # atomically, so a reader never observes a half-written entry.
        fd, tmp = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:  # write to exact path; np.save appends .npy to names
                np.save(fh, arr)
            os.replace(tmp, path)  # atomic — no half-written cache files
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return arr
