"""SAM 3 PCS backend, ISOLATED. ``transformers==4.57`` in the main env cannot load SAM 3,
and upgrading it would break the DINOv3 path. So this backend never imports SAM 3
in-process: it serialises ``(support image+mask, query image)`` to an NPZ, invokes an
isolated-env python running ``scripts/sam3_worker.py``, and reads back per-instance masks.
A missing env/worker raises :class:`BackendUnavailable` (the race skips SAM 3 and still
reports the other backends).
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

from active_segmenter.segment.base import BackendUnavailable
from active_segmenter.types import InstanceMask

_DEFAULT_PY = os.environ.get("SAM3_PYTHON", "/disk1/prusek/sam3_env/bin/python")


class Sam3PcsBackend:
    # `fit` REASSIGNS `_support` unconditionally (to None when no shot has foreground), so no exemplar
    # from a previous draw can survive.
    stateless_support = True

    def __init__(self, device: str | None = None, python_bin: str | None = None,
                 worker: str = "scripts/sam3_worker.py"):
        self.device = device
        self.python_bin = python_bin or _DEFAULT_PY
        self.worker = worker
        self._support = None

    def available(self) -> bool:
        return os.path.exists(self.python_bin) and os.path.exists(self.worker)

    def fit(self, support) -> None:
        # PCS conditions on ONE exemplar; keep the first labeled example with foreground.
        self._support = next(
            (ex for ex in support if (np.asarray(ex.label_map) > 0).any()), None
        )

    def score_map(self, image, feat_grid, class_id: int = 1) -> np.ndarray:
        # SAM 3 emits masks directly; there is no dense correspondence score. Return a
        # zero grid so the harness's fg-IoU convention still has a shape to work with.
        g = np.asarray(feat_grid).shape[0] if feat_grid is not None else 32
        return np.zeros((g, g), np.float32)

    def predict(self, image, feat_grid, class_id: int = 1) -> list[InstanceMask]:
        if not self.available() or self._support is None:
            raise BackendUnavailable(f"SAM 3 env/worker not available at {self.python_bin}")
        with tempfile.TemporaryDirectory() as td:
            inp, outp = os.path.join(td, "in.npz"), os.path.join(td, "out.npz")
            np.savez(inp, support_image=np.asarray(self._support.image),
                     support_mask=(np.asarray(self._support.label_map) > 0),
                     query_image=np.asarray(image))
            r = subprocess.run([self.python_bin, self.worker, inp, outp],
                               capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(outp):
                raise BackendUnavailable(f"SAM 3 worker failed: {r.stderr[-500:]}")
            data = np.load(outp)
            masks = data["masks"]  # [N, H, W] bool
        return [InstanceMask(mask=masks[i].astype(bool), points=None, class_id=class_id,
                             instance_id=i, score=1.0) for i in range(len(masks))]

    def foreground(self, image, feat_grid, class_id: int = 1) -> np.ndarray:
        insts = self.predict(image, feat_grid, class_id)
        hw = np.asarray(image).shape[:2]
        fg = np.zeros(hw, bool)
        for m in insts:
            fg |= m.mask
        return fg
