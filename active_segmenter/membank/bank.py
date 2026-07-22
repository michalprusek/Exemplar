"""The memory bank — the non-parametric "model".

Per class, a set of :class:`InstanceExemplar`. Growing the bank (from oracle-
revealed corrections) IS the learning. Curation caps the bank per class via
k-center, which is also the AL diversity term.
"""
from __future__ import annotations

import base64
import json
from collections import defaultdict

import numpy as np
from skimage.transform import resize

from active_segmenter.membank.curation import kcenter
from active_segmenter.membank.exemplar import InstanceExemplar


def _mask_to_grid(mask: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """Downsample a boolean mask to a ``[gh, gw]`` boolean grid (bilinear > 0.5),
    matching the encoder-side fg/bg gridding. Supports non-square grids (native-res
    ConvNeXt features are ``[G0, G1]`` with ``G0 != G1``)."""
    r = resize(mask.astype(np.float32), (gh, gw), order=1, mode="edge", anti_aliasing=False)
    return r > 0.5


class MemoryBank:
    def __init__(self):
        self._by_class: dict[int, list[InstanceExemplar]] = defaultdict(list)

    # -- construction --------------------------------------------------------
    def add_from_annotation(self, feat_grid, label_map, class_of: dict[int, int], round: int):
        """Add one exemplar per annotated instance.

        ``feat_grid``: ``[G, G, D]`` features. ``label_map``: ``[H, W]`` int, one
        id per instance (0 = background). ``class_of``: maps instance id -> class id.
        """
        feat_grid = np.asarray(feat_grid, np.float32)
        gh, gw = feat_grid.shape[:2]
        flat = feat_grid.reshape(-1, feat_grid.shape[-1])
        bg_grid = _mask_to_grid(np.asarray(label_map) == 0, gh, gw).reshape(-1)
        bg_feats = flat[bg_grid]
        for inst_id, class_id in class_of.items():
            fg_grid = _mask_to_grid(np.asarray(label_map) == inst_id, gh, gw).reshape(-1)
            fg_feats = flat[fg_grid]
            if fg_feats.shape[0] == 0:  # instance smaller than one patch -> take nearest patch
                ys, xs = np.where(np.asarray(label_map) == inst_id)
                if len(ys) == 0:
                    continue
                gy = min(gh - 1, int(ys.mean() * gh / label_map.shape[0]))
                gx = min(gw - 1, int(xs.mean() * gw / label_map.shape[1]))
                fg_feats = feat_grid[gy, gx][None]
            self._by_class[class_id].append(
                InstanceExemplar(class_id, inst_id, fg_feats, bg_feats, None, round)
            )

    def add_from_grid_mask(self, feat_grid, fg_grid_bool, class_id: int, round: int):
        """Add one exemplar directly from a grid-resolution fg mask (no image-space
        round-trip). Used to insert PSEUDO-exemplars (the model's own prediction) for
        one-step look-ahead acquisition."""
        feat_grid = np.asarray(feat_grid, np.float32)
        flat = feat_grid.reshape(-1, feat_grid.shape[-1])
        m = np.asarray(fg_grid_bool, bool).reshape(-1)
        if not m.any():
            return
        self._by_class[class_id].append(
            InstanceExemplar(class_id, -1, flat[m], flat[~m], None, round)
        )

    def copy(self) -> "MemoryBank":
        b = MemoryBank()
        for cid, exs in self._by_class.items():
            b._by_class[cid] = list(exs)
        return b

    # -- queries -------------------------------------------------------------
    def classes(self) -> list[int]:
        return sorted(self._by_class.keys())

    def fg(self, class_id: int) -> np.ndarray:
        exs = self._by_class.get(class_id, [])
        if not exs:
            return np.empty((0, 0), np.float32)
        return np.concatenate([e.fg_feats for e in exs], 0)

    def bg(self, class_id: int) -> np.ndarray:
        exs = self._by_class.get(class_id, [])
        if not exs:
            return np.empty((0, 0), np.float32)
        return np.concatenate([e.bg_feats for e in exs], 0)

    def size(self, class_id: int) -> int:
        return len(self._by_class.get(class_id, []))

    def exemplars(self, class_id: int) -> list[InstanceExemplar]:
        return list(self._by_class.get(class_id, []))

    # -- curation ------------------------------------------------------------
    def curate(self, cap: int, seed: int = 0):
        """Cap each class to ``cap`` exemplars via k-center over exemplar centroids."""
        for class_id, exs in self._by_class.items():
            if len(exs) <= cap:
                continue
            cents = np.stack([e.centroid() for e in exs])
            keep = kcenter(cents, cap, seed=seed)
            self._by_class[class_id] = [exs[i] for i in keep]

    # -- (de)serialisation ---------------------------------------------------
    def to_json(self) -> str:
        def enc(a):
            a = np.ascontiguousarray(a, np.float32)
            return {"b64": base64.b64encode(a.tobytes()).decode(), "shape": list(a.shape)}

        payload = {
            str(cid): [
                {
                    "class_id": e.class_id,
                    "instance_id": e.instance_id,
                    "fg": enc(e.fg_feats),
                    "bg": enc(e.bg_feats),
                    "round": e.round,
                }
                for e in exs
            ]
            for cid, exs in self._by_class.items()
        }
        return json.dumps(payload)

    @classmethod
    def from_json(cls, s: str) -> "MemoryBank":
        def dec(d):
            arr = np.frombuffer(base64.b64decode(d["b64"]), np.float32)
            return arr.reshape(d["shape"]).copy()

        bank = cls()
        for cid, exs in json.loads(s).items():
            for e in exs:
                bank._by_class[int(cid)].append(
                    InstanceExemplar(
                        e["class_id"], e["instance_id"], dec(e["fg"]), dec(e["bg"]), None, e["round"]
                    )
                )
        return bank
