"""HyperBank segmenter backend — classical native-res bank, optionally fused with DINOv3 semantics.

Two modes behind one backend:
- ``fusion=False``: the vendored classical bank alone (native-resolution Frangi/LoG/Sauvola/… + a
  306-param head), trained on the support set. Resolves small blobs a patch-16 foundation grid
  cannot (the HyperBank result, reproduced in-harness).
- ``fusion=True``: adds ONE DINOv3 **semantic-saliency** meta-channel (the correspondence score-map
  vs the support foreground, upsampled to native) into the head, so classical structure responses
  are gated by semantics. The head can zero that channel → fusion cannot degrade the classical
  result and only helps where semantics disambiguate (clutter / artifacts / OOD).

Consumes the ``SegmenterBackend`` interface (``fit``/``foreground``/``predict``); the native image is
``LabeledExample.image`` and the grid features are ``.feat_grid``.
"""
from __future__ import annotations

import numpy as np

from active_segmenter.config import MatchConfig
from active_segmenter.membank.bank import MemoryBank
from active_segmenter.propose import correspondence as corr
from active_segmenter.segment.hyperbank_bank import HyperBank
from active_segmenter.types import InstanceMask

_MC = MatchConfig()


def _gray01(image) -> np.ndarray:
    a = np.asarray(image).astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    return (a - a.min()) / (np.ptp(a) + 1e-6)


def _soft_dice(pred, target, eps=1e-6):
    inter = (pred * target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)


class HyperBankBackend:
    def __init__(self, device: str, fusion: bool = False, lr: float = 0.05, epochs_base: int = 30):
        self.device = device
        self.fusion = fusion
        self.lr = lr
        self.epochs_base = epochs_base
        self.bank = MemoryBank()
        self.model = None
        self._min_area = 200

    def _build(self):
        return HyperBank(
            frangi_sigmas=(1.0, 2.0, 4.0, 8.0, 16.0),
            sauvola_windows=(15, 51, 151),
            struct_sigmas=(2.0, 8.0),
            use_log=True,  # Ω: LoG blob detector — the small-crumb lever
            n_meta_channels=1 if self.fusion else 0,
        ).to(self.device)

    def _saliency(self, image, feat_grid) -> np.ndarray:
        """DINOv3 semantic saliency: correspondence score-map vs support fg, upsampled to native [0,1]."""
        from skimage.transform import resize

        s = np.asarray(corr.score_map(feat_grid, self.bank, 1, _MC, device=self.device), np.float32)
        h, w = np.asarray(image).shape[:2]
        s = resize(s, (h, w), order=1, mode="edge", anti_aliasing=True)
        return (s - s.min()) / (np.ptp(s) + 1e-6)

    def _tensors(self, ex, torch):
        img = torch.from_numpy(_gray01(ex.image))[None, None].to(self.device)
        m = torch.from_numpy((np.asarray(ex.label_map) > 0).astype(np.float32))[None, None].to(self.device)
        meta = None
        if self.fusion:
            meta = torch.from_numpy(self._saliency(ex.image, ex.feat_grid).astype(np.float32))
            meta = meta[None, None].to(self.device)
        return img, m, meta

    def fit(self, support) -> None:
        import torch
        from skimage.measure import label, regionprops

        if self.fusion:
            self.bank = MemoryBank()
            for ex in support:
                fg = (np.asarray(ex.label_map) > 0).astype(int)
                self.bank.add_from_annotation(ex.feat_grid, fg, {1: 1} if fg.any() else {}, 0)

        areas = [r.area for ex in support for r in regionprops(label(np.asarray(ex.label_map) > 0))]
        if areas:
            self._min_area = max(200, int(min(areas) * 0.4))

        data = [self._tensors(ex, torch) for ex in support]  # saliency computed once (bank fixed)
        self.model = self._build()
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        rng = np.random.default_rng(0)
        n_epochs = self.epochs_base + max(0, len(support) - 1) * 5
        for _ in range(n_epochs):
            order = rng.permutation(len(data))
            for i in order:
                img, m, meta = _augment(*data[i], rng, torch)
                pred = self.model(img, meta)
                loss = (torch.nn.functional.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), m)
                        + (1.0 - _soft_dice(pred, m)))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        self.model.eval()

    def _prob(self, image, feat_grid) -> np.ndarray:
        import torch

        img = torch.from_numpy(_gray01(image))[None, None].to(self.device)
        meta = None
        if self.fusion:
            meta = torch.from_numpy(self._saliency(image, feat_grid).astype(np.float32))
            meta = meta[None, None].to(self.device)
        with torch.no_grad():
            return self.model(img, meta)[0, 0].cpu().numpy()

    def foreground(self, image, feat_grid, class_id: int = 1) -> np.ndarray:
        return self._prob(image, feat_grid) > 0.5

    def predict(self, image, feat_grid, class_id: int = 1):
        from skimage.measure import label

        lab = label(self._prob(image, feat_grid) > 0.5)
        out = []
        for i in range(1, int(lab.max()) + 1):
            m = lab == i
            if int(m.sum()) >= self._min_area:
                out.append(InstanceMask(mask=m, points=None, class_id=class_id,
                                        instance_id=len(out), score=1.0))
        return out

    def score_map(self, image, feat_grid, class_id: int = 1) -> np.ndarray:
        g = np.asarray(feat_grid).shape[0] if feat_grid is not None else 32
        return np.zeros((g, g), np.float32)


def _augment(img, m, meta, rng, torch):
    """Flip/rot90 applied identically to image, mask, and the semantic meta-channel."""
    if rng.random() < 0.5:
        img, m = torch.flip(img, dims=[3]), torch.flip(m, dims=[3])
        meta = torch.flip(meta, dims=[3]) if meta is not None else None
    if rng.random() < 0.5:
        img, m = torch.flip(img, dims=[2]), torch.flip(m, dims=[2])
        meta = torch.flip(meta, dims=[2]) if meta is not None else None
    k = int(rng.integers(0, 4))
    if k:
        img, m = torch.rot90(img, k, dims=[2, 3]), torch.rot90(m, k, dims=[2, 3])
        meta = torch.rot90(meta, k, dims=[2, 3]) if meta is not None else None
    return img, m, meta
