"""UniverSeg few-shot baseline (Butoi et al., ICCV'23) as a segmenter backend for the panel.

UniverSeg is a purpose-built cross-task FEW-SHOT medical-image segmenter: a CNN that, at inference, takes a
query image + a support set of (image, label) pairs and predicts the query mask — no fine-tuning. It is the
most directly comparable academic few-shot baseline to our method. Fixed 128×128 grayscale I/O (its training
resolution); we resize in/out. Semantic foreground only → instances via connected components (its honest
limitation on touching objects, same as any semantic-only method without a separation stage)."""
from __future__ import annotations

import numpy as np

from active_segmenter.types import InstanceMask


def _gray01(image) -> np.ndarray:
    a = np.asarray(image).astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    return (a - a.min()) / (np.ptp(a) + 1e-6)


class UniverSegBackend:
    """fit(support) stores the support set; foreground/predict run UniverSeg per query. Matches the
    SegmenterBackend interface (fit / foreground / predict) so it drops into panel_benchmark."""

    # The frozen net derives nothing from the support draw: `fit` REASSIGNS `_sup` wholesale, so no
    # part of a previous draw can reach the next one and there is no reset hook to expose.
    stateless_support = True

    def __init__(self, device=None, res: int = 128, max_support: int = 32):
        self.device = device or "cpu"
        self.res = res
        self.max_support = max_support          # UniverSeg support-size cap (memory/training regime)
        self._model = None
        self._sup = None                        # (support_images, support_labels) tensors on device

    def _net(self):
        if self._model is None:
            from universeg import universeg
            self._model = universeg(pretrained=True).to(self.device).eval()
        return self._model

    def _img128(self, image) -> np.ndarray:
        from skimage.transform import resize
        return resize(_gray01(image), (self.res, self.res), order=1,
                      mode="edge", anti_aliasing=True).astype(np.float32)

    def _lab128(self, label) -> np.ndarray:
        from skimage.transform import resize
        m = (np.asarray(label) > 0).astype(np.float32)
        return (resize(m, (self.res, self.res), order=0, mode="edge") > 0.5).astype(np.float32)

    def fit(self, support) -> None:
        import torch
        sup = support[: self.max_support]
        imgs = np.stack([self._img128(ex.image) for ex in sup])           # [S,128,128]
        labs = np.stack([self._lab128(ex.label_map) for ex in sup])
        self._sup = (torch.from_numpy(imgs)[None, :, None].to(self.device),   # [1,S,1,128,128]
                     torch.from_numpy(labs)[None, :, None].to(self.device))

    def _prob(self, image) -> np.ndarray:
        import torch
        from skimage.transform import resize
        q = torch.from_numpy(self._img128(image))[None, None].to(self.device)  # [1,1,128,128]
        with torch.no_grad():
            logit = self._net()(q, self._sup[0], self._sup[1])                 # [1,1,128,128]
        prob = torch.sigmoid(logit)[0, 0].float().cpu().numpy()
        return resize(prob, np.asarray(image).shape[:2], order=1, mode="edge",
                      anti_aliasing=True).astype(np.float32)

    def foreground(self, image, feat_grid, class_id: int = 1) -> np.ndarray:
        if self._sup is None:
            return np.zeros(np.asarray(image).shape[:2], bool)
        return self._prob(image) > 0.5

    def predict(self, image, feat_grid, class_id: int = 1) -> list[InstanceMask]:
        from skimage.measure import label
        fg = self.foreground(image, feat_grid, class_id)
        lab = label(fg)                                                       # semantic → CC instances
        out = []
        for i in range(1, int(lab.max()) + 1):
            mm = lab == i
            if mm.any():
                out.append(InstanceMask(mask=mm, points=None, class_id=class_id,
                                        instance_id=i, score=1.0))
        return out
