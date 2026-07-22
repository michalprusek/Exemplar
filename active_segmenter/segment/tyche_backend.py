"""Tyche few-shot in-context baseline (Rakic et al., CVPR'24, MIT CSAIL) as a segmenter backend.

Tyche is a purpose-built cross-task FEW-SHOT medical-image segmenter from the same lab / same family as
UniverSeg: at inference it takes a query image + a context set of (image, label) support pairs and predicts
the query mask with NO retraining — the exact paradigm as our method. It differs from UniverSeg in that it is
STOCHASTIC: it emits a *set* of plausible candidate masks (diversity injected via per-candidate noise), rather
than one deterministic mask. Fixed 128×128 grayscale I/O (its training resolution); we resize in/out — the
same protocol as our UniverSeg backend. Semantic foreground only → instances via connected components (its
honest limitation on touching objects, same as any semantic-only method without a separation stage).

FAIRNESS: to summarize the stochastic set as a single mask WITHOUT any oracle/GT, we take the MEAN over the
``n_pred`` candidate probability maps (the marginal / expected prediction) and threshold at 0.5. No GT is used
to pick a "best" candidate (that would be an oracle upper bound, unfair). The mean-of-set is deterministic
given the harness seed and is low-variance (n_pred candidates averaged). Tyche's own default inference is used.

The Tyche package is not pip-published; point ``TYCHE_SRC`` at the cloned repo (default: the tulen checkout)."""
from __future__ import annotations

import os
import sys

import numpy as np

from active_segmenter.types import InstanceMask

TYCHE_SRC = os.environ.get("TYCHE_SRC", "/disk1/prusek/tyche_src")


def _gray01(image) -> np.ndarray:
    a = np.asarray(image).astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    return (a - a.min()) / (np.ptp(a) + 1e-6)


class TycheBackend:
    """fit(support) stores the context set; foreground/predict run Tyche per query. Matches the
    SegmenterBackend interface (fit / foreground / predict) so it drops into the panel harness exactly
    like UniverSegBackend. n_pred = size of the stochastic candidate set that is averaged into the mask."""

    # As UniverSeg: `fit` REASSIGNS `_sup` wholesale and the model is frozen, so a support draw leaves
    # nothing behind. `_model` is built once on purpose (uniform noise across seeds) and is not draw state.
    stateless_support = True

    def __init__(self, device=None, res: int = 128, max_support: int = 32, n_pred: int = 16):
        self.device = device or "cpu"
        self.res = res
        self.max_support = max_support          # context-size cap (memory/training regime; native ≈16)
        self.n_pred = n_pred                     # stochastic candidate set size (averaged → single mask)
        self._sup = None                         # (support_images, support_labels) tensors on device
        self._model = self._build()              # built once (before the harness seed loop) → uniform noise

    def _build(self):
        if TYCHE_SRC not in sys.path:
            sys.path.insert(0, TYCHE_SRC)
        import torch
        from tyche import tychets
        return tychets(version="v1", pretrained=True).to(self.device).eval()

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
        self._sup = (torch.from_numpy(imgs)[:, None].to(self.device),     # [S,1,128,128]
                     torch.from_numpy(labs)[:, None].to(self.device))

    def _prob(self, image) -> np.ndarray:
        import torch
        from skimage.transform import resize
        q = torch.from_numpy(self._img128(image))[None, None].to(self.device)   # [1,1,128,128]
        with torch.no_grad():
            # pred_ged_stats returns [1, n_pred, H, W] sigmoid probs — the stochastic candidate set.
            yhat = self._model.pred_ged_stats(
                {"x": q, "sx": self._sup[0], "sy": self._sup[1], "target_size": self.n_pred},
                sigmoid=True)
        prob = yhat[0].mean(0).float().cpu().numpy()                            # MEAN over the set (no oracle)
        return resize(prob, np.asarray(image).shape[:2], order=1, mode="edge",
                      anti_aliasing=True).astype(np.float32)

    def foreground(self, image, feat_grid, class_id: int = 1) -> np.ndarray:
        if self._sup is None:
            return np.zeros(np.asarray(image).shape[:2], bool)
        return self._prob(image) > 0.5

    def predict(self, image, feat_grid, class_id: int = 1) -> list[InstanceMask]:
        from skimage.measure import label
        fg = self.foreground(image, feat_grid, class_id)
        lab = label(fg)                                                          # semantic → CC instances
        out = []
        for i in range(1, int(lab.max()) + 1):
            mm = lab == i
            if mm.any():
                out.append(InstanceMask(mask=mm, points=None, class_id=class_id,
                                        instance_id=i, score=1.0))
        return out
