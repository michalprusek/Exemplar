"""Trainable-head backend. ``fit`` (re)trains :class:`DINOHead` on the labeled support each
round (Dice+BCE, augmentation via feature-grid flips/rot90, warm-start, fixed epoch
budget). ``foreground``/``predict`` run the trained head; ``last_logits`` is the Spec-B
acquisition hook (the head's dense logits are the coupling to its weights)."""
from __future__ import annotations

import numpy as np

from active_segmenter.config import ClusterConfig
from active_segmenter.propose import instances as inst
from active_segmenter.segment.base import LabeledExample, foreground_from_score
from active_segmenter.types import InstanceMask


def _dice_bce(logits, target):
    import torch.nn.functional as F

    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = logits.sigmoid()
    dice = 1 - (2 * (p * target).sum() + 1) / (p.sum() + target.sum() + 1)
    return bce + dice


def _grid_target(label_map, gh, gw):
    from skimage.transform import resize

    t = resize((np.asarray(label_map) > 0).astype(np.float32), (gh, gw), order=1,
               mode="edge", anti_aliasing=False)
    return (t > 0.5).astype(np.float32)


class TrainableHeadBackend:
    def __init__(self, device: str | None = None, in_dim: int = 1024, hidden: int = 256,
                 epochs: int = 60, lr: float = 1e-3, warm_start: bool = True,
                 cluster_cfg: ClusterConfig | None = None):
        self.device = device or "cpu"
        self.in_dim, self.hidden = in_dim, hidden
        self.epochs, self.lr, self.warm_start = epochs, lr, warm_start
        self.cc = cluster_cfg or ClusterConfig()
        self.head = None

    def _ensure_head(self, in_dim: int):
        from active_segmenter.segment.head import DINOHead

        if self.head is None or not self.warm_start or self.in_dim != in_dim:
            self.in_dim = in_dim
            self.head = DINOHead(in_dim, self.hidden, 1).to(self.device)
        return self.head

    def fit(self, support: list[LabeledExample]) -> None:
        import torch

        if not support:
            return
        # infer feature dim from the data so the head works for ANY encoder
        # (ViT-L=1024, ConvNeXt stages=192/384/768/1536) with no constructor change.
        in_dim = int(np.asarray(support[0].feat_grid).shape[-1])
        head = self._ensure_head(in_dim)
        opt = torch.optim.Adam(head.parameters(), lr=self.lr)
        # per-image tensors — NOT stacked, because native-resolution grids differ in size
        # per image (only square fixed-res grids could be batched).
        items = []
        for ex in support:
            fg = np.asarray(ex.feat_grid, np.float32)          # [G0, G1, D]
            gh, gw = fg.shape[:2]
            X = torch.from_numpy(fg.transpose(2, 0, 1))[None].to(self.device)      # [1, D, G0, G1]
            Y = torch.from_numpy(_grid_target(ex.label_map, gh, gw))[None, None].to(self.device)
            items.append((X, Y))
        head.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = 0.0
            for X, Y in items:
                # augmentation: random rot90 + horizontal flip over the spatial dims (X and Y
                # transform together so features stay aligned with their target)
                k = int(torch.randint(0, 4, (1,)))
                Xa, Ya = torch.rot90(X, k, (2, 3)), torch.rot90(Y, k, (2, 3))
                if torch.rand(1) < 0.5:
                    Xa, Ya = torch.flip(Xa, (3,)), torch.flip(Ya, (3,))
                loss = loss + _dice_bce(head(Xa), Ya)
            (loss / len(items)).backward()
            opt.step()
        head.eval()

    def score_map(self, image, feat_grid, class_id: int = 1) -> np.ndarray:
        import torch

        head = self.head
        if head is None:
            g = np.asarray(feat_grid).shape[0]
            return np.zeros((g, g), np.float32)
        fg = np.asarray(feat_grid, np.float32).transpose(2, 0, 1)[None]  # [1, D, G, G]
        with torch.no_grad():
            logits = head(torch.from_numpy(fg).to(self.device))[0, 0]
        return logits.cpu().numpy().astype(np.float32)  # >0 == fg (sigmoid > 0.5)

    def foreground(self, image, feat_grid, class_id: int = 1) -> np.ndarray:
        return foreground_from_score(self.score_map(image, feat_grid), np.asarray(image).shape[:2])

    def predict(self, image, feat_grid, class_id: int = 1) -> list[InstanceMask]:
        s = self.score_map(image, feat_grid, class_id)
        masks = inst.decompose(s, self.cc, class_id, feat_grid=None)
        return inst.upsample_masks(masks, np.asarray(image).shape[:2])

    def last_logits(self, feat_grid) -> np.ndarray:   # Spec-B gradient/uncertainty hook
        return self.score_map(None, feat_grid)

    def grad_embedding(self, feat_grid) -> np.ndarray:
        """BADGE-style weight-coupled acquisition signal: the loss-gradient w.r.t. the head's
        1x1 classifier weights, under the head's own PSEUDO-label, pooled over patches into a
        single ``[D_hidden+1]`` vector per image. ``||g||`` = expected-gradient-length (how much
        this image would move the head); its direction = which parameters it moves (BADGE
        clusters on that). Returns a zero vector if the head is untrained."""
        import torch

        head = self.head
        if head is None:
            return np.zeros(self.hidden + 1, np.float32)
        fg = np.asarray(feat_grid, np.float32).transpose(2, 0, 1)[None]
        with torch.no_grad():
            logits, pen = head.forward_with_penultimate(torch.from_numpy(fg).to(self.device))
            p = logits.sigmoid()                          # [1,1,G0,G1]
            y = (p > 0.5).float()                         # self pseudo-label
            resid = (p - y)[0, 0]                         # [G0,G1] dL/dlogit
            h = pen[0]                                    # [Dh,G0,G1]
            # per-patch grad wrt classifier weight = resid * h ; wrt bias = resid. Pool (mean).
            gw = (h * resid[None]).reshape(h.shape[0], -1).mean(1)   # [Dh]
            gb = resid.reshape(-1).mean().reshape(1)                 # [1]
            g = torch.cat([gw, gb])
        return g.cpu().numpy().astype(np.float32)
