"""OpenPhenom (Recursion CA-MAE, ViT-S/16) as a frozen dense-feature extractor, duck-typed like
``Dinov3Encoder``. OpenPhenom is THE general microscopy foundation model (masked-autoencoder pretrained on
high-content fluorescence screens, RxRx3 + JUMP). It exposes only a pooled embedding via ``predict``; we reach
the patch tokens the same way ``predict`` does — ``model.encoder.vit_backbone.forward_features`` (returns
``[N, 1+P, D]``) — drop the CLS token, and reshape to a ``[G, G, D]`` grid. Channel-agnostic → we feed the
support-chosen single channel. Native ViT-S/16 grid is coarse (~16x16), so it stresses the classical/fine
branches; that is an honest limitation of this backbone, reported as such."""
from __future__ import annotations

import numpy as np

_OPENPHENOM_IDS = ("recursionpharma/OpenPhenom",)


def is_openphenom(model_id: str) -> bool:
    mid = model_id[len("openphenom:"):] if model_id.startswith("openphenom:") else model_id
    return mid in _OPENPHENOM_IDS or model_id.startswith("openphenom:")


class OpenPhenomEncoder:
    """Frozen OpenPhenom CA-MAE dense feature extractor (ViT-S/16, 384-d)."""

    def __init__(self, cfg, device: str):
        import torch
        from transformers import AutoModel

        self.cfg = cfg
        self.device = device
        self._torch = torch
        self.model = AutoModel.from_pretrained("recursionpharma/OpenPhenom", trust_remote_code=True).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.patch = 16
        self.res = 256                                    # OpenPhenom native (its pos-embed grid); fixed
        self.feat_dim = 384
        cfg.patch_stride = self.patch                     # backend/superres read this
        self.grid = self.res // self.patch                # 16

    def _prep(self, image):
        """support-chosen single channel → [1,1,res,res] float on device (min-max to [0,1])."""
        import torch
        from skimage.transform import resize
        a = np.asarray(image, np.float32)
        g = a if a.ndim == 2 else a[..., :3].mean(-1)     # grayscale (channel choice handled upstream by the head)
        g = (g - g.min()) / (np.ptp(g) + 1e-6)
        g = resize(g, (self.res, self.res), order=1, mode="edge", anti_aliasing=True).astype(np.float32)
        return torch.from_numpy(g)[None, None].to(self.device)

    def _tokens(self, t):
        """[1,1,res,res] → [G,G,D] patch grid (L2-normalised), via the same path predict() uses."""
        import torch
        with torch.no_grad():
            x = self.model.input_norm(t)
            X = self.model.encoder.vit_backbone.forward_features(x)   # [1, 1+P, D]
        X = X[0, 1:, :]                                    # drop CLS -> [P, D]
        g = self.grid
        X = X[: g * g].reshape(g, g, -1)
        X = torch.nn.functional.normalize(X, dim=-1)
        return X.float().cpu().numpy()

    def extract(self, image, res=None) -> np.ndarray:      # res ignored: OpenPhenom's grid is fixed at 256/16
        return self._tokens(self._prep(image))

    def extract_batch(self, images, res=None):
        return np.stack([self.extract(im) for im in images])

    def extract_cls(self, image) -> np.ndarray:
        import torch
        with torch.no_grad():
            emb = self.model.predict(self._prep(image))     # [1, 384] pooled
        v = emb[0].float().cpu().numpy()
        return v / (np.linalg.norm(v) + 1e-6)
