"""Frozen HISTOPATHOLOGY foundation backbone — the H&E-domain alternative to DINOv3.

Purpose: the decisive "is the MoNuSeg wall a property of a GENERAL frozen representation,
or would an H&E-pretrained backbone close it?" experiment. Everything downstream (the light
fusion head, self-config, affinity decoder, K/pool/test/seeds/res protocol) is kept BYTE-for-byte
identical — the ONLY variable is the frozen backbone. See ``docs/ISBI-NATMETHODS-POSITIONING.md``.

The head auto-adapts to the backbone feature dim (``HeadFusionBackend.fit`` derives ``in_dim`` from
``feat_grid.shape[-1]`` and FiLM's hypernet reads ``2*in_dim``), so a 1024-d (UNI ViT-L), 1280-d
(Virchow2 ViT-H) or 1536-d (H-optimus-0 ViT-g / UNI2-h ViT-H) backbone all drop in with NO head edit.

These backbones ship as ``timm`` ViTs on the HF hub (NOT ``transformers.AutoModel``), so this is a
separate encoder class rather than a new ``model_id`` for :class:`Dinov3Encoder`. It is duck-typed
identically (``extract`` -> ``[G, G, D]`` L2-normalised, ``extract_batch`` -> ``[B, G, G, D]``,
``extract_cls`` -> ``[D]``), so it slots into the same cache + backends unchanged.

Key contract details honoured so the swap is fair:
- prefix/register tokens differ per model (UNI: 1 CLS + 0 reg; Virchow2/H-optimus-0: 1 CLS + 4 reg;
  UNI2-h: 1 CLS + 8 reg). We use timm's ``forward_intermediates`` / ``get_intermediate_layers`` with
  ``return_prefix_tokens=False`` so the prefix count is handled by timm from ``model.num_prefix_tokens``
  — no hard-coded ``n_prefix``.
- patch stride differs (UNI: 16; Virchow2/H-optimus-0/UNI2-h: 14). We read it from
  ``model.patch_embed.patch_size`` and write it back into ``cfg.patch_stride`` so the backend's
  fine-branch grid map (head_fusion_backend.py:1233) and shift-merge superres compute the right grid.
  ``resolution=672`` is a clean multiple of BOTH 16 (grid 42) and 14 (grid 48).
- each model's OWN channel normalisation (mean/std) is used (via ``timm.data.resolve_model_data_config``)
  — using ImageNet stats on an H&E model would unfairly handicap it. The geometric preprocessing
  (grayscale->RGB, global min-max to [0,1], bilinear resize to ``cfg.resolution``) is kept identical to
  :class:`Dinov3Encoder` so ONLY the channel stats + weights change.
"""
from __future__ import annotations

import numpy as np

from active_segmenter.config import EncoderConfig
from active_segmenter.encoder.dinov3 import _IMAGENET_MEAN, _IMAGENET_STD, _to_rgb01

# Known H&E foundation backbones. Keyed on the BARE hub id (an optional "hf-hub:" prefix is stripped
# before lookup). ``kwargs`` are passed to ``timm.create_model``; ``swiglu`` toggles the SwiGLU MLP that
# Virchow/UNI2 need (the mlp_layer/act_layer objects are attached at build time — they need torch imported).
_HISTO_SPECS: dict[str, dict] = {
    # ViT-L/16, 1024-d, 0 register tokens. GATED (request access + HF token). The CLEAN control:
    # matched capacity/patch/dim to DINOv3-ViT-L/16 -> isolates exactly ONE variable (SSL domain).
    "MahmoodLab/UNI": dict(kwargs=dict(init_values=1e-5, dynamic_img_size=True), swiglu=False),
    # ViT-H/14, 1536-d, 8 register tokens, SwiGLU. GATED. Bigger than DINOv3-L (capacity confound).
    "MahmoodLab/UNI2-h": dict(
        kwargs=dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
                    embed_dim=1536, mlp_ratio=2 * 2.66667, num_classes=0, no_embed_class=True,
                    reg_tokens=8, dynamic_img_size=True),
        swiglu=True),
    # ViT-H/14, 1280-d, 4 register tokens, SwiGLU. GATED.
    "paige-ai/Virchow2": dict(kwargs=dict(dynamic_img_size=True), swiglu=True),
    # ViT-g/14, 1536-d, 4 register tokens. Apache-2.0, UNGATED (no approval wait) -> the immediate probe.
    # ViT-g + patch-14 vs DINOv3-ViT-L/16 -> a WIN is confounded by model size; still answers
    # "does ANY histo backbone help". Strongest current H&E model.
    "bioptimus/H-optimus-0": dict(kwargs=dict(init_values=1e-5, dynamic_img_size=True), swiglu=False),
}


def is_histo_model(model_id: str) -> bool:
    """True if ``model_id`` names an H&E timm backbone (known id, ``hf-hub:``/``histo:`` prefix)."""
    mid = model_id
    for p in ("histo:", "hf-hub:"):
        if mid.startswith(p):
            return True
    return _bare_id(mid) in _HISTO_SPECS


def _bare_id(model_id: str) -> str:
    for p in ("histo:", "hf-hub:"):
        if model_id.startswith(p):
            model_id = model_id[len(p):]
    return model_id


class HistoEncoder:
    """Frozen H&E ViT dense feature extractor, duck-typed like :class:`Dinov3Encoder`."""

    def __init__(self, cfg: EncoderConfig, device: str):
        import timm
        import torch

        self.cfg = cfg
        self.device = device
        self._torch = torch
        bare = _bare_id(cfg.model_id)
        spec = _HISTO_SPECS.get(bare, dict(kwargs=dict(dynamic_img_size=True), swiglu=False))
        create_kwargs = dict(spec["kwargs"])
        if spec.get("swiglu"):
            from timm.layers import SwiGLUPacked

            create_kwargs.update(mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        create_kwargs.setdefault("num_classes", 0)   # some specs (UNI2-h) already set it -> avoid a dup kwarg
        # timm needs the "hf-hub:" scheme to pull weights from the hub.
        self.model = timm.create_model(f"hf-hub:{bare}", pretrained=True,
                                       **create_kwargs).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Derive the real patch stride / feature dim / prefix count FROM the loaded model and write the
        # stride back into cfg (the backend + superres read cfg.patch_stride). feat_dim is informational.
        ps = self.model.patch_embed.patch_size
        self.patch_stride = int(ps[0] if isinstance(ps, (tuple, list)) else ps)
        cfg.patch_stride = self.patch_stride
        cfg.feat_dim = int(getattr(self.model, "embed_dim", getattr(self.model, "num_features", 0)))
        cfg.n_prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 1))
        if cfg.resolution % self.patch_stride != 0:
            raise ValueError(f"resolution {cfg.resolution} not divisible by patch stride "
                             f"{self.patch_stride} for {bare} — pick a multiple (672 works for 14 and 16)")

        # Each model's OWN normalisation (fair to the H&E backbone); geometric preprocessing stays as
        # DINOv3's (global min-max to [0,1] via _to_rgb01), so ONLY channel stats + weights change.
        try:
            dc = timm.data.resolve_model_data_config(self.model)
            mean, std = tuple(dc["mean"]), tuple(dc["std"])
        except Exception:
            mean, std = _IMAGENET_MEAN, _IMAGENET_STD
        self._mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
        self._std = torch.tensor(std).view(1, 3, 1, 1).to(device)
        self._amp_dtype = torch.bfloat16 if device.startswith("cuda") else None

    # -- preprocessing (parity with Dinov3Encoder except the model-specific mean/std) --------------
    def _preprocess(self, image: np.ndarray, resolution: int):
        torch = self._torch
        a = _to_rgb01(image)
        t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(self.device)
        t = torch.nn.functional.interpolate(
            t, size=(resolution, resolution), mode="bilinear", align_corners=False)
        return (t - self._mean) / self._std

    def _autocast(self):
        torch = self._torch
        if self._amp_dtype is not None:
            return torch.autocast(device_type="cuda", dtype=self._amp_dtype)
        return _NullCtx()

    def _feature_map(self, px):
        """px: [B, 3, R, R] -> [B, C, g, g] patch feature map (prefix tokens dropped by timm)."""
        torch = self._torch
        with torch.no_grad(), self._autocast():
            feats = self.model.get_intermediate_layers(
                px, n=1, reshape=True, return_prefix_tokens=False, norm=True)[0]
        return feats.float()  # [B, C, g, g]

    def _forward_grid(self, image: np.ndarray, resolution: int) -> np.ndarray:
        """One tile -> ``[g, g, C]`` L2-normalised (cosine=dot convention, matching DINOv3)."""
        torch = self._torch
        fmap = self._feature_map(self._preprocess(image, resolution))[0]     # [C, g, g]
        fmap = torch.nn.functional.normalize(fmap, dim=0)                    # L2 over channels
        return fmap.permute(1, 2, 0).cpu().numpy()                          # [g, g, C]

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Dense ``[G, G, D]`` features. Honours ``superres_factor`` via the shared shift-merge util
        (parity with the DINOv3 coarse-branch densifier); else a single forward. ``tile`` for oversized
        images is left to the fine branch / native ConvNeXt path, as in best_v2."""
        res = self.cfg.resolution
        if getattr(self.cfg, "superres_factor", 1) > 1:
            from active_segmenter.encoder.superres import jbu_snap, shift_merge

            feat = shift_merge(self._forward_grid, image, res, self.patch_stride,
                               self.cfg.superres_factor)
            if getattr(self.cfg, "jbu", False):
                feat = jbu_snap(feat, image, self.cfg.jbu_sigma_spatial, self.cfg.jbu_sigma_range)
            return feat
        return self._forward_grid(image, res)

    def extract_batch(self, images, resolution: int | None = None) -> np.ndarray:
        """Dense features for a LIST of tiles in ONE forward -> ``[B, G, G, D]`` (native scale, used by
        the scale-fusion FINE branch). Same normalisation as :meth:`_forward_grid`."""
        torch = self._torch
        res = resolution or self.cfg.resolution
        px = torch.cat([self._preprocess(im, res) for im in images], dim=0)  # [B, 3, R, R]
        fmap = self._feature_map(px)                                         # [B, C, g, g]
        fmap = torch.nn.functional.normalize(fmap, dim=1)                    # L2 over channels
        return fmap.permute(0, 2, 3, 1).cpu().numpy()                       # [B, g, g, C]

    def extract_cls(self, image: np.ndarray) -> np.ndarray:
        """Normalised CLS token ``[D]`` — global descriptor for cold start."""
        torch = self._torch
        with torch.no_grad(), self._autocast():
            tokens = self.model.forward_features(self._preprocess(image, self.cfg.resolution))
        cls = tokens[0, 0].float()                                          # CLS is token 0
        return torch.nn.functional.normalize(cls, dim=0).cpu().numpy()


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
