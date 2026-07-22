"""Encoder factory — pick ViT vs ConvNeXt DINOv3 backbone from the config.

Selects :class:`Dinov3ConvNeXtEncoder` when the model is a ConvNeXt variant (either
``backbone == "convnext"`` or ``"convnext"`` in the model id), else the ViT
:class:`Dinov3Encoder`. Both are duck-typed (``extract`` / ``extract_cls``)."""
from __future__ import annotations

from active_segmenter.config import EncoderConfig


def is_convnext(cfg: EncoderConfig) -> bool:
    if cfg.backbone == "convnext":
        return True
    if cfg.backbone == "vit":
        return False
    return "convnext" in cfg.model_id.lower()


def make_encoder(cfg: EncoderConfig, device: str):
    # HISTOPATHOLOGY foundation backbone (UNI / Virchow2 / H-optimus-0 / UNI2-h): a timm ViT on the HF
    # hub, NOT a transformers.AutoModel — routed by model_id (known id, or an "hf-hub:"/"histo:" prefix)
    # or backbone == "histo". Duck-typed like Dinov3Encoder. Lazy import so timm is only required when used.
    from active_segmenter.encoder.histo import is_histo_model
    from active_segmenter.encoder.openphenom import is_openphenom

    if is_openphenom(cfg.model_id):        # OpenPhenom CA-MAE microscopy foundation model (custom AutoModel)
        from active_segmenter.encoder.openphenom import OpenPhenomEncoder

        return OpenPhenomEncoder(cfg, device)
    if cfg.backbone == "histo" or is_histo_model(cfg.model_id):
        from active_segmenter.encoder.histo import HistoEncoder

        return HistoEncoder(cfg, device)
    if is_convnext(cfg):
        from active_segmenter.encoder.convnext import Dinov3ConvNeXtEncoder

        return Dinov3ConvNeXtEncoder(cfg, device)
    if cfg.resolution <= 0:
        raise ValueError("native resolution (resolution<=0) needs a convnext backbone; the "
                         "ViT patch grid requires a fixed square input")
    from active_segmenter.encoder.dinov3 import Dinov3Encoder

    return Dinov3Encoder(cfg, device)


def cache_tag(cfg: EncoderConfig) -> str:
    """Cache-key suffix distinguishing backbone / stage / resolution so ViT-672,
    ConvNeXt-stage2-1024 and ConvNeXt-native features never collide on disk."""
    res = "NAT" if cfg.resolution <= 0 else str(cfg.resolution)
    base = f"{cfg.model_id.split('/')[-1]}-res{res}"
    if is_convnext(cfg):
        return f"{base}-cnxs{cfg.convnext_stage}"
    # EVERY feature-affecting knob must be in the key, else different layer/gram/tile settings
    # silently collide on disk (they did — an invalid HRF layer sweep).
    parts = [base]
    if getattr(cfg, "tile", False):
        parts.append("tiled")
    if getattr(cfg, "layer", -1) != -1:
        parts.append(f"L{cfg.layer}")
    if getattr(cfg, "layers", ()):                  # layer-fusion set (feature-affecting → must be in the key)
        parts.append("Lf" + "_".join(str(x) for x in cfg.layers))
    if getattr(cfg, "gram_refine", False):
        parts.append("gram")
    if getattr(cfg, "project_pos_bias", False):     # feature-affecting; was missing → latent collision
        parts.append("ppb")
    if getattr(cfg, "superres_factor", 1) > 1:
        parts.append(f"sr{cfg.superres_factor}")
    if getattr(cfg, "jbu", False):
        parts.append("jbu")
    up = getattr(cfg, "feat_upsampler", "none")
    if up not in (None, "none"):
        # ...v1 = the feat_upsample.py code-path version (raw-input + guards) AND the pinned _ANYUP_REF;
        # bump BOTH together on an intentional upgrade so old cached features never masquerade as new.
        parts.append(f"{up}{getattr(cfg, 'feat_upsample_factor', 2)}v1")
    return "-".join(parts)
