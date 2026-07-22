"""Frozen DINOv3-L dense feature extractor.

Follows the spike's proven recipe (see docs/project-notes.md):
- grayscale -> 3-channel, per-image min-max normalisation, ImageNet mean/std;
- ``last_hidden_state`` is ``[1 + 4 + n_patches, D]`` -> drop the first 5 tokens
  (CLS + 4 register) before reshaping the patch tokens to the stride-16 grid;
- L2-normalise patch features so cosine similarity is a plain dot product.

The backbone is frozen (``eval()``, ``no_grad``). Weights never change — the
"learning" is the growing memory bank, not gradient steps.
"""
from __future__ import annotations

import numpy as np

from active_segmenter.config import EncoderConfig

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_rgb01(image: np.ndarray) -> np.ndarray:
    a = np.asarray(image).astype(np.float32)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.shape[2] == 4:
        a = a[..., :3]
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    a = (a - a.min()) / (np.ptp(a) + 1e-6)
    return a


class Dinov3Encoder:
    def __init__(self, cfg: EncoderConfig, device: str):
        import torch
        from transformers import AutoModel

        self.cfg = cfg
        self.device = device
        self._torch = torch
        self.model = AutoModel.from_pretrained(cfg.model_id).eval().to(device)
        self._mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
        self._std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1).to(device)

    def _preprocess(self, image: np.ndarray, resolution: int):
        torch = self._torch
        a = _to_rgb01(image)
        t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(self.device)
        t = torch.nn.functional.interpolate(
            t, size=(resolution, resolution), mode="bilinear", align_corners=False
        )
        return (t - self._mean) / self._std

    def _forward_grid(self, image: np.ndarray, resolution: int,
                      normalize: bool = True) -> np.ndarray:
        """One tile: returns ``[G, G, D]`` patch features (float32). By default L2-normalises them
        (the cosine=dot convention) and applies the gram / pos-bias refinements. ``normalize=False``
        returns the RAW backbone features (their per-patch norm carries salience) — the input a learned
        feature upsampler (AnyUp) was trained on, so the upsampler path must NOT pre-normalise.

        LAYER-FUSION (``cfg.layers`` non-empty, e.g. ``(-1, 12)``): CONCAT several DINOv3 blocks along the
        feature axis → ``[G, G, D·len(layers)]``. Each layer is L2-normalised SEPARATELY first (equal starting
        weight, so no layer dominates by raw magnitude), then the concat is re-normalised to unit; the light
        head then LEARNS the per-layer weighting from the K labels (self-configuring — FSSDINO shows no
        heuristic reliably picks the layer, so give the head both and let it decide). Empty ``cfg.layers`` =
        single ``cfg.layer`` (``-1`` = last), the original behaviour."""
        torch = self._torch
        g = resolution // self.cfg.patch_stride
        layers = tuple(getattr(self.cfg, "layers", ()) or ())
        with torch.no_grad():
            out = self.model(pixel_values=self._preprocess(image, resolution),
                             output_hidden_states=(bool(layers) or self.cfg.layer != -1))
            if layers:                                           # -1 → POST-norm last_hidden_state (baseline parity;
                parts = [(out.last_hidden_state if i == -1       # hidden_states[-1] is PRE final-LayerNorm ≠ baseline)
                          else out.hidden_states[i])[0][self.cfg.n_prefix_tokens:] for i in layers]
            else:
                hs = (out.hidden_states[self.cfg.layer] if self.cfg.layer != -1
                      else out.last_hidden_state)[0]
                parts = [hs[self.cfg.n_prefix_tokens:]]
        if not normalize:
            patch = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
            return patch.reshape(g, g, -1).float().cpu().numpy()      # RAW features (upsampler input)
        parts = [torch.nn.functional.normalize(p, dim=1) for p in parts]   # L2 per LAYER (equal weight)
        patch = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        if len(parts) > 1:
            patch = torch.nn.functional.normalize(patch, dim=1)      # unit whole (keeps cosine=dot convention)
        feat = patch.reshape(g, g, -1)
        if self.cfg.gram_refine:
            feat = _gram_refine(feat)
        if self.cfg.project_pos_bias:
            feat = _project_out_pos_bias(feat)
        return feat.float().cpu().numpy()

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Dense features ``[G, G, D]``. Densifies the coarse grid via a learned upsampler
        (``feat_upsampler``) OR shift-merge superres (``superres_factor``) — mutually exclusive —
        else tiles large images when ``cfg.tile``, else a single forward pass."""
        res = self.cfg.resolution
        up_name = getattr(self.cfg, "feat_upsampler", "none")
        if up_name not in (None, "none"):
            if getattr(self.cfg, "superres_factor", 1) > 1:
                raise ValueError("feat_upsampler and superres_factor are mutually exclusive "
                                 "coarse-grid densifiers — set exactly one")
            if getattr(self.cfg, "layers", ()):
                raise ValueError("feat_upsampler + layers (layer-fusion) unsupported — AnyUp expects "
                                 "single-layer raw features, not a multi-layer concat")
            from active_segmenter.encoder.feat_upsample import upsample_grid

            base = self._forward_grid(image, res, normalize=False)  # RAW features — AnyUp's expected input
            guide = self._preprocess(image, res)                    # [1, 3, res, res] ImageNet-normed guide
            return upsample_grid(up_name, base, guide,
                                 int(self.cfg.feat_upsample_factor), self.device)
        if getattr(self.cfg, "superres_factor", 1) > 1:
            from active_segmenter.encoder.superres import jbu_snap, shift_merge

            feat = shift_merge(self._forward_grid, image, res,
                               self.cfg.patch_stride, self.cfg.superres_factor)
            if getattr(self.cfg, "jbu", False):
                feat = jbu_snap(feat, image, self.cfg.jbu_sigma_spatial,
                                self.cfg.jbu_sigma_range)
            return feat
        h, w = np.asarray(image).shape[:2]
        if not self.cfg.tile or max(h, w) <= res:
            return self._forward_grid(image, res)
        return self._tiled_extract(image, res)

    def extract_batch(self, images, resolution: int | None = None) -> np.ndarray:
        """Dense features for a LIST of tiles in ONE model forward → ``[B, G, G, D]`` (float32). Used by
        the crop/fine branches to encode an image's tiles together instead of one sequential forward
        each — a big GPU-utilisation speedup. Same normalisation as :meth:`_forward_grid`.

        NB this is NATIVE-scale by design: it honours ``cfg.layer`` but NOT ``superres_factor``/``jbu``/
        ``feat_upsampler`` (unlike :meth:`extract`). That is intentional — the scale-fusion FINE branch is
        already the native-detail path (it tiles at ``res``), so densifying it too would over-densify (the
        negative ``superres_factor=4`` regime); ``superres`` is a COARSE-branch densification lever only.
        When a run sets ``superres_factor>1`` the coarse branch (``extract``) is super-resolved and the
        fine branch here stays native — the intended asymmetry, not a silent divergence."""
        torch = self._torch
        res = resolution or self.cfg.resolution
        g = res // self.cfg.patch_stride
        px = torch.cat([self._preprocess(im, res) for im in images], dim=0)   # [B,3,res,res]
        layers = tuple(getattr(self.cfg, "layers", ()) or ())
        with torch.no_grad():
            out = self.model(pixel_values=px, output_hidden_states=(bool(layers) or self.cfg.layer != -1))
            if layers:                                                        # -1 → post-norm last_hidden_state
                parts = [(out.last_hidden_state if i == -1
                          else out.hidden_states[i])[:, self.cfg.n_prefix_tokens:] for i in layers]
            else:
                hs = (out.hidden_states[self.cfg.layer] if self.cfg.layer != -1 else out.last_hidden_state)
                parts = [hs[:, self.cfg.n_prefix_tokens:]]
        parts = [torch.nn.functional.normalize(p, dim=2) for p in parts]      # L2 per layer (feature dim=2)
        patch = torch.cat(parts, dim=2) if len(parts) > 1 else parts[0]       # [B, G*G, D·L]
        if len(parts) > 1:
            patch = torch.nn.functional.normalize(patch, dim=2)               # unit whole (cosine convention)
        feats = patch.reshape(len(images), g, g, -1)
        if self.cfg.gram_refine or self.cfg.project_pos_bias:                 # per-tile ops
            return np.stack([self._forward_grid(im, res) for im in images])
        return feats.float().cpu().numpy()

    def extract_cls(self, image: np.ndarray) -> np.ndarray:
        """Normalised CLS token ``[D]`` — a global image descriptor for cold start."""
        torch = self._torch
        with torch.no_grad():
            hs = self.model(
                pixel_values=self._preprocess(image, self.cfg.resolution)
            ).last_hidden_state[0]
        cls = torch.nn.functional.normalize(hs[0], dim=0)
        return cls.float().cpu().numpy()

    def _tiled_extract(self, image: np.ndarray, res: int) -> np.ndarray:
        """Feature-tile a large image and feather-blend overlapping tiles.

        Instances are not stitched here (that is inference-level, deferred); this
        only builds a dense feature grid at native scale for the propose stage."""
        img = _to_rgb01(image)
        h, w = img.shape[:2]
        stride = int(res * (1 - self.cfg.tile_overlap))
        gs = self.cfg.patch_stride
        gh, gw = -(-h // gs), -(-w // gs)  # ceil: native-scale grid
        acc = None
        wsum = np.zeros((gh, gw, 1), np.float32)
        ys = list(range(0, max(1, h - res + 1), stride)) or [0]
        xs = list(range(0, max(1, w - res + 1), stride)) or [0]
        if ys[-1] != max(0, h - res):
            ys.append(max(0, h - res))
        if xs[-1] != max(0, w - res):
            xs.append(max(0, w - res))
        for y in ys:
            for x in xs:
                tile = img[y : y + res, x : x + res]
                th, tw = tile.shape[:2]
                pad = np.zeros((res, res, 3), np.float32)
                pad[:th, :tw] = tile
                fg = self._forward_grid(pad, res)  # [res/gs, res/gs, D]
                if acc is None:
                    acc = np.zeros((gh, gw, fg.shape[-1]), np.float32)
                gy, gx = y // gs, x // gs
                fh = min(fg.shape[0], gh - gy)
                fw = min(fg.shape[1], gw - gx)
                win = _feather(fh, fw)[..., None]
                acc[gy : gy + fh, gx : gx + fw] += fg[:fh, :fw] * win
                wsum[gy : gy + fh, gx : gx + fw] += win
        feat = acc / np.maximum(wsum, 1e-6)
        norm = np.linalg.norm(feat, axis=2, keepdims=True)
        return (feat / np.maximum(norm, 1e-6)).astype(np.float32)


def _feather(h: int, w: int) -> np.ndarray:
    """Triangular window so tile centres dominate over seams."""
    wy = 1 - np.abs(np.linspace(-1, 1, h))
    wx = 1 - np.abs(np.linspace(-1, 1, w))
    win = np.outer(wy, wx).astype(np.float32)
    return np.maximum(win, 0.05)


def _gram_refine(feat, temp: float = 0.1):
    """One step of Gram (patch self-similarity) propagation: aggregate each patch feature
    over all patches weighted by softmax cosine similarity, then renormalise. Sharpens dense
    correspondence by pulling a patch toward its semantic neighbours (FSSDINO-style Gram
    refinement). ``feat`` is a ``[G0, G1, D]`` torch tensor of unit patch vectors."""
    import torch

    g0, g1, d = feat.shape
    f = feat.reshape(-1, d)                                  # [N, D] unit
    attn = torch.softmax((f @ f.T) / temp, dim=1)            # [N, N] similarity weights
    refined = torch.nn.functional.normalize(attn @ f, dim=1)
    return refined.reshape(g0, g1, d)


def _project_out_pos_bias(feat):
    """Remove the dominant shared component (INSID3-style positional-bias
    suppression): subtract the mean patch direction and renormalise. ``feat`` is
    a ``[G, G, D]`` torch tensor."""
    import torch

    mean = feat.mean(dim=(0, 1), keepdim=True)
    return torch.nn.functional.normalize(feat - mean, dim=2)
