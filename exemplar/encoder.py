"""The frozen DINOv3 ViT-L/16 backbone and the coarse view.

The backbone is never trained. It is read twice: here, as the *coarse view* (the whole image resized to 672 x 672 and
encoded once, a 42 x 42 grid of semantic context over the entire field), and in `fineview.py`, as the magnified *fine
view*. Both readings take the last hidden state (after the final LayerNorm) and the output of block 12 of 24, drop the
5 prefix tokens (CLS + 4 registers), L2-normalise each block over the feature axis, concatenate them and normalise the
2048-vector again.

Only the transformer runs under bfloat16 autocast; the pixel preprocessing and the normalisations stay in float32 and
the transformer's outputs are cast back to float32 before any of them (`install_bf16_forward`).

The coarse view is cached on disk, keyed by a hash of the image bytes and the encoder configuration. The key and the
directory naming (`<cache>_bf16`) are those of the AutoSeg benchmark, so its caches can be reused as they are.

Source: active_segmenter/encoder/{dinov3,cache,cached,factory}.py and scripts/closed_form_bench.py:327-346 (AutoSeg).
Line references point to the authors' research code, which is not public.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from typing import Callable

import numpy as np

MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
RESOLUTION = 672          # encoder input side; a multiple of the patch, 42 x 42 patch grid
PATCH = 16                # patch stride of DINOv3 ViT-L/16
N_PREFIX = 5              # CLS + 4 register tokens, dropped
LAYERS = (-1, 12)         # the last hidden state (post-norm) and the output of block 12
LAST_ONLY = (-1,)         # ablation: the last hidden state alone (AutoSeg: EncoderConfig(layers=()), no `_lf12` token)
CACHE_TAG = "dinov3-vitl16-pretrain-lvd1689m-res672-Lf-1_12"   # factory.cache_tag(EncoderConfig(layers=(-1, 12)))


def check_layers(layers) -> tuple:
    """The blocks read: (-1,) (the last alone) or (-1, n), n an intermediate block 1..23 (the `_lf<n>` token,
    al_testbed.encoder_layers_for_method)."""
    layers = tuple(int(x) for x in layers)
    if layers == LAST_ONLY or (len(layers) == 2 and layers[0] == -1 and 1 <= layers[1] <= 23):
        return layers
    raise ValueError(f"layers must be (-1,) or (-1, n) with 1 <= n <= 23, got {layers}")


def cache_tag(layers=LAYERS) -> str:
    """The coarse-view cache key suffix of AutoSeg's factory.cache_tag for these layers: the `Lf` part names a fused
    set, the last block alone has none. (factory.py:57-90)"""
    layers = check_layers(layers)
    base = "dinov3-vitl16-pretrain-lvd1689m-res672"
    return base if layers == LAST_ONLY else base + "-Lf" + "_".join(str(x) for x in layers)


def hidden_states_needed(layers) -> bool:
    """Whether the backbone must return its hidden states: only for an intermediate block. (dinov3.py:72-73,
    closed_form_bench.py:427, `bool(layers) or cfg.layer != -1` with layers=() for the last block alone)"""
    return check_layers(layers) != LAST_ONLY


def fuse_blocks(parts, dim):
    """L2-normalise each block over the feature axis; with two blocks concatenate them and normalise again, with one
    keep it as it is (a second normalisation would not be bit-stable). (dinov3.py:83-86, closed_form_bench.py:432-434)"""
    import torch
    parts = [torch.nn.functional.normalize(p, dim=dim) for p in parts]
    if len(parts) == 1:
        return parts[0]
    return torch.nn.functional.normalize(torch.cat(parts, dim=dim), dim=dim)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_rgb01(image: np.ndarray) -> np.ndarray:
    """Grey -> 3 channels, alpha dropped, whole-image min-max to [0, 1]. (dinov3.py:22-31)"""
    a = np.asarray(image).astype(np.float32)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.shape[2] == 4:
        a = a[..., :3]
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    a = (a - a.min()) / (np.ptp(a) + 1e-6)
    return a


def install_bf16_forward(model, device_type: str) -> None:
    """Run only the transformer under bfloat16 autocast and cast its outputs back to float32.

    The wrapper replaces `model.forward`, so both the coarse view and the fine view (which call the model directly)
    go through it, and the classical priors never do. (closed_form_bench.py:333-346)"""
    import torch

    _orig_forward = model.forward

    def _amp_forward(*args, **kw):
        with torch.autocast(device_type, dtype=torch.bfloat16):
            out = _orig_forward(*args, **kw)
        out["last_hidden_state"] = out.last_hidden_state.float()
        if getattr(out, "hidden_states", None) is not None:
            out["hidden_states"] = tuple(h.float() for h in out.hidden_states)
        return out
    model.forward = _amp_forward


class Dinov3Encoder:
    """The frozen DINOv3 ViT-L/16 (`facebook/dinov3-vitl16-pretrain-lvd1689m`, a gated Hugging Face checkpoint)."""

    def __init__(self, device: str, bf16: bool = True, model_id: str = MODEL_ID, layers=LAYERS):
        import torch
        from transformers import AutoModel

        self.device = device
        self.layers = check_layers(layers)
        self._torch = torch
        self.model = AutoModel.from_pretrained(model_id).eval().to(device)
        self._mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
        self._std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1).to(device)
        if bf16:
            install_bf16_forward(self.model, torch.device(device).type)

    def _preprocess(self, image: np.ndarray, resolution: int):
        """Whole-image min-max, bilinear resize to resolution x resolution (not aspect-preserving), ImageNet norm.
        (dinov3.py:46-53)"""
        torch = self._torch
        a = _to_rgb01(image)
        t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(self.device)
        t = torch.nn.functional.interpolate(
            t, size=(resolution, resolution), mode="bilinear", align_corners=False
        )
        return (t - self._mean) / self._std

    def coarse(self, image: np.ndarray) -> np.ndarray:
        """The coarse view: [42, 42, 2048] float32, unit vectors ([42, 42, 1024] with layers=LAST_ONLY).
        (dinov3.py:55-93, `_forward_grid` with layers=(-1, 12) or (), normalize=True, no Gram refinement or
        positional-bias projection)"""
        torch = self._torch
        resolution = RESOLUTION
        g = resolution // PATCH
        layers = getattr(self, "layers", LAYERS)
        with torch.no_grad():
            out = self.model(pixel_values=self._preprocess(image, resolution),
                             output_hidden_states=hidden_states_needed(layers))
            parts = [(out.last_hidden_state if i == -1         # -1 -> the POST-norm last_hidden_state
                      else out.hidden_states[i])[0][N_PREFIX:] for i in layers]
        patch = fuse_blocks(parts, 1)                       # L2 per block (equal weight), the unit 2048-vector
        feat = patch.reshape(g, g, -1)
        return feat.float().cpu().numpy()


class EmbeddingCache:
    """Disk cache of arrays keyed by sha1(image bytes + shape + dtype + tag), one `.npy` per entry, written
    atomically. (encoder/cache.py:18-55)"""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _key(self, image: np.ndarray, extra: str) -> str:
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(image).tobytes())
        h.update(str(image.shape).encode())
        h.update(str(image.dtype).encode())
        h.update(extra.encode())
        return h.hexdigest()

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, key + ".npy")

    def get_or_compute(self, image: np.ndarray, extra: str, fn: Callable[[], np.ndarray]) -> np.ndarray:
        path = self._path(self._key(image, extra))
        if os.path.exists(path):
            return np.load(path)
        arr = np.asarray(fn(), dtype=np.float32)
        # a temp file unique per writer, published by an atomic rename: a reader never sees a half-written entry
        fd, tmp = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                np.save(fh, arr)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return arr


def coarse_cache_dir(cache: str, bf16: bool = True) -> str:
    """The cache directory the coarse view uses: bfloat16 features never share a directory with float32 ones.
    (closed_form_bench.py:328-329)"""
    if bf16 and not cache.rstrip("/").endswith("_bf16"):
        return cache.rstrip("/") + "_bf16"
    return cache


class CoarseView:
    """The coarse view of an image, through the on-disk cache when a cache directory is given (keyed by the encoder's
    layers, so the two readings never share an entry)."""

    def __init__(self, encoder: Dinov3Encoder, cache_dir: str | None = None):
        self.encoder = encoder
        self.cache = EmbeddingCache(cache_dir) if cache_dir else None
        self.tag = cache_tag(getattr(encoder, "layers", LAYERS))

    def __call__(self, image) -> np.ndarray:
        if self.cache is None:
            return np.asarray(self.encoder.coarse(image), dtype=np.float32)
        return self.cache.get_or_compute(image, self.tag, lambda: self.encoder.coarse(image))
