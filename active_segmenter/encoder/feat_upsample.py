"""Learned frozen-feature upsampler for the DINOv3 coarse grid.

The LABEL-FREE alternative to shift-merge super-resolution (``encoder/superres.py``). A pretrained,
frozen-backbone upsampler lifts the coarse ``[G, G, D]`` /16 patch grid to ``[factor·G, factor·G, D]``
in ONE feed-forward pass, guided by the RGB image:

- ``anyup`` — AnyUp (Wimmer et al., ICLR'26, ``torch.hub`` ``wimmerth/anyup``): a *feature-agnostic*
  window-attention upsampler trained once on DINOv2, shown to transfer to DINOv3 unchanged. Runs with
  ``use_natten=False`` (pure-PyTorch attention — no CUDA-extension build, unlike BRIXEL).
- ``jafar`` — JAFAR (Couairon et al., NeurIPS'25): ships per-backbone DINOv3 S/S+/B/L checkpoints.
  Guarded until its hub API is verified on the GPU box (we don't ship an unverified call).

Why this over the head-level :class:`GuidedUp`/JBU upsamplers in ``segment/upsamplers.py``: those lift
the *projected 32-d* embedding and are trained on the K≈8 labels (data-starved risk); these lift the
*full 1024-d* grid with a *label-free pretrained* upsampler — the axis the A/B isolates.

CAVEAT (carry into the paper): a learned upsampler RECOVERS/sharpens high-frequency detail from the
guide image but adds no sub-patch information that the single /16 forward pass never encoded. Expect
boundary sharpening on partially-resolved structures, not the recovery of a 1-px filament. shift-merge
superres (``factor²`` REAL sub-patch forward passes) does inject genuinely new signal — hence the A/B.

The backbone stays frozen; nothing here trains. Output is L2-normalised per cell to match the pipeline's
cosine-similarity convention (as :func:`shift_merge` / :func:`jbu_snap` do).
"""
from __future__ import annotations

import numpy as np

# PINNED AnyUp commit — torch.hub defaults to branch HEAD, which drifts; a weight/arch change on `main`
# would silently produce different features under the same cache tag (the project's known cache-poison
# bug class). Bump this AND the "v" token in factory.cache_tag together when intentionally upgrading.
_ANYUP_REF = "wimmerth/anyup:351807a9c4287368732cc247f26c7c81c9139af4"

# One upsampler instance per (name, device) — the model + checkpoint load once and stay on GPU.
_MODELS: dict = {}


def _load(name: str, device: str):
    key = (name, device)
    if key in _MODELS:
        return _MODELS[key]
    import torch

    if name == "anyup":
        # anyup_multi_backbone = multi-backbone-trained variant (best cross-backbone generalisation);
        # use_natten=False → pure-PyTorch window attention, no CUDA-extension build required.
        model = torch.hub.load(_ANYUP_REF, "anyup_multi_backbone",
                               use_natten=False, trust_repo=True)
    elif name == "jafar":
        raise NotImplementedError(
            "feat_upsampler='jafar' is not wired yet: verify the torch.hub entrypoint + the DINOv3-L "
            "checkpoint name on the GPU box before enabling (we refuse to ship an unverified hub call).")
    else:
        raise ValueError(f"unknown feat_upsampler={name!r} (expected 'anyup' or 'jafar')")
    model = model.eval().to(device)
    _MODELS[key] = model
    return model


def upsample_grid(name: str, grid_hwd: np.ndarray, guide_bchw, factor: int, device: str) -> np.ndarray:
    """Upsample a coarse ``[G, G, D]`` RAW patch grid to ``[factor·G, factor·G, D]`` (L2-normalised out).

    Args:
        name: upsampler id (``"anyup"``).
        grid_hwd: ``[G, G, D]`` RAW (NOT pre-normalised) patch grid from
            ``Dinov3Encoder._forward_grid(..., normalize=False)`` — AnyUp was trained on raw backbone
            features whose per-patch norm carries salience; pre-normalising would be out-of-distribution.
        guide_bchw: ``[1, 3, res, res]`` ImageNet-normalised image tensor on ``device`` — the HR guide
            (from ``Dinov3Encoder._preprocess``, so preprocessing matches the feature pass).
        factor: densification factor per axis (>=2).
        device: torch device string.

    Returns:
        ``[factor·G, factor·G, D]`` float32, L2-normalised per cell (the pipeline's unit-vector convention).

    Raises loudly (never returns degraded features silently — a silent substitution would bias the A/B):
        - ``factor <= 1`` (a no-op mislabelled as the upsampler arm);
        - the upsampler ignored ``output_size`` / dropped channels (wrong scale served as the right one);
        - non-finite (NaN/Inf) output (would be cached and poison every later run under this tag);
        - a degenerate output (many ~zero-norm cells → random unit directions after renormalisation).
    """
    import torch

    if factor <= 1:
        raise ValueError(f"feat_upsampler={name!r} needs factor>=2 (got {factor}); factor 1 would return "
                         "the coarse grid untouched yet be labelled as the upsampler arm")
    up = _load(name, device)
    g0, g1, d = grid_hwd.shape
    gt = (factor * g0, factor * g1)
    feat = torch.from_numpy(np.ascontiguousarray(grid_hwd)).permute(2, 0, 1).unsqueeze(0)
    feat = feat.to(device=device, dtype=guide_bchw.dtype)                 # [1, D, G, G]
    # chunk the query axis for the heavy ×4 grid so window attention stays memory-bounded
    q_chunk = 2048 if factor >= 4 else None
    with torch.no_grad():
        hr = up(guide_bchw, feat, output_size=gt, q_chunk_size=q_chunk)   # [1, D, factor·G, factor·G]
    if tuple(hr.shape[-2:]) != gt or hr.shape[1] != d:                    # M7: output_size honoured?
        raise RuntimeError(f"{name} returned shape {tuple(hr.shape)}, expected [1, {d}, {gt[0]}, {gt[1]}] "
                           "— output_size/kwargs silently ignored (wrong feature scale)")
    out = hr[0].permute(1, 2, 0).float().cpu().numpy()                    # [factor·G, factor·G, D]
    if not np.isfinite(out).all():                                        # C1: NaN/Inf would cache + poison
        raise RuntimeError(f"{name} produced non-finite (NaN/Inf) features — refusing to cache them")
    norm = np.linalg.norm(out, axis=2, keepdims=True)
    zero_frac = float((norm < 1e-6).mean())                              # H5: mass-collapse → garbage dirs
    if zero_frac > 0.01:
        raise RuntimeError(f"{name}: {zero_frac:.1%} of cells have ~zero norm — degenerate upsampler output")
    return (out / np.maximum(norm, 1e-6)).astype(np.float32)
