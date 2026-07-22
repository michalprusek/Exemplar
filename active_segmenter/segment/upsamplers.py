"""Learned feature upsamplers — SOTA alternatives to bilinear ``F.interpolate`` for lifting the
coarse/fine DINOv3 embeddings to the classical native resolution inside :class:`DINOHeadFusion`.

Bilinear is content-blind (it stretches a 42- or 160-grid with no regard for edges). These are
content-aware or guidance-aware, and kept LIGHTWEIGHT (few params) because the head trains on ~8
labels — a heavy upsampler would be data-starved (cf. the Phase-J learned-boundary NULL). All support
ARBITRARY output size (our upsampling factor is huge, ~10–36×) and are memory-conscious at ≤1536².

- ``dysample`` — DySample (Liu et al., ICCV'23): learn to upsample by learning to SAMPLE. Predicts a
  per-output-pixel 2-D offset from the low-res features and ``grid_sample``s the input there. ~1 conv.
- ``guided``   — FeatUp/JBU-style joint upsampling GUIDED by the native classical priors (which carry
  true pixel-level edges/ridges the DINOv3 grid lacks): a small net turns the guide into a spatial
  modulation that sharpens the bilinearly-lifted features toward the guide's structure.
- ``carafe``   — CARAFE-lite (Wang et al., ICCV'19): content-aware reassembly — predict a normalized
  k×k kernel per output pixel and reassemble the aligned input neighborhood (k small to bound memory).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _boxfilter(t, r: int):
    """Mean over each (2r+1)² window, stride-1, reflect-padded → same spatial size. O(N) enough at
    the low bank resolution where all guided-filter statistics are computed."""
    k = 2 * r + 1
    return F.avg_pool2d(F.pad(t, (r, r, r, r), mode="reflect"), k, stride=1)


def fast_guided_upsample(x_low, guide_native, radius: int = 4, eps: float = 1e-3):
    """Parameter-free Fast Guided Filter (He et al., ECCV'10 / arXiv'15) in JOINT-UPSAMPLING form:
    lift a low-res feature stack ``x_low`` [1,C,h,w] to the native resolution of ``guide_native``
    [1,1,H,W] using the native image as the guide that carries the true pixel-level edges. Used to
    upsample the CHEAP low-res classical bank to native — the inverse of :class:`GuidedUp` (there the
    classical priors guide the DINOv3 features; here the native image guides the classical bank).

    All per-window statistics (means, covariance → local linear model ``x ≈ a·I + b``) are solved at
    the LOW bank resolution; only the final ``a,b`` upsample + ``a·I+b`` apply touch native res, done
    per-channel so the native-res peak is a single channel (memory-safe, matching the head's ethos).
    Zero learned params. Scale-specificity caveat: structures that vanish at ``x_low``'s resolution
    (sub-pixel vessels) cannot be recovered — the guide only sharpens edges of what already sampled."""
    x = x_low.float()
    I = guide_native.float()
    H, W = I.shape[-2:]
    h, w = x.shape[-2:]
    C = x.shape[1]
    r = max(1, min(radius, h - 1, w - 1))                     # radius must fit the low-res grid
    I_lo = F.interpolate(I, size=(h, w), mode="bilinear", align_corners=False)
    mean_I = _boxfilter(I_lo, r)                              # [1,1,h,w]
    var_I = _boxfilter(I_lo * I_lo, r) - mean_I * mean_I
    mean_p = _boxfilter(x, r)                                 # [1,C,h,w]
    cov_Ip = _boxfilter(I_lo * x, r) - mean_I * mean_p        # I_lo broadcasts over C
    a = cov_Ip / (var_I + eps)                               # local slope per channel
    b = mean_p - a * mean_I                                  # local intercept
    a, b = _boxfilter(a, r), _boxfilter(b, r)                # guided-filter output smoothing (low-res)
    out = torch.empty((1, C, H, W), dtype=x.dtype, device=x.device)
    for c in range(C):                                       # per-channel apply → native peak = 1 ch
        a_c = F.interpolate(a[:, c:c + 1], size=(H, W), mode="bilinear", align_corners=False)
        b_c = F.interpolate(b[:, c:c + 1], size=(H, W), mode="bilinear", align_corners=False)
        out[:, c:c + 1] = a_c * I + b_c
    return out


def make_upsampler(name, ch: int, guide_ch: int = 0):
    if name in (None, "bilinear"):
        return None
    if name == "dysample":
        return DySample(ch)
    if name == "guided":
        return GuidedUp(ch, guide_ch)
    if name == "guided_lite":
        return GuidedUp(ch, guide_ch, mem_light=True)
    if name == "carafe":
        return CARAFE(ch)
    raise ValueError(f"unknown upsampler: {name}")


class DySample(nn.Module):
    def __init__(self, ch: int, scope_init: float = 0.25):
        super().__init__()
        self.offset = nn.Conv2d(ch, 2, 3, padding=1)     # per-pixel (dx, dy) sampling offset
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)                 # start at bilinear (zero offset)
        self.scope = nn.Parameter(torch.tensor(float(scope_init)))

    def forward(self, x, out_hw, guide=None):
        b, _, h, w = x.shape
        H, W = out_hw
        off = F.interpolate(self.offset(x), size=(H, W), mode="bilinear", align_corners=False)
        ys = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack([gx, gy], dim=-1)[None].expand(b, -1, -1, -1)   # [B,H,W,2] over input
        norm = torch.tensor([2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1)], device=x.device, dtype=x.dtype)
        grid = base + off.permute(0, 2, 3, 1) * self.scope * norm          # shift in input px units
        return F.grid_sample(x, grid, mode="bilinear", align_corners=False, padding_mode="border")


class GuidedUp(nn.Module):
    """EFFICIENT guided upsampling — Deep-Guided-Filter (Wu et al. CVPR'18) / FeatUp-JBU idea. The naive
    version ran 32-ch convs on the guide at 1536² → OOM. Here ALL guide-dependent computation runs at LOW
    (feature) resolution; the only high-res ops are a 1×1 guide→1ch reduce, two bilinear upsamples and
    two elementwise multiplies — NO high-res convolution. Output = bilinear(x) + A·edge, where the
    per-channel gain A is predicted low-res from [x, guide↓] and upsampled. Zero-init → starts at
    bilinear (safe). ``mem_light``: predict a SINGLE shared gain channel (broadcast) for the tightest
    memory when even the per-channel A is too big."""
    def __init__(self, ch: int, guide_ch: int, mem_light: bool = False):
        super().__init__()
        self.a_ch = 1 if mem_light else ch
        self.reduce = nn.Conv2d(max(guide_ch, 1), 1, 1)          # guide → 1-ch edge (cheap at high res)
        self.coef = nn.Sequential(nn.Conv2d(ch + 1, ch, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(ch, self.a_ch, 1))   # LOW-res per-channel (or shared) gain A
        nn.init.zeros_(self.coef[-1].weight)
        nn.init.zeros_(self.coef[-1].bias)                       # start ≈ bilinear

    def forward(self, x, out_hw, guide=None):
        xu = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        if guide is None:
            return xu
        g1 = self.reduce(guide)                                  # [B,1,H,W] high-res edge signal
        g_lo = F.interpolate(g1, size=x.shape[-2:], mode="bilinear", align_corners=False)
        a = self.coef(torch.cat([x, g_lo], dim=1))              # [B,a_ch,h,w] LOW-res gain
        a = F.interpolate(a, size=out_hw, mode="bilinear", align_corners=False)
        return xu + a * g1                                       # edge-guided correction, elementwise


class CARAFE(nn.Module):
    def __init__(self, ch: int, k: int = 3, enc: int = 32):
        super().__init__()
        self.k = k
        self.comp = nn.Conv2d(ch, enc, 1)
        self.kernel = nn.Conv2d(enc, k * k, 3, padding=1)

    def forward(self, x, out_hw, guide=None):
        b, c, h, w = x.shape
        H, W = out_hw
        wk = F.softmax(self.kernel(self.comp(x)), dim=1)                    # [B,k²,h,w] normalized
        wk = F.interpolate(wk, size=(H, W), mode="bilinear", align_corners=False)
        xu = F.interpolate(x, size=(H, W), mode="nearest")                 # aligned input at out res
        xp = F.unfold(xu, self.k, padding=self.k // 2).view(b, c, self.k * self.k, H, W)
        return (xp * wk[:, None]).sum(2)                                   # content-aware reassembly
