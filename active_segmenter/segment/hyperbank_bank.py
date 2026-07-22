"""VENDORED from github.com/michalprusek/hyperbank (methods differentiable bank, MIT).

The differentiable bank of CLASSICAL operators at native pixel resolution — the small-object
lever a patch-16 foundation grid cannot reach. Copied verbatim (feature_bank.py) so AutoSeg can
fuse it with DINOv3 semantics via the head's n_meta_channels. Upstream: Prusek et al., ICIP 2026.
"""
import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def reflect_pad(img: torch.Tensor, pad: int) -> torch.Tensor:
    """Symmetric pad by ``pad`` on all four sides, safe for images smaller than ``pad``.

    ``torch``'s reflect padding raises when the pad meets or exceeds the dimension it reflects into,
    so EVERY operator in this bank that reflect-pads by a scale-derived radius -- Gaussian blur,
    Sauvola/box windows, structure-tensor and Gabor convolutions, the guided-upsample box -- crashes
    outright on an image smaller than that radius, taking the whole run down. This was found the hard
    way: fixing only ``gaussian_blur`` left the SAME defect in ``_box_avg``, which then failed with
    "padding (75, 75) ... input [1, 1, 66, 58]". One helper used everywhere is the fix; a per-call
    clamp is a fix waiting to be forgotten at the next call site.

    The output is the FULL requested size (input + 2*pad per axis), NOT a clamped one, because every
    caller convolves or pools with a kernel sized to that pad and a shorter pad would shrink the
    result and break the equal-size concatenation the bank depends on. Where reflect cannot reach the
    full width -- only on an image narrower than the pad -- the remainder is filled by REPLICATE,
    which has no size limit. So:

      * pad <= min(H,W)-1  : pure reflect, byte-for-byte the old behaviour -> a no-op on every image
        that runs today, hence safe to deploy mid-campaign.
      * pad larger         : reflect as far as it goes, then extend the edge. The old code raised
        here, so there is no prior number to preserve; any finite boundary is strictly better.
    """
    if pad <= 0:
        return img
    h, w = img.shape[-2], img.shape[-1]
    wl, hl = min(pad, max(w - 1, 0)), min(pad, max(h - 1, 0))
    out = F.pad(img, (wl, wl, hl, hl), mode="reflect") if (wl or hl) else img
    rw, rh = pad - wl, pad - hl
    if rw or rh:
        out = F.pad(out, (rw, rw, rh, rh), mode="replicate")
    return out


def normalize_image(img: torch.Tensor, p_low: float = 0.01, p_high: float = 0.99) -> torch.Tensor:
    """Per-image percentile normalization to [0, 1] — robust against outliers.

    Handles varying illumination/exposure across cell lines. img: [B, 1, H, W].
    """
    B = img.shape[0]
    flat = img.reshape(B, -1)
    lo = torch.quantile(flat, p_low, dim=1, keepdim=True).view(B, 1, 1, 1)
    hi = torch.quantile(flat, p_high, dim=1, keepdim=True).view(B, 1, 1, 1)
    return ((img - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)


def _resolve_sigma(sigma_rel_or_abs: float, ref_size: float) -> float:
    """If sigma < 1, it is a relative fraction; otherwise an absolute pixel value.
    σ=1.0 is interpreted as 1 pixel (absolute), not as 1.0× ref_size.
    """
    if sigma_rel_or_abs < 1.0:
        return sigma_rel_or_abs * ref_size
    return float(sigma_rel_or_abs)


def _resolve_window(win_rel_or_abs, ref_size: float) -> int:
    """Window size: a relative fraction (<1) is rounded to an odd absolute size; otherwise treated as an absolute int."""
    if win_rel_or_abs < 1.0:
        w = int(round(win_rel_or_abs * ref_size))
    else:
        w = int(win_rel_or_abs)
    if w < 3:
        w = 3
    if w % 2 == 0:
        w += 1
    return w


def _gaussian_kernel_1d(sigma: float, device, dtype=torch.float32):
    ksize = max(3, int(2 * 4 * sigma + 0.5) + 1)
    if ksize % 2 == 0:
        ksize += 1
    half = ksize // 2
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    k = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    k = k / k.sum()
    return k, ksize


def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur. img: [B, C, H, W]."""
    if sigma <= 0:
        return img
    k1, ksize = _gaussian_kernel_1d(sigma, img.device, img.dtype)
    half = ksize // 2
    C = img.shape[1]
    kx = k1.view(1, 1, 1, -1).expand(C, 1, 1, ksize)
    ky = k1.view(1, 1, -1, 1).expand(C, 1, ksize, 1)
    img = reflect_pad(img, half)     # full half-width via reflect-then-replicate; keeps output size
    img = F.conv2d(img, kx, groups=C)
    img = F.conv2d(img, ky, groups=C)
    return img


def _central_diff_x(img: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-0.5, 0.0, 0.5]], device=img.device, dtype=img.dtype).view(1, 1, 1, 3)
    return F.conv2d(F.pad(img, (1, 1, 0, 0), mode="reflect"), kx)


def _central_diff_y(img: torch.Tensor) -> torch.Tensor:
    ky = torch.tensor([[-0.5], [0.0], [0.5]], device=img.device, dtype=img.dtype).view(1, 1, 3, 1)
    return F.conv2d(F.pad(img, (0, 0, 1, 1), mode="reflect"), ky)


def hessian(img: torch.Tensor, sigma: float):
    """Hessian of the Gaussian-smoothed image. Returns (Ixx, Iyy, Ixy) scale-normalized by σ²."""
    blurred = gaussian_blur(img, sigma)
    Ix = _central_diff_x(blurred)
    Iy = _central_diff_y(blurred)
    Ixx = _central_diff_x(Ix)
    Iyy = _central_diff_y(Iy)
    Ixy = _central_diff_y(Ix)
    s2 = sigma ** 2
    return s2 * Ixx, s2 * Iyy, s2 * Ixy


def frangi_response(img: torch.Tensor, sigma: float, beta: torch.Tensor, gamma: torch.Tensor,
                     return_polarity: bool = False) -> torch.Tensor:
    """2D Frangi vesselness response at a single scale.

    img: [B, 1, H, W] grayscale
    beta, gamma: scalar tensors (learnable)
    return_polarity: if True, returns [B, 2, H, W] (V_dark, V_bright); otherwise [B, 1, H, W]
    """
    Ixx, Iyy, Ixy = hessian(img, sigma)
    tr = Ixx + Iyy
    sqrt_part = torch.sqrt((Ixx - Iyy).pow(2) + 4 * Ixy.pow(2) + 1e-12)
    lam_a = (tr + sqrt_part) / 2
    lam_b = (tr - sqrt_part) / 2
    abs_a = lam_a.abs()
    abs_b = lam_b.abs()
    swap = abs_a > abs_b
    lam1 = torch.where(swap, lam_b, lam_a)
    lam2 = torch.where(swap, lam_a, lam_b)
    Rb = lam1.abs() / (lam2.abs() + 1e-12)
    S = torch.sqrt(lam1.pow(2) + lam2.pow(2))
    V = torch.exp(-Rb.pow(2) / (2 * beta.pow(2) + 1e-12)) * (
        1.0 - torch.exp(-S.pow(2) / (2 * gamma.pow(2) + 1e-12))
    )
    if not return_polarity:
        return V
    # Polarity: for a dark blob (low intensity inside) λ2 > 0 (positive curvature)
    dark_mask = (lam2 > 0).float()
    bright_mask = 1.0 - dark_mask
    return torch.cat([V * dark_mask, V * bright_mask], dim=1)


class FrangiBank(nn.Module):
    """Bank of Frangi units. Sigmas are relative (≤1.0 = fraction of min(H,W)) or absolute (>1).

    polarity_split=True: each scale produces 2 channels (dark + bright structures).
    """

    def __init__(self, sigmas: Iterable[float] = (0.001, 0.002, 0.004, 0.008, 0.016),
                 polarity_split: bool = True):
        super().__init__()
        self.sigmas_spec = tuple(float(s) for s in sigmas)
        self.polarity_split = polarity_split
        n = len(self.sigmas_spec)
        self.log_beta = nn.Parameter(torch.full((n,), math.log(0.5)))
        self.log_gamma = nn.Parameter(torch.full((n,), math.log(15.0)))

    @property
    def n_channels(self) -> int:
        return len(self.sigmas_spec) * (2 if self.polarity_split else 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        ref = float(min(img.shape[-2], img.shape[-1]))
        sigmas_abs = [_resolve_sigma(s, ref) for s in self.sigmas_spec]
        img255 = img * 255.0
        beta = self.log_beta.exp().clamp(0.05, 10.0)
        gamma = self.log_gamma.exp().clamp(0.05, 50.0)
        responses = []
        for i, s in enumerate(sigmas_abs):
            r = frangi_response(img255, s, beta[i].view(1, 1, 1, 1),
                                 gamma[i].view(1, 1, 1, 1),
                                 return_polarity=self.polarity_split)
            responses.append(r)
        return torch.cat(responses, dim=1)


def _box_avg(img: torch.Tensor, ksize: int) -> torch.Tensor:
    img_p = reflect_pad(img, ksize // 2)     # full window; smaller images stay defined
    return F.avg_pool2d(img_p, kernel_size=ksize, stride=1)


def differentiable_sauvola(img: torch.Tensor, k: torch.Tensor, window_size: int,
                            R: float = 0.5, hard_temp: float = 16.0) -> torch.Tensor:
    """Differentiable Sauvola: T = μ·(1 + k·(σ/R - 1)). Sigmoid surrogate for binarization.

    img in [0, 1], R defaults to 0.5 (half of the dynamic range).
    Detects DARK structures (img < T → output ≈ 1).
    """
    mu = _box_avg(img, window_size)
    img2 = img.pow(2)
    mu2 = _box_avg(img2, window_size)
    sigma = torch.sqrt((mu2 - mu.pow(2)).clamp(min=0.0) + 1e-12)
    T = mu * (1.0 + k * (sigma / R - 1.0))
    return torch.sigmoid(hard_temp * (T - img))


class SauvolaBank(nn.Module):
    """Bank of differentiable Sauvola units. Windows are relative (≤1) or absolute (>1)."""

    def __init__(self, window_sizes: Iterable[float] = (0.015, 0.051, 0.151),
                 hard_temp: float = 16.0):
        super().__init__()
        self.window_sizes_spec = tuple(float(w) for w in window_sizes)
        n = len(self.window_sizes_spec)
        self.k = nn.Parameter(torch.full((n,), 0.2))
        self.hard_temp = hard_temp

    @property
    def n_channels(self) -> int:
        return 2 * len(self.window_sizes_spec)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        ref = float(min(img.shape[-2], img.shape[-1]))
        windows_abs = [_resolve_window(w, ref) for w in self.window_sizes_spec]
        outs = []
        for i, w in enumerate(windows_abs):
            r = differentiable_sauvola(img, self.k[i].view(1, 1, 1, 1), w, hard_temp=self.hard_temp)
            outs.append(r)
            outs.append(1.0 - r)
        return torch.cat(outs, dim=1)


class StructureTensorBank(nn.Module):
    """Structure tensor eigenvalues. Sigmas are relative or absolute."""

    def __init__(self, sigmas: Iterable[float] = (0.002, 0.008)):
        super().__init__()
        self.sigmas_spec = tuple(float(s) for s in sigmas)

    @property
    def n_channels(self) -> int:
        return 2 * len(self.sigmas_spec)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        ref = float(min(img.shape[-2], img.shape[-1]))
        sigmas = [_resolve_sigma(s, ref) for s in self.sigmas_spec]
        outs = []
        for s in sigmas:
            blurred = gaussian_blur(img, s)
            Ix = _central_diff_x(blurred)
            Iy = _central_diff_y(blurred)
            Sxx = gaussian_blur(Ix.pow(2), s)
            Syy = gaussian_blur(Iy.pow(2), s)
            Sxy = gaussian_blur(Ix * Iy, s)
            tr = Sxx + Syy
            sqrt_part = torch.sqrt((Sxx - Syy).pow(2) + 4 * Sxy.pow(2) + 1e-12)
            l1 = (tr + sqrt_part) / 2
            l2 = (tr - sqrt_part) / 2
            outs.append(l1)
            outs.append(l2)
        return torch.cat(outs, dim=1)


class IntensityBank(nn.Module):
    """Intensity bank: returns (img_raw, img, 1-img). img is normalized, img_raw is the original.

    include_raw=False: only (img, 1-img) — backward-compatible 2 channels.
    include_raw=True: (img_raw, img, 1-img) — 3 channels (V8: original grayscale as a feature).
    """

    def __init__(self, include_raw: bool = False):
        super().__init__()
        self.include_raw = include_raw

    @property
    def n_channels(self) -> int:
        return 3 if self.include_raw else 2

    def forward(self, img: torch.Tensor, img_raw: torch.Tensor | None = None) -> torch.Tensor:
        if self.include_raw:
            if img_raw is None:
                img_raw = img
            return torch.cat([img_raw, img, 1.0 - img], dim=1)
        return torch.cat([img, 1.0 - img], dim=1)


class LindebergNJetBank(nn.Module):
    """V13: compact 6-channel Lindeberg N-jet rotation invariant basis.

    For each scale σ returns 3 rotation-invariant polynomial combinations
    of the scale-normalized 2-jet (γ=1):
      L_ξ = σ·L_x,  L_η = σ·L_y,  L_ξξ = σ²·L_xx,  L_ξη = σ²·L_xy,  L_ηη = σ²·L_yy

    Channels (per scale):
      1) |∇L|² = L_ξ² + L_η²   (gradient magnitude squared)
      2) ΔL = L_ξξ + L_ηη       (Laplacian, SIGNED → polarity-aware)
      3) det H = L_ξξ·L_ηη - L_ξη²   (Gaussian curvature)

    From the Hessian eigenvalue identity λ² - Trace·λ + Det = 0, having
    Trace + Det gives full access to any eigenvalue functional, so
    Frangi/Structure-tensor become redundant.

    Source: Lindeberg, "Scale-Covariant and Scale-Invariant Gaussian Derivative
    Networks", J Math Imaging Vis 2022.
    """

    def __init__(self, sigmas: Iterable[float] = (0.016, 0.064)):
        super().__init__()
        self.sigmas_spec = tuple(float(s) for s in sigmas)

    @property
    def n_channels(self) -> int:
        return 3 * len(self.sigmas_spec)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        ref = float(min(img.shape[-2], img.shape[-1]))
        sigmas = [_resolve_sigma(s, ref) for s in self.sigmas_spec]
        outs = []
        for s in sigmas:
            blurred = gaussian_blur(img, s)
            Ix = _central_diff_x(blurred)
            Iy = _central_diff_y(blurred)
            Ixx = _central_diff_x(Ix)
            Iyy = _central_diff_y(Iy)
            Ixy = _central_diff_y(Ix)
            # Scale-normalized (Lindeberg γ=1): σ for 1st order, σ² for 2nd order
            Lx = s * Ix; Ly = s * Iy
            Lxx = s * s * Ixx; Lyy = s * s * Iyy; Lxy = s * s * Ixy
            grad_mag_sq = Lx.pow(2) + Ly.pow(2)
            laplacian = Lxx + Lyy
            det_H = Lxx * Lyy - Lxy.pow(2)
            outs.extend([grad_mag_sq, laplacian, det_H])
        return torch.cat(outs, dim=1)


class LocalMomentsBank(nn.Module):
    """Local Hu moments φ_1, φ_2 in sliding windows.

    φ_1 = μ_20 + μ_02   (rotation-invariant: tr(Hessian-like 2nd moments))
    φ_2 = (μ_20 - μ_02)² + 4·μ_11²  (Hu's 2nd invariant)

    For each window W: 2 channels (φ_1, φ_2). Total: 2 × len(window_sizes).
    Implemented via convolutions with precomputed kernels (u^p · v^q).
    """

    def __init__(self, window_sizes: Iterable[float] = (0.025, 0.051, 0.151)):
        super().__init__()
        self.window_sizes_spec = tuple(float(w) for w in window_sizes)

    @property
    def n_channels(self) -> int:
        return 2 * len(self.window_sizes_spec)

    def _moment_kernels(self, w: int, device, dtype):
        half = w // 2
        coords = torch.arange(-half, half + 1, device=device, dtype=dtype)
        u, v = torch.meshgrid(coords, coords, indexing="ij")
        # Normalize by w² (so moment values are independent of window size)
        norm = float(w * w)
        k_uu = (u * u).view(1, 1, w, w) / norm
        k_vv = (v * v).view(1, 1, w, w) / norm
        k_uv = (u * v).view(1, 1, w, w) / norm
        return k_uu, k_vv, k_uv

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        ref = float(min(img.shape[-2], img.shape[-1]))
        outs = []
        for w_rel in self.window_sizes_spec:
            w = _resolve_window(w_rel, ref)
            half = w // 2
            k_uu, k_vv, k_uv = self._moment_kernels(w, img.device, img.dtype)
            img_p = reflect_pad(img, half)
            m20 = F.conv2d(img_p, k_uu)
            m02 = F.conv2d(img_p, k_vv)
            m11 = F.conv2d(img_p, k_uv)
            phi1 = m20 + m02
            phi2 = (m20 - m02).pow(2) + 4 * m11.pow(2)
            outs.append(phi1)
            outs.append(phi2)
        return torch.cat(outs, dim=1)


class ScatteringBank(nn.Module):
    """Wavelet scattering features (Mallat) via kymatio.

    Fixed-feature multi-scale invariant texture descriptor.
    Output: [B, n_coeffs, H/2^J, W/2^J] → upsampled to (H, W).
    For J=2, L=4: 1 + L·J + L²·J(J-1)/2 = 1 + 8 + 8 = 17 coefficients.
    """

    def __init__(self, J: int = 2, L: int = 4, shape=(1000, 1000)):
        super().__init__()
        self.J = J
        self.L = L
        self.target_shape = shape
        # kymatio Scattering2D is an expensive object — lazily loaded on the first forward
        self._scattering = None

    @property
    def n_channels(self) -> int:
        # 1 + L·J + L²·J·(J-1)/2 (orders 0, 1, 2)
        return 1 + self.L * self.J + (self.L * self.L * self.J * (self.J - 1)) // 2

    def _ensure_scattering(self, img):
        if self._scattering is None or self._last_shape != tuple(img.shape[-2:]):
            from kymatio.torch import Scattering2D
            self._scattering = Scattering2D(J=self.J, shape=img.shape[-2:], L=self.L).to(img.device)
            self._last_shape = tuple(img.shape[-2:])

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        self._ensure_scattering(img)
        H, W = img.shape[-2], img.shape[-1]
        # kymatio requires a contiguous tensor
        x = img.squeeze(1).contiguous()
        s = self._scattering(x)   # [B, n_coeffs, H/2^J, W/2^J]
        s = F.interpolate(s, size=(H, W), mode="bilinear", align_corners=False)
        return s


def _gabor_kernel(sigma: float, theta: float, freq: float, kernel_size: int,
                   device, dtype=torch.float32):
    """2D Gabor kernel — Gaussian envelope × oriented sinusoid.
    Returns (real, imag) — the complex Gabor.
    """
    half = kernel_size // 2
    y, x = torch.meshgrid(
        torch.arange(-half, half + 1, device=device, dtype=dtype),
        torch.arange(-half, half + 1, device=device, dtype=dtype),
        indexing="ij",
    )
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x_t = x * cos_t + y * sin_t
    y_t = -x * sin_t + y * cos_t
    envelope = torch.exp(-(x_t.pow(2) + y_t.pow(2)) / (2 * sigma * sigma))
    arg = 2 * math.pi * freq * x_t
    return envelope * torch.cos(arg), envelope * torch.sin(arg)


class GaborBank(nn.Module):
    """Gabor filter bank — multi-orientation, multi-scale linear feature extractor.

    For each (σ_rel, θ) pair returns magnitude responses |Gabor * I|.
    Frequency is derived from sigma as freq = 1/(λ_factor·σ).
    """

    def __init__(self,
                 sigmas: Iterable[float] = (0.002, 0.008, 0.032),
                 orientations: int = 4,
                 lambda_factor: float = 4.0):
        super().__init__()
        self.sigmas_spec = tuple(float(s) for s in sigmas)
        self.orientations = orientations
        self.lambda_factor = lambda_factor

    @property
    def n_channels(self) -> int:
        return len(self.sigmas_spec) * self.orientations

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        ref = float(min(img.shape[-2], img.shape[-1]))
        sigmas = [_resolve_sigma(s, ref) for s in self.sigmas_spec]
        device = img.device
        dtype = img.dtype
        outs = []
        for sigma in sigmas:
            ksize = max(7, int(2 * 4 * sigma + 0.5) + 1)
            if ksize % 2 == 0: ksize += 1
            half = ksize // 2
            freq = 1.0 / (self.lambda_factor * sigma)
            for k in range(self.orientations):
                theta = math.pi * k / self.orientations
                kr, ki = _gabor_kernel(sigma, theta, freq, ksize, device, dtype)
                kr = kr.view(1, 1, ksize, ksize)
                ki = ki.view(1, 1, ksize, ksize)
                img_p = reflect_pad(img, half)
                resp_r = F.conv2d(img_p, kr)
                resp_i = F.conv2d(img_p, ki)
                magnitude = torch.sqrt(resp_r.pow(2) + resp_i.pow(2) + 1e-12)
                outs.append(magnitude)
        return torch.cat(outs, dim=1)


class LoGBank(nn.Module):
    """V-Ω: Laplacian-of-Gaussian (LoG) blob detector at multiple scales.

    Returns 2 channels per scale:
      1) signed LoG: σ²·(L_xx + L_yy)   — polarity-aware
      2) |LoG|:      |σ²·(L_xx + L_yy)| — polarity-invariant blob detector

    LoG is the optimal blob detector (Lindeberg γ=1 normalization ensures
    cross-scale comparability). The polarity-invariant channel directly
    addresses the SpheroidJ brightfield problem (dark halo + bright interior).
    """

    def __init__(self, sigmas: Iterable[float] = (0.001, 0.002, 0.004, 0.008, 0.016),
                 include_signed: bool = True, include_abs: bool = True):
        super().__init__()
        self.sigmas_spec = tuple(float(s) for s in sigmas)
        self.include_signed = include_signed
        self.include_abs = include_abs

    @property
    def n_channels(self) -> int:
        per_scale = (1 if self.include_signed else 0) + (1 if self.include_abs else 0)
        return len(self.sigmas_spec) * per_scale

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        ref = float(min(img.shape[-2], img.shape[-1]))
        sigmas = [_resolve_sigma(s, ref) for s in self.sigmas_spec]
        outs = []
        for s in sigmas:
            blurred = gaussian_blur(img, s)
            Ix = _central_diff_x(blurred)
            Iy = _central_diff_y(blurred)
            Ixx = _central_diff_x(Ix)
            Iyy = _central_diff_y(Iy)
            log_signed = (s * s) * (Ixx + Iyy)  # γ=1 scale-normalized
            if self.include_signed:
                outs.append(log_signed)
            if self.include_abs:
                outs.append(log_signed.abs())
        return torch.cat(outs, dim=1)


class DetHessianBank(nn.Module):
    """V-Ω: scale-normalized det(Hessian) blob detector.

    det(H) = L_xx · L_yy − L_xy² ; γ²-normalized: σ⁴·det(H).
    Positive at local extrema (blobs); negative at saddle points.
    Returns 2 channels per scale: signed + |det(H)| (polarity-invariant blob).
    """

    def __init__(self, sigmas: Iterable[float] = (0.001, 0.002, 0.004, 0.008, 0.016),
                 include_signed: bool = True, include_abs: bool = True):
        super().__init__()
        self.sigmas_spec = tuple(float(s) for s in sigmas)
        self.include_signed = include_signed
        self.include_abs = include_abs

    @property
    def n_channels(self) -> int:
        per_scale = (1 if self.include_signed else 0) + (1 if self.include_abs else 0)
        return len(self.sigmas_spec) * per_scale

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        ref = float(min(img.shape[-2], img.shape[-1]))
        sigmas = [_resolve_sigma(s, ref) for s in self.sigmas_spec]
        outs = []
        for s in sigmas:
            blurred = gaussian_blur(img, s)
            Ix = _central_diff_x(blurred)
            Iy = _central_diff_y(blurred)
            Ixx = _central_diff_x(Ix)
            Iyy = _central_diff_y(Iy)
            Ixy = _central_diff_y(Ix)
            s4 = (s ** 4)
            det_h = s4 * (Ixx * Iyy - Ixy.pow(2))
            if self.include_signed:
                outs.append(det_h)
            if self.include_abs:
                outs.append(det_h.abs())
        return torch.cat(outs, dim=1)


class MonogenicPCBank(nn.Module):
    """V-Ω: Monogenic-signal phase congruency proxy (FFT-based).

    Implements approximate phase-congruency-style polarity-invariant feature via:
      1) Log-Gabor radial bandpass at K scales (fixed, non-learnable)
      2) Riesz transform (1st order) ⇒ analytical signal in 2D (a, b1, b2)
      3) Local energy E_s = sqrt(a² + b1² + b2²) per scale s
      4) Output channels:
         - per-scale energy E_s (K channels) — polarity-invariant
         - phase congruency-like: sum_s E_s / (sum_s sqrt(a_s² + b_s²)·noise_T + ε)

    This is polarity-invariant by construction: |x*ψ| does not depend on sign(x).
    Implemented in PyTorch with torch.fft, no external deps.
    """

    def __init__(self, n_scales: int = 4, mult: float = 2.1,
                 min_wavelength: float = 6.0, sigma_on_f: float = 0.55,
                 noise_threshold: float = 0.5):
        super().__init__()
        self.n_scales = n_scales
        self.mult = mult
        self.min_wavelength = min_wavelength
        self.sigma_on_f = sigma_on_f
        # Learnable noise threshold (controls phase-congruency noise denoising)
        self.log_noise_threshold = nn.Parameter(torch.tensor(math.log(noise_threshold)))
        self.softness = nn.Parameter(torch.tensor(2.0))

    @property
    def n_channels(self) -> int:
        # Per-scale energy + 1 PC-like aggregate
        return self.n_scales + 1

    def _build_filters(self, H: int, W: int, device, dtype):
        """Constructs K log-Gabor radial filters and Riesz kernels in freq domain.

        Returns:
          log_gabors: [K, H, W] real freq-domain magnitudes
          riesz1, riesz2: [H, W] complex Riesz kernel components
        """
        # Frequency grid
        fy = torch.fft.fftfreq(H, d=1.0, device=device, dtype=dtype).view(H, 1).expand(H, W)
        fx = torch.fft.fftfreq(W, d=1.0, device=device, dtype=dtype).view(1, W).expand(H, W)
        f = torch.sqrt(fx * fx + fy * fy + 1e-12)
        # Avoid log(0) at DC
        f = f.clamp(min=1e-6)

        log_gabors = []
        for s in range(self.n_scales):
            wavelength = self.min_wavelength * (self.mult ** s)
            f0 = 1.0 / wavelength
            log_gabor = torch.exp(-(torch.log(f / f0).pow(2)) /
                                    (2.0 * (math.log(self.sigma_on_f) ** 2)))
            # Set DC to 0
            log_gabor = log_gabor * (f > 1e-5).to(dtype)
            log_gabors.append(log_gabor)
        log_gabors = torch.stack(log_gabors, dim=0)  # [K, H, W]

        # Riesz transforms: H1 = -i·fx/|f|, H2 = -i·fy/|f|
        riesz1 = -1j * (fx / f)
        riesz2 = -1j * (fy / f)
        # Zero DC
        riesz1 = torch.where(f > 1e-5, riesz1, torch.zeros_like(riesz1))
        riesz2 = torch.where(f > 1e-5, riesz2, torch.zeros_like(riesz2))
        return log_gabors, riesz1, riesz2

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        assert C == 1, "MonogenicPCBank expects grayscale [B,1,H,W]"
        device = img.device
        dtype = img.dtype

        log_gabors, riesz1, riesz2 = self._build_filters(H, W, device, dtype)

        # FFT image
        F_img = torch.fft.fft2(img.squeeze(1))  # [B, H, W] complex

        scale_energies = []
        sum_a_sq = torch.zeros((B, H, W), device=device, dtype=dtype)
        sum_b1_sq = torch.zeros_like(sum_a_sq)
        sum_b2_sq = torch.zeros_like(sum_a_sq)
        for s in range(self.n_scales):
            lg = log_gabors[s]  # [H, W]
            # Even component (band-passed image)
            a = torch.fft.ifft2(F_img * lg).real
            # Odd components (Riesz × log_gabor)
            b1 = torch.fft.ifft2(F_img * (riesz1 * lg)).real
            b2 = torch.fft.ifft2(F_img * (riesz2 * lg)).real
            energy = torch.sqrt(a.pow(2) + b1.pow(2) + b2.pow(2) + 1e-12)
            scale_energies.append(energy.unsqueeze(1))
            sum_a_sq = sum_a_sq + a.pow(2)
            sum_b1_sq = sum_b1_sq + b1.pow(2)
            sum_b2_sq = sum_b2_sq + b2.pow(2)

        # Soft phase-congruency aggregate: sum of energies / (total amplitude + T)
        sum_E = torch.cat(scale_energies, dim=1).sum(dim=1, keepdim=True)
        total_amp = torch.sqrt(sum_a_sq + sum_b1_sq + sum_b2_sq + 1e-12).unsqueeze(1)
        T = self.log_noise_threshold.exp()
        # sigmoid soft threshold; softness controls steepness
        pc = sum_E / (total_amp + T + 1e-6)
        pc = torch.sigmoid(self.softness.clamp(0.5, 10.0) * (pc - 0.5))

        # Stack: K per-scale energies + PC aggregate
        return torch.cat(scale_energies + [pc], dim=1)


def _fast_large_gaussian(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Approximate large-σ Gaussian via downsample → blur → upsample.

    For σ > ~10 px, direct separable conv has huge kernel (8σ wide). Instead, downsample
    by factor s = max(1, σ/4) so the post-downsampling sigma ≈ 4 px (manageable),
    blur there, then upsample bilinearly. Loses a bit of accuracy but is much faster
    and adequate for vignette estimation.
    """
    factor = max(1, int(sigma / 4))
    if factor <= 1:
        return gaussian_blur(img, sigma)
    H, W = img.shape[-2], img.shape[-1]
    small = F.avg_pool2d(img, factor, stride=factor)
    small = gaussian_blur(small, sigma / factor)
    return F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)


class VignetteCorrectedBank(nn.Module):
    """Vignette-corrected intensity channel. Divides image by a large-σ Gaussian
    estimate of the illumination field, flattening intensity falloff caused by
    objective shadows / vignetting in microscopy. Output is a single channel
    that suppresses smooth corner darkening while preserving local spheroid
    contrast.
    """

    def __init__(self, sigma_rel: float = 0.20, eps: float = 1e-3):
        super().__init__()
        self.sigma_rel = float(sigma_rel)
        self.eps = float(eps)

    @property
    def n_channels(self) -> int:
        return 1

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        ref = float(min(img.shape[-2], img.shape[-1]))
        sigma_px = max(2.0, self.sigma_rel * ref)
        bg = _fast_large_gaussian(img, sigma_px)
        return img / (bg + self.eps)


class DoGBank(nn.Module):
    """Difference of Gaussians (DoG) at multiple fine scales: DoG_σ = G_σ - G_{2σ}.

    Bandpass filter that preserves boundary frequencies without the broad smoothing
    footprint of Laplacian-of-Gaussian. Useful for sharpening prediction edges that
    LoG/Frangi at large σ would over-smooth.
    """

    def __init__(self, sigmas: Iterable[float] = (1.0, 2.0, 4.0)):
        super().__init__()
        self.sigmas_spec = tuple(float(s) for s in sigmas)

    @property
    def n_channels(self) -> int:
        return len(self.sigmas_spec)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        outs = []
        for s in self.sigmas_spec:
            g_s = gaussian_blur(img, s)
            g_2s = gaussian_blur(img, 2 * s)
            outs.append(g_s - g_2s)
        return torch.cat(outs, dim=1)


class GradMagBank(nn.Module):
    """Multi-scale gradient magnitude. Sigmas are relative or absolute."""

    def __init__(self, sigmas: Iterable[float] = (0.001, 0.004, 0.016)):
        super().__init__()
        self.sigmas_spec = tuple(float(s) for s in sigmas)

    @property
    def n_channels(self) -> int:
        return len(self.sigmas_spec)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        ref = float(min(img.shape[-2], img.shape[-1]))
        sigmas = [_resolve_sigma(s, ref) for s in self.sigmas_spec]
        outs = []
        for s in sigmas:
            blurred = gaussian_blur(img, s)
            Ix = _central_diff_x(blurred)
            Iy = _central_diff_y(blurred)
            mag = torch.sqrt(Ix.pow(2) + Iy.pow(2) + 1e-12) * s  # scale-normalized
            outs.append(mag)
        return torch.cat(outs, dim=1)


class HyperBank(nn.Module):
    """Differentiable feature bank with GroupNorm → spatial conv head fusion.

    V2 improvements over V1:
    - Polarity-aware Frangi (2× channels for dark/bright structures)
    - Gradient magnitude bank (multi-scale edges)
    - Dilated 3×3 conv head instead of a pure per-pixel linear regression — adds
      light spatial context (RF ≈ 5 px) with few parameters (~9× channel count).
    """

    def __init__(
        self,
        frangi_sigmas: Iterable[float] = (0.001, 0.002, 0.004, 0.008, 0.016),
        sauvola_windows: Iterable[float] = (0.015, 0.051, 0.151),
        struct_sigmas: Iterable[float] = (0.002, 0.008),
        gradmag_sigmas: Iterable[float] = (0.001, 0.004, 0.016),
        sauvola_temp: float = 16.0,
        polarity_frangi: bool = True,
        spatial_kernel: int = 3,
        spatial_dilation: int = 4,
        n_meta_channels: int = 0,
        use_sauvola: bool = True,
        use_struct: bool = True,            # ablation flag
        use_gradmag: bool = True,           # ablation flag
        use_intensity: bool = True,          # ablation flag (drop IntensityBank)
        use_gabor: bool = False,
        gabor_sigmas: Iterable[float] = (0.002, 0.008, 0.032),
        gabor_orientations: int = 4,
        normalize_input: bool = True,
        normalize_p_low: float = 0.01,
        normalize_p_high: float = 0.99,
        include_raw_intensity: bool = False,
        learnable_min_area: bool = False,
        min_area_rel_init: float = 0.0002,
        soft_area_kernel_rel: float = 0.05,
        learnable_sauvola_threshold: bool = False,
        sauvola_thresh_window_rel_init: float = 0.051,  # 51 px @ 1000
        sauvola_thresh_k_init: float = 0.2,
        sauvola_thresh_temp: float = 8.0,
        use_moments: bool = False,           # V9: Hu φ_1, φ_2 in local windows
        moments_windows: Iterable[float] = (0.025, 0.051, 0.151),
        use_scattering: bool = False,        # V9: Mallat wavelet scattering
        scattering_J: int = 2,
        scattering_L: int = 4,
        use_lindeberg: bool = False,         # V13: COMPACT 6-ch Lindeberg N-jet
        lindeberg_sigmas: Iterable[float] = (0.016, 0.064),
        # V-Ω: new polarity-invariant blob detectors for Decay+SpheroidJ
        use_log: bool = False,               # LoG bank (signed + |LoG|)
        log_sigmas: Iterable[float] = (0.001, 0.002, 0.004, 0.008, 0.016),
        log_include_signed: bool = True,
        log_include_abs: bool = True,
        use_dethessian: bool = False,        # det(Hessian) bank
        dethessian_sigmas: Iterable[float] = (0.001, 0.002, 0.004, 0.008, 0.016),
        dethessian_include_signed: bool = True,
        dethessian_include_abs: bool = True,
        use_phase_congruency: bool = False,  # MonogenicPC bank
        pc_n_scales: int = 4,
        pc_min_wavelength: float = 6.0,
        # V-Ω: vignette correction + DoG fine-scale (sharper boundary, corner-shadow rejection)
        use_vignette: bool = False,
        vignette_sigma_rel: float = 0.20,
        use_dog: bool = False,
        dog_sigmas: Iterable[float] = (1.0, 2.0, 4.0),
    ):
        super().__init__()
        # V13/V14: use_lindeberg replaces Frangi/Struct/GradMag (which are redundant
        # with the Lindeberg differential invariants). Sauvola + Intensity remain optional.
        self.use_lindeberg = use_lindeberg
        if use_lindeberg:
            self.lindeberg = LindebergNJetBank(lindeberg_sigmas)
            self.frangi = None; self.struct = None; self.gradmag = None
            self.gabor = None; self.moments = None; self.scattering = None
            # V14: Sauvola + Intensity are also optional in lindeberg mode
            self.sauvola = SauvolaBank(sauvola_windows, hard_temp=sauvola_temp) if use_sauvola else None
            self.intensity = IntensityBank(include_raw=include_raw_intensity)
        else:
            self.lindeberg = None
            self.frangi = FrangiBank(frangi_sigmas, polarity_split=polarity_frangi)
            self.sauvola = SauvolaBank(sauvola_windows, hard_temp=sauvola_temp) if use_sauvola else None
            self.struct = StructureTensorBank(struct_sigmas) if use_struct else None
            self.intensity = IntensityBank(include_raw=include_raw_intensity) if use_intensity else None
            self.gradmag = GradMagBank(gradmag_sigmas) if use_gradmag else None
            self.gabor = GaborBank(gabor_sigmas, gabor_orientations) if use_gabor else None
            self.moments = LocalMomentsBank(moments_windows) if use_moments else None
            self.scattering = ScatteringBank(J=scattering_J, L=scattering_L) if use_scattering else None
        self.use_sauvola = use_sauvola
        self.use_gabor = use_gabor
        self.use_moments = use_moments
        self.use_scattering = use_scattering
        self.include_raw_intensity = include_raw_intensity

        # V-Ω: polarity-invariant blob banks
        self.use_log = use_log
        self.use_dethessian = use_dethessian
        self.use_phase_congruency = use_phase_congruency
        self.log_bank = LoGBank(log_sigmas, log_include_signed, log_include_abs) if use_log else None
        self.dethessian_bank = DetHessianBank(
            dethessian_sigmas, dethessian_include_signed, dethessian_include_abs) if use_dethessian else None
        self.pc_bank = MonogenicPCBank(n_scales=pc_n_scales,
                                          min_wavelength=pc_min_wavelength) if use_phase_congruency else None
        # V-Ω+: vignette + DoG (sharper boundaries, corner-shadow rejection)
        self.use_vignette = use_vignette
        self.use_dog = use_dog
        self.vignette_bank = VignetteCorrectedBank(sigma_rel=vignette_sigma_rel) if use_vignette else None
        self.dog_bank = DoGBank(dog_sigmas) if use_dog else None

        # V8: learnable min_area_rel — relative to image area
        self.learnable_min_area = learnable_min_area
        self.soft_area_kernel_rel = soft_area_kernel_rel
        if learnable_min_area:
            self.log_min_area_rel = nn.Parameter(torch.tensor(math.log(min_area_rel_init)))
        else:
            self.register_buffer("log_min_area_rel", torch.tensor(math.log(min_area_rel_init)))

        # V8: learnable Sauvola threshold (final binarization layer)
        self.learnable_sauvola_threshold = learnable_sauvola_threshold
        self.sauvola_thresh_temp = sauvola_thresh_temp
        if learnable_sauvola_threshold:
            self.log_sauvola_thresh_window_rel = nn.Parameter(
                torch.tensor(math.log(sauvola_thresh_window_rel_init)))
            self.sauvola_thresh_k = nn.Parameter(torch.tensor(sauvola_thresh_k_init))
        else:
            self.register_buffer("log_sauvola_thresh_window_rel",
                                 torch.tensor(math.log(sauvola_thresh_window_rel_init)))
            self.register_buffer("sauvola_thresh_k", torch.tensor(sauvola_thresh_k_init))

        self.normalize_input = normalize_input
        self.normalize_p_low = normalize_p_low
        self.normalize_p_high = normalize_p_high

        if use_lindeberg:
            n_frangi = n_struct = n_gradmag = 0
            n_gabor = n_moments = n_scattering = 0
            n_lindeberg = self.lindeberg.n_channels
            n_sauvola = self.sauvola.n_channels if use_sauvola else 0
            n_int = self.intensity.n_channels  # always present in lindeberg mode
        else:
            n_frangi = self.frangi.n_channels
            n_sauvola = self.sauvola.n_channels if use_sauvola else 0
            n_struct = self.struct.n_channels if use_struct else 0
            n_int = self.intensity.n_channels if use_intensity else 0
            n_gradmag = self.gradmag.n_channels if use_gradmag else 0
            n_gabor = self.gabor.n_channels if use_gabor else 0
            n_moments = self.moments.n_channels if use_moments else 0
            n_scattering = self.scattering.n_channels if use_scattering else 0
            n_lindeberg = 0
        # V-Ω banks: can be added in both modes (lindeberg and classical)
        n_log = self.log_bank.n_channels if use_log else 0
        n_deth = self.dethessian_bank.n_channels if use_dethessian else 0
        n_pc = self.pc_bank.n_channels if use_phase_congruency else 0
        n_vignette = self.vignette_bank.n_channels if use_vignette else 0
        n_dog = self.dog_bank.n_channels if use_dog else 0

        if use_lindeberg:
            self.gn_lindeberg = nn.GroupNorm(num_groups=1, num_channels=n_lindeberg, affine=False)
            if use_sauvola:
                self.gn_sauvola = nn.GroupNorm(num_groups=1, num_channels=n_sauvola, affine=False)
            self.gn_int = nn.GroupNorm(num_groups=1, num_channels=n_int, affine=False)
        else:
            self.gn_frangi = nn.GroupNorm(num_groups=1, num_channels=n_frangi, affine=False)
            if use_sauvola:
                self.gn_sauvola = nn.GroupNorm(num_groups=1, num_channels=n_sauvola, affine=False)
            if use_struct:
                self.gn_struct = nn.GroupNorm(num_groups=1, num_channels=n_struct, affine=False)
            if use_intensity:
                self.gn_int = nn.GroupNorm(num_groups=1, num_channels=n_int, affine=False)
            if use_gradmag:
                self.gn_gradmag = nn.GroupNorm(num_groups=1, num_channels=n_gradmag, affine=False)
            if use_gabor:
                self.gn_gabor = nn.GroupNorm(num_groups=1, num_channels=n_gabor, affine=False)
            if use_moments:
                self.gn_moments = nn.GroupNorm(num_groups=1, num_channels=n_moments, affine=False)
            if use_scattering:
                self.gn_scattering = nn.GroupNorm(num_groups=1, num_channels=n_scattering, affine=False)
        if use_log:
            self.gn_log = nn.GroupNorm(num_groups=1, num_channels=n_log, affine=False)
        if use_dethessian:
            self.gn_deth = nn.GroupNorm(num_groups=1, num_channels=n_deth, affine=False)
        if use_phase_congruency:
            self.gn_pc = nn.GroupNorm(num_groups=1, num_channels=n_pc, affine=False)
        if use_vignette:
            self.gn_vignette = nn.GroupNorm(num_groups=1, num_channels=n_vignette, affine=False)
        if use_dog:
            self.gn_dog = nn.GroupNorm(num_groups=1, num_channels=n_dog, affine=False)

        n_total = (n_frangi + n_sauvola + n_struct + n_int + n_gradmag
                    + n_gabor + n_moments + n_scattering + n_lindeberg
                    + n_log + n_deth + n_pc + n_vignette + n_dog)
        n_total_with_meta = n_total + n_meta_channels
        # Spatial 3×3 dilated conv head: per-pixel classifier with spatial
        # context. Input is n_total + n_meta_channels (V4 adds external
        # support-conditioned channels).
        pad = spatial_kernel // 2 * spatial_dilation
        self.head = nn.Conv2d(
            in_channels=n_total_with_meta, out_channels=1, kernel_size=spatial_kernel,
            dilation=spatial_dilation, padding=pad, bias=True,
        )
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.n_total = n_total
        self.n_meta_channels = n_meta_channels
        self.n_total_with_meta = n_total_with_meta
        self.split = (n_frangi, n_sauvola, n_struct, n_int, n_gradmag)

    def feature_maps(self, img: torch.Tensor) -> torch.Tensor:
        img_raw = img
        if self.normalize_input:
            img = normalize_image(img, self.normalize_p_low, self.normalize_p_high)
        if self.use_lindeberg:
            parts = [self.gn_lindeberg(self.lindeberg(img))]
            if self.sauvola is not None:
                parts.append(self.gn_sauvola(self.sauvola(img)))
            if self.intensity is not None:
                parts.append(self.gn_int(self.intensity(img, img_raw)))
            if self.use_log:
                parts.append(self.gn_log(self.log_bank(img)))
            if self.use_dethessian:
                parts.append(self.gn_deth(self.dethessian_bank(img)))
            if self.use_phase_congruency:
                parts.append(self.gn_pc(self.pc_bank(img)))
            return torch.cat(parts, dim=1)
        parts = [self.gn_frangi(self.frangi(img))]
        if self.use_sauvola:
            parts.append(self.gn_sauvola(self.sauvola(img)))
        if self.struct is not None:
            parts.append(self.gn_struct(self.struct(img)))
        if self.intensity is not None:
            parts.append(self.gn_int(self.intensity(img, img_raw)))
        if self.gradmag is not None:
            parts.append(self.gn_gradmag(self.gradmag(img)))
        if self.use_gabor:
            parts.append(self.gn_gabor(self.gabor(img)))
        if self.use_moments:
            parts.append(self.gn_moments(self.moments(img)))
        if self.use_scattering:
            parts.append(self.gn_scattering(self.scattering(img)))
        if self.use_log:
            parts.append(self.gn_log(self.log_bank(img)))
        if self.use_dethessian:
            parts.append(self.gn_deth(self.dethessian_bank(img)))
        if self.use_phase_congruency:
            parts.append(self.gn_pc(self.pc_bank(img)))
        if self.use_vignette:
            parts.append(self.gn_vignette(self.vignette_bank(img)))
        if self.use_dog:
            parts.append(self.gn_dog(self.dog_bank(img)))
        return torch.cat(parts, dim=1)

    def soft_area_filter(self, prob: torch.Tensor) -> torch.Tensor:
        """Soft min-area regularizer — differentiable approximation of a component filter.

        Pixels in a local window with low mean probability (i.e. likely in a small
        component or at the edge of noise) are suppressed.

        prob: [B, 1, H, W] in [0, 1]. Uses self.log_min_area_rel as the threshold.
        """
        H, W = prob.shape[-2], prob.shape[-1]
        ref = min(H, W)
        kernel = max(3, int(self.soft_area_kernel_rel * ref))
        if kernel % 2 == 0: kernel += 1
        pad = kernel // 2
        local_avg = F.avg_pool2d(prob, kernel, stride=1, padding=pad)
        # min_area_rel is a fraction of the whole image. Within the kernel area (= kernel²),
        # required avg = (min_area_rel · H·W) / kernel² = min_area_rel · (HW / kernel²)
        min_area_rel = self.log_min_area_rel.exp()
        required_avg = (min_area_rel * H * W / (kernel ** 2)).clamp(0.01, 0.99)
        keep = torch.sigmoid(8.0 * (local_avg - required_avg))
        return prob * keep

    def min_area_pixels(self, H: int, W: int) -> int:
        """Current min_area in absolute pixels for the given image size."""
        return int(self.log_min_area_rel.exp().item() * H * W)

    def differentiable_sauvola_threshold(self, prob: torch.Tensor) -> torch.Tensor:
        """Learnable Sauvola threshold applied to the probability map.

        T(x,y) = μ_w(x,y) · (1 + k · (σ_w/R - 1))    R=0.5 (half range of [0,1])
        Soft binary surrogate: sigmoid(temp · (prob - T)).
        Both window and k are nn.Parameter when learnable_sauvola_threshold=True.
        """
        H, W = prob.shape[-2], prob.shape[-1]
        ref = min(H, W)
        # Window is discretized from a learnable rel-fraction. The gradient through
        # window is blocked by the discrete kernel size, but k is learned via prob → T → loss.
        window_rel = self.log_sauvola_thresh_window_rel.exp().clamp(0.005, 0.5).item()
        kernel = max(3, int(window_rel * ref))
        if kernel % 2 == 0: kernel += 1
        prob_p = reflect_pad(prob, kernel // 2)
        mu = F.avg_pool2d(prob_p, kernel, stride=1)
        mu2 = F.avg_pool2d(prob_p.pow(2), kernel, stride=1)
        sigma = torch.sqrt((mu2 - mu.pow(2)).clamp(min=0.0) + 1e-12)
        R = 0.5
        T = mu * (1.0 + self.sauvola_thresh_k * (sigma / R - 1.0))
        return torch.sigmoid(self.sauvola_thresh_temp * (prob - T))

    def forward_score(self, img: torch.Tensor, meta_channels: torch.Tensor | None = None) -> torch.Tensor:
        """Raw score map (pre-sigmoid). For V5 — separates channel mix from threshold."""
        feats = self.feature_maps(img)
        if self.n_meta_channels > 0:
            if meta_channels is None:
                raise ValueError(f"HyperBank has n_meta_channels={self.n_meta_channels}, "
                                  f"but forward received meta_channels=None")
            feats = torch.cat([feats, meta_channels], dim=1)
        return self.head(feats)

    def forward(self, img: torch.Tensor, meta_channels: torch.Tensor | None = None) -> torch.Tensor:
        """img: [B, 1, H, W] → sigmoid(score). Backward-compat default."""
        return torch.sigmoid(self.forward_score(img, meta_channels))

    def feature_importance(self) -> torch.Tensor:
        """Diagnostic: per-channel weight (sum of absolute values of the conv kernel)."""
        with torch.no_grad():
            # head.weight: [1, n_total, k, k] — sum over spatial dims
            return self.head.weight.detach().abs().sum(dim=(0, 2, 3)).cpu()
