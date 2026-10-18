"""The 35 frozen classical priors, at full native resolution.

A 16-pixel patch grid cannot describe structure narrower than a patch, so beside the two DINOv3 views the readout
reads 35 classical filter responses computed on the whole native image in one pass (no downscaling, no tiling). They
are frozen: nothing here is fitted, and the scales are fixed for every dataset.

Input: the grey image (channel mean), min-max normalised per image (`_gray01`), then clipped to its 1st-99th
percentiles (`normalize_image`). Channel order, each family standardised per image by GroupNorm with one group and no
affine parameters (eps 1e-5):

    Frangi vesselness        10   sigma 1, 2, 4, 8, 16 px; per sigma dark-on-bright then bright-on-dark
    Sauvola threshold         6   windows 15, 51, 151 px; per window r then 1 - r
    structure tensor          4   sigma 2, 8 px; per sigma lambda_1 then lambda_2
    intensity                 2   the normalised image and its complement
    gradient magnitude        3   sigma 0.001, 0.004, 0.016 x min(H, W)
    Laplacian of Gaussian    10   sigma 0.001, 0.002, 0.004, 0.008, 0.016 x min(H, W); per sigma signed then |.|

A sigma or window below 1 is a fraction of min(H, W) (`_resolve_sigma`), so the gradient and LoG scales grow with the
field. The filter parameters (Frangi beta 0.5 and gamma 15, Sauvola k 0.2) are kept as frozen float32 parameters
exactly as the original bank stores them, so `exp(log(0.5))` rounds as it did there.

Source: active_segmenter/segment/hyperbank_bank.py (vendored from github.com/michalprusek/hyperbank, MIT) and
head_fusion_backend.py `_gray01`, `_bank_module`, `_classical` (AutoSeg); only the pieces this configuration runs.
Line references point to the authors' research code, which is not public.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FRANGI_SIGMAS = (1.0, 2.0, 4.0, 8.0, 16.0)          # head_fusion_backend.py:2070
SAUVOLA_WINDOWS = (15, 51, 151)                     # head_fusion_backend.py:2068
STRUCT_SIGMAS = (2.0, 8.0)                          # head_fusion_backend.py:2071
GRADMAG_SIGMAS = (0.001, 0.004, 0.016)              # HyperBank default (fractions of min(H, W))
LOG_SIGMAS = (0.001, 0.002, 0.004, 0.008, 0.016)    # HyperBank default (fractions of min(H, W))


def _channel_names() -> list:
    names = []
    for s in FRANGI_SIGMAS:
        names += [f"frangi_s{s:g}_dark", f"frangi_s{s:g}_bright"]
    for w in SAUVOLA_WINDOWS:
        names += [f"sauvola_w{w}", f"sauvola_w{w}_inv"]
    for s in STRUCT_SIGMAS:
        names += [f"struct_s{s:g}_l1", f"struct_s{s:g}_l2"]
    names += ["intensity", "intensity_inv"]
    names += [f"gradmag_s{s:g}" for s in GRADMAG_SIGMAS]
    for s in LOG_SIGMAS:
        names += [f"log_s{s:g}_signed", f"log_s{s:g}_abs"]
    return names


PRIOR_CHANNELS = _channel_names()                   # the 35 channels, in the order the bank returns them
FAMILY_SLICES = {"frangi": slice(0, 10), "sauvola": slice(10, 16), "struct": slice(16, 20),
                 "intensity": slice(20, 22), "gradmag": slice(22, 25), "log": slice(25, 35)}


def _gray01(image) -> np.ndarray:
    """The grey image fed to the bank: channel mean, per-image min-max. (head_fusion_backend.py:343-347)"""
    a = np.asarray(image).astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    return (a - a.min()) / (np.ptp(a) + 1e-6)


def reflect_pad(img: torch.Tensor, pad: int) -> torch.Tensor:
    """Symmetric pad by `pad` on all four sides; where reflect cannot reach (an image narrower than the pad) the
    remainder is filled by replicate. (hyperbank_bank.py:15-45)"""
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
    """Per-image percentile normalisation to [0, 1]. img: [B, 1, H, W]. (hyperbank_bank.py:48-57)
    torch.quantile refuses inputs above 2^24 elements."""
    B = img.shape[0]
    flat = img.reshape(B, -1)
    lo = torch.quantile(flat, p_low, dim=1, keepdim=True).view(B, 1, 1, 1)
    hi = torch.quantile(flat, p_high, dim=1, keepdim=True).view(B, 1, 1, 1)
    return ((img - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)


def _resolve_sigma(sigma_rel_or_abs: float, ref_size: float) -> float:
    """Below 1: a fraction of ref_size; otherwise pixels (1.0 is one pixel). (hyperbank_bank.py:60-66)"""
    if sigma_rel_or_abs < 1.0:
        return sigma_rel_or_abs * ref_size
    return float(sigma_rel_or_abs)


def _resolve_window(win_rel_or_abs, ref_size: float) -> int:
    """Window size: a fraction (< 1) is rounded to an odd pixel size, otherwise an integer, at least 3, odd.
    (hyperbank_bank.py:69-79)"""
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
    """(hyperbank_bank.py:82-90)"""
    ksize = max(3, int(2 * 4 * sigma + 0.5) + 1)
    if ksize % 2 == 0:
        ksize += 1
    half = ksize // 2
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    k = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    k = k / k.sum()
    return k, ksize


def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur, reflect-padded to keep the size. img: [B, C, H, W]. (hyperbank_bank.py:93-105)"""
    if sigma <= 0:
        return img
    k1, ksize = _gaussian_kernel_1d(sigma, img.device, img.dtype)
    half = ksize // 2
    C = img.shape[1]
    kx = k1.view(1, 1, 1, -1).expand(C, 1, 1, ksize)
    ky = k1.view(1, 1, -1, 1).expand(C, 1, ksize, 1)
    img = reflect_pad(img, half)
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
    """Hessian of the Gaussian-smoothed image, scale-normalised by sigma^2. (hyperbank_bank.py:118-127)"""
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
    """2-D Frangi vesselness at one scale; with return_polarity [B, 2, H, W] = (dark, bright).
    (hyperbank_bank.py:130-158)"""
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
    dark_mask = (lam2 > 0).float()          # a dark structure (low inside) has lambda_2 > 0
    bright_mask = 1.0 - dark_mask
    return torch.cat([V * dark_mask, V * bright_mask], dim=1)


class FrangiBank(nn.Module):
    """Frangi at several scales, polarity-split, on the image scaled to [0, 255]. (hyperbank_bank.py:161-192)"""

    def __init__(self, sigmas: Iterable[float], polarity_split: bool = True):
        super().__init__()
        self.sigmas_spec = tuple(float(s) for s in sigmas)
        self.polarity_split = polarity_split
        n = len(self.sigmas_spec)
        self.log_beta = nn.Parameter(torch.full((n,), math.log(0.5)))
        self.log_gamma = nn.Parameter(torch.full((n,), math.log(15.0)))

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
    img_p = reflect_pad(img, ksize // 2)
    return F.avg_pool2d(img_p, kernel_size=ksize, stride=1)


def differentiable_sauvola(img: torch.Tensor, k: torch.Tensor, window_size: int,
                           R: float = 0.5, hard_temp: float = 16.0) -> torch.Tensor:
    """Sauvola T = mu (1 + k (sigma / R - 1)), sigmoid(temp (T - img)): ~1 on dark structures.
    (hyperbank_bank.py:200-212)"""
    mu = _box_avg(img, window_size)
    img2 = img.pow(2)
    mu2 = _box_avg(img2, window_size)
    sigma = torch.sqrt((mu2 - mu.pow(2)).clamp(min=0.0) + 1e-12)
    T = mu * (1.0 + k * (sigma / R - 1.0))
    return torch.sigmoid(hard_temp * (T - img))


class SauvolaBank(nn.Module):
    """(hyperbank_bank.py:215-238)"""

    def __init__(self, window_sizes: Iterable[float], hard_temp: float = 16.0):
        super().__init__()
        self.window_sizes_spec = tuple(float(w) for w in window_sizes)
        n = len(self.window_sizes_spec)
        self.k = nn.Parameter(torch.full((n,), 0.2))
        self.hard_temp = hard_temp

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
    """Eigenvalues of the structure tensor. (hyperbank_bank.py:241-269)"""

    def __init__(self, sigmas: Iterable[float]):
        super().__init__()
        self.sigmas_spec = tuple(float(s) for s in sigmas)

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
    """(img, 1 - img) of the normalised image. (hyperbank_bank.py:272-292, include_raw=False)"""

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return torch.cat([img, 1.0 - img], dim=1)


class GradMagBank(nn.Module):
    """Scale-normalised gradient magnitude. (hyperbank_bank.py:744-765)"""

    def __init__(self, sigmas: Iterable[float]):
        super().__init__()
        self.sigmas_spec = tuple(float(s) for s in sigmas)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        ref = float(min(img.shape[-2], img.shape[-1]))
        sigmas = [_resolve_sigma(s, ref) for s in self.sigmas_spec]
        outs = []
        for s in sigmas:
            blurred = gaussian_blur(img, s)
            Ix = _central_diff_x(blurred)
            Iy = _central_diff_y(blurred)
            mag = torch.sqrt(Ix.pow(2) + Iy.pow(2) + 1e-12) * s
            outs.append(mag)
        return torch.cat(outs, dim=1)


class LoGBank(nn.Module):
    """Scale-normalised Laplacian of Gaussian, signed and absolute. (hyperbank_bank.py:491-530)"""

    def __init__(self, sigmas: Iterable[float]):
        super().__init__()
        self.sigmas_spec = tuple(float(s) for s in sigmas)

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
            log_signed = (s * s) * (Ixx + Iyy)
            outs.append(log_signed)
            outs.append(log_signed.abs())
        return torch.cat(outs, dim=1)


class PriorBank(nn.Module):
    """The six families above with their per-family GroupNorm, frozen. `feature_maps` is HyperBank.feature_maps for
    the configuration HyperBank(frangi_sigmas=(1, 2, 4, 8, 16), sauvola_windows=(15, 51, 151), struct_sigmas=(2, 8),
    use_log=True), the bank of the reported method. (hyperbank_bank.py:841-1103, head_fusion_backend.py:2047-2076)"""

    def __init__(self):
        super().__init__()
        self.frangi = FrangiBank(FRANGI_SIGMAS, polarity_split=True)
        self.sauvola = SauvolaBank(SAUVOLA_WINDOWS, hard_temp=16.0)
        self.struct = StructureTensorBank(STRUCT_SIGMAS)
        self.intensity = IntensityBank()
        self.gradmag = GradMagBank(GRADMAG_SIGMAS)
        self.log_bank = LoGBank(LOG_SIGMAS)
        self.gn_frangi = nn.GroupNorm(num_groups=1, num_channels=10, affine=False)
        self.gn_sauvola = nn.GroupNorm(num_groups=1, num_channels=6, affine=False)
        self.gn_struct = nn.GroupNorm(num_groups=1, num_channels=4, affine=False)
        self.gn_int = nn.GroupNorm(num_groups=1, num_channels=2, affine=False)
        self.gn_gradmag = nn.GroupNorm(num_groups=1, num_channels=3, affine=False)
        self.gn_log = nn.GroupNorm(num_groups=1, num_channels=10, affine=False)
        for p in self.parameters():
            p.requires_grad_(False)

    def feature_maps(self, img: torch.Tensor) -> torch.Tensor:
        """[B, 1, H, W] grey image in [0, 1] -> [B, 35, H, W]."""
        img = normalize_image(img, 0.01, 0.99)
        parts = [self.gn_frangi(self.frangi(img)),
                 self.gn_sauvola(self.sauvola(img)),
                 self.gn_struct(self.struct(img)),
                 self.gn_int(self.intensity(img)),
                 self.gn_gradmag(self.gradmag(img)),
                 self.gn_log(self.log_bank(img))]
        return torch.cat(parts, dim=1)


def make_prior_bank(device) -> PriorBank:
    return PriorBank().to(device).eval()


def priors(image, bank: PriorBank, device) -> torch.Tensor:
    """The priors of one image at its native resolution: [1, 35, H, W] float32 on `device`.
    (head_fusion_backend.py:2324-2429 with the reported configuration: grey channel, no CLAHE, no z-scoring,
    no tiling, no cap since max_side = 1e9)"""
    g = _gray01(image)
    t = torch.from_numpy(g)[None, None].to(device)
    with torch.no_grad():
        feats = bank.feature_maps(t)
    return feats
