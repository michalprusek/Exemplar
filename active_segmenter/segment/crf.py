"""Native-resolution boundary recovery for the frozen (INSID3-style) backend.

Mode is chosen by the ``ASG_CRF`` env var (default ``dense``):
- ``dense`` (default): dense CRF (pydensecrf) with a bilateral pairwise term keyed on the
  image, so the coarse DINOv3 correspondence prob snaps to image edges — INSID3's 1024^2+CRF
  recipe. If pydensecrf cannot be imported this **raises** (no silent substitution — a silent
  guided-filter fallback would undersell the dense baseline in a benchmark).
- ``guided``: force the edge-aware guided filter (pure numpy/scipy) even if pydensecrf is
  present — the config that AVOIDS collapsing thin structures.
- ``none``: return the raw prob unchanged (no refinement).
Only the INSID3 baseline calls ``refine_probability``; our method's refine is ``--refine amodal``.
"""
from __future__ import annotations

import numpy as np

_CRF_MARK = [False]


def guided_upsample(image, prob, radius: int = 8, eps: float = 1e-3) -> np.ndarray:
    """Edge-aware guided filter (He et al.): filter ``prob`` using ``image`` as guide so
    the output follows the image's edges. Pure numpy/scipy fallback for dense CRF."""
    from scipy.ndimage import uniform_filter

    I = np.asarray(image, np.float32)
    if I.ndim == 3:
        I = I[..., :3].mean(-1)
    ptp = float(I.max() - I.min())
    I = (I - I.min()) / (ptp + 1e-6)
    p = np.asarray(prob, np.float32)
    win = 2 * radius + 1
    mean_I = uniform_filter(I, win)
    mean_p = uniform_filter(p, win)
    corr_I = uniform_filter(I * I, win)
    corr_Ip = uniform_filter(I * p, win)
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    q = uniform_filter(a, win) * I + uniform_filter(b, win)
    return np.clip(q, 0.0, 1.0).astype(np.float32)


def refine_probability(image, prob, n_iters: int = 5) -> np.ndarray:
    """Sharpen a native-res ``[H, W]`` fg probability toward image edges. Mode via the
    ``ASG_CRF`` env var (default ``dense``): ``none`` returns the prob unchanged; ``guided``
    forces the guided filter; ``dense`` uses dense CRF (pydensecrf) and **raises** if it is
    unavailable (never silently substitutes the weaker guided filter, which would undersell
    the dense baseline). Any other value is a hard error."""
    p = np.asarray(prob, np.float32)
    import os as _os
    import sys as _s
    _mode = _os.environ.get("ASG_CRF", "dense").strip().lower()
    if _mode not in ("dense", "guided", "none"):
        raise ValueError(f"ASG_CRF={_mode!r} is not one of dense|guided|none")
    if _mode == "none":
        if not _CRF_MARK[0]:
            _s.stderr.write("[crf] mode=NONE (raw prob, no refine)\n"); _CRF_MARK[0] = True
        return p
    if _mode == "guided":
        if not _CRF_MARK[0]:
            _s.stderr.write("[crf] mode=GUIDED-filter (forced)\n"); _CRF_MARK[0] = True
        return guided_upsample(image, p)
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except Exception as _e:
        raise RuntimeError(
            "ASG_CRF=dense (default) but pydensecrf is unavailable; refusing to silently "
            "substitute the guided filter (that undersells the dense-CRF baseline). Install "
            f"pydensecrf or set ASG_CRF=guided explicitly. ({_e})") from _e
    if not _CRF_MARK[0]:
        _s.stderr.write('[crf] using DENSE-CRF (pydensecrf) sxy=40 srgb=13\n'); _CRF_MARK[0] = True
    H, W = p.shape
    img = np.asarray(image)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    img = np.ascontiguousarray(img[..., :3].astype(np.uint8))
    eps = 1e-6
    probs = np.stack([1 - p, p], 0).clip(eps, 1 - eps)  # [2, H, W]
    d = dcrf.DenseCRF2D(W, H, 2)
    d.setUnaryEnergy(unary_from_softmax(probs))
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=40, srgb=13, rgbim=img, compat=10)
    Q = d.inference(n_iters)
    return np.asarray(Q, np.float32).reshape(2, H, W)[1]
