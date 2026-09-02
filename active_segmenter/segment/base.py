"""SegmenterBackend interface — the one abstraction the AL harness swaps on ``--segmenter``.

Each backend maps ``(image, frozen feat_grid, labeled support)`` to a native-resolution
foreground mask (fg-IoU continuity with prior findings) and to independent per-instance
masks (instance metrics). The overlap-safe invariant is interface-level: ``predict``
returns a list of independent boolean masks, never a shared label raster.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from skimage.transform import resize

from active_segmenter.types import InstanceMask


class BackendUnavailable(RuntimeError):
    """Raised when a backend's runtime (e.g. the isolated SAM 3 env) is not reachable."""


@dataclass
class LabeledExample:
    image: np.ndarray       # [H, W] or [H, W, C] native image
    feat_grid: np.ndarray   # [G, G, D] frozen DINOv3 features
    label_map: np.ndarray   # [H, W] int: {0,1} for single-class fg, or per-instance ids


def foreground_from_score(score_grid: np.ndarray, hw, thresh: float = 0.0) -> np.ndarray:
    """Upsample a ``[G, G]`` score grid to a native bool mask using the fg-IoU convention
    (order=0, edge, ``> 0.5`` after resizing the ``{0,1}`` thresholded grid) so backends
    match the quantity the current AL testbed measures."""
    fg_grid = (np.asarray(score_grid, np.float32) > thresh).astype(np.float32)
    up = resize(fg_grid, tuple(hw), order=0, mode="edge", anti_aliasing=False)
    return up > 0.5


def reset_backend_for_new_support(be) -> None:
    """Prepare a REUSED backend to be re-fitted on an unrelated support draw.

    Benchmarks construct the backend once and loop over seeds, so anything a backend derived from the
    previous draw must go — otherwise later seeds silently inherit the first draw's configuration and
    the reported across-seed variance is not the variance of the whole method. Backends that
    self-configure expose ``reset_support_state``; the rest only carry a trained head. Wrapper backends
    (refine / crop) delegate to the backend they wrap.

    Do NOT call this in an active-learning loop that grows a single support set — there the carry-over
    is the intended behaviour.

    A backend whose ``fit`` REBUILDS every piece of support state from scratch has nothing to reset and
    declares ``stateless_support = True``; that opt-out must be a deliberate statement about ``fit``,
    never a way to quiet the error below.
    """
    inner = getattr(be, "base", None) or getattr(be, "inner", None)   # crop uses .base, refine uses .inner
    if inner is not None:
        reset_backend_for_new_support(inner)
    if hasattr(be, "reset_support_state"):
        be.reset_support_state()
    elif hasattr(be, "head"):
        be.head = None
    elif inner is None and not getattr(be, "stateless_support", False):
        # No reset hook, no head, no recognised wrapped backend, no stateless declaration: this call did
        # NOTHING. Staying silent here is how seed-0 state survives a whole run, so refuse instead of
        # pretending.
        raise TypeError(f"reset_backend_for_new_support: {type(be).__name__} exposes neither "
                        f"reset_support_state, head, nor a wrapped backend (.base/.inner), and does "
                        f"not declare stateless_support — support state cannot be cleared between seeds")


class SegmenterBackend(Protocol):
    def fit(self, support: list[LabeledExample]) -> None: ...
    def score_map(self, image, feat_grid, class_id: int = 1) -> np.ndarray: ...
    def foreground(self, image, feat_grid, class_id: int = 1) -> np.ndarray: ...
    def predict(self, image, feat_grid, class_id: int = 1) -> list[InstanceMask]: ...
