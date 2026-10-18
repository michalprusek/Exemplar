"""The public API.

    from exemplar import Exemplar

    model = Exemplar(device="cuda", cache_dir="/data/cache", metric="iou")
    model.fit(images, masks, rng=np.random.default_rng(0))
    model.add(image, mask)          # exact update: the same readout as a cold fit of all the masks
    fg = model.predict(image)       # bool mask at native resolution

`fit` decides the magnification of the fine view from the masks, reads the frozen inputs of every support image and
solves the readout (`readout.ridge_fit`). `add` re-runs the magnification rule from the kept per-mask measurements; if
the magnification is unchanged it adds only the new image's sums, otherwise every kept fine grid is stale and the
support is refitted cold (with the same column-shift sample), so the result always equals a cold fit. `predict` reads
the image's inputs at the support's magnification and cuts the blended score at 1/2.

`metric` is the held-out metric that chooses the blend weight: "iou" (foreground IoU) for objects, "cldice"
(centreline Dice) for thin structures such as vessels, membranes or filaments.

Ablations (Table 2 of the paper; the defaults are the reported method): `blocks` restricts the readout to a subset of
"coarse", "fine", "bank" (a subset without "fine" never computes the fine view); `zoom_fixed` sets the fine view's
magnification instead of deciding it from the masks ("fine view not magnified" is zoom_fixed=1); `layers=(-1,)` reads
only the last block of the backbone in both views ("last layer only", passed to FrozenInputs).

The `rng` passed to `fit` is drawn exactly once, for the 4096 pixels of the first support image that fix the column
shift. The paper's protocol passes the generator that drew the support indices (`benchmark/run.py`).

Numbers in parentheses are line numbers of scripts/closed_form_bench.py (AutoSeg).
Line references point to the authors' research code, which is not public.
"""
from __future__ import annotations

import numpy as np
import torch

from exemplar.encoder import LAYERS, CoarseView, Dinov3Encoder, check_layers, coarse_cache_dir
from exemplar.fineview import decide_zoom, fine_max_grid, fine_view
from exemplar.priors import make_prior_bank, priors
from exemplar.readout import BLOCKS, check_blocks, logit_map, ridge_fit

_METRICS = {"iou": "fg_iou", "fg_iou": "fg_iou", "cldice": "cldice"}


class FrozenInputs:
    """The frozen inputs of one image: the coarse view [1, 2048, 42, 42], the fine view at magnification z
    [1, 2048, hg, wg] and the priors [1, 35, H, W], all float32 on the device. Nothing here is fitted.
    (closed_form_bench.py:471-492, `blocks`)

    layers: the backbone blocks both views read, (-1, 12) (the reported method) or (-1,) (the ablation "last layer
    only": 1024 channels per view). fine=False: the fine view is not computed (None), for a readout without it."""

    def __init__(self, device, cache_dir: str | None = None, bf16: bool = True, encoder: Dinov3Encoder | None = None,
                 layers=LAYERS, fine: bool = True):
        self.device = device
        self.layers = check_layers(layers)
        self.encoder = encoder if encoder is not None else Dinov3Encoder(device, bf16=bf16, layers=self.layers)
        if tuple(getattr(self.encoder, "layers", LAYERS)) != self.layers:
            raise ValueError(f"the encoder reads layers {getattr(self.encoder, 'layers', LAYERS)}, not {self.layers}")
        self.coarse = CoarseView(self.encoder, coarse_cache_dir(cache_dir, bf16) if cache_dir else None)
        self.bank = make_prior_bank(device)
        self.fine = bool(fine)

    def __call__(self, image, zoom: int):
        image = np.asarray(image)
        x = np.asarray(self.coarse(image), np.float32).transpose(2, 0, 1)[None]
        xc = torch.from_numpy(x).to(self.device, torch.float32)
        bank = priors(image, self.bank, self.device).float()
        if not self.fine:                                # an ablation without the fine view: not computed (487)
            return xc, None, bank, tuple(bank.shape[-2:])
        fine = fine_view(self.encoder.model, image, zoom, fine_max_grid(zoom), self.device, layers=self.layers)
        return xc, fine.float(), bank, tuple(bank.shape[-2:])


class Exemplar:
    """In-context segmentation from a few annotated images: a closed-form linear readout of frozen inputs."""

    def __init__(self, device=None, cache_dir: str | None = None, metric: str = "iou", bf16: bool = True,
                 inputs=None, blocks=BLOCKS, zoom_fixed: int | None = None, layers=LAYERS):
        """device: "cuda", "cpu", ... (default: cuda when available). cache_dir: directory of the on-disk coarse-view
        cache (None: no cache). metric: the held-out metric of the blend, "iou" or "cldice". bf16: run the
        transformer under bfloat16 autocast, as the reported method does. inputs: a callable (image, zoom) ->
        (xc, fine, bank, hw) replacing FrozenInputs (shared between models, or synthetic in tests).
        Ablations (defaults = the reported method): blocks, a subset of ("coarse", "fine", "bank"); zoom_fixed, the
        fine view's magnification for every support set instead of the decided one (None or 0: decided); layers,
        (-1, 12) or (-1,) for FrozenInputs (ignored when `inputs` is given)."""
        if metric not in _METRICS:
            raise ValueError(f"metric must be 'iou' or 'cldice', got {metric!r}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.metric = _METRICS[metric]
        self.blocks = check_blocks(blocks)
        if zoom_fixed is not None and int(zoom_fixed) < 0:
            raise ValueError(f"zoom_fixed must be a positive magnification (or 0/None: decided), got {zoom_fixed}")
        self.zoom_fixed = int(zoom_fixed) if zoom_fixed else None
        self.inputs = inputs if inputs is not None else FrozenInputs(device, cache_dir, bf16, layers=layers,
                                                                     fine="fine" in self.blocks)
        self.counters = {}                               # float64 fallbacks and rejected Newton evaluations
        self.images, self.masks, self._measures = [], [], []
        self.zoom = None
        self.result = None
        self._state = None
        self._i0 = None

    # -- fitting -------------------------------------------------------------------------------------------------------
    def fit(self, images, masks, rng=None) -> "Exemplar":
        """Fit on support images and their masks (> 0 is foreground). rng: a numpy Generator (or a seed) drawn once
        for the column-shift sample."""
        images = [np.asarray(im) for im in images]
        masks = [np.asarray(m) for m in masks]
        if not images or len(images) != len(masks):
            raise ValueError("fit: one mask per image, at least one image")
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(0 if rng is None else rng)
        self.images, self.masks = images, masks
        self._measures = [{} for _ in masks]
        self.zoom = self._decide_zoom()
        self._solve(rng=rng)
        return self

    def _decide_zoom(self) -> int:
        """The fine view's magnification: decided from the masks, or the fixed one of the ablation (2419-2420)."""
        if self.zoom_fixed:
            return self.zoom_fixed
        return decide_zoom(self.masks, [max(im.shape[:2]) for im in self.images], self._measures)

    def add(self, image, mask) -> "Exemplar":
        """Add one annotated image. Exact: the result equals a cold fit of all the masks with the same column shift."""
        if self.result is None:
            raise RuntimeError("add: fit first")
        self.images.append(np.asarray(image))
        self.masks.append(np.asarray(mask))
        self._measures.append({})
        z = self._decide_zoom()
        if z != self.zoom or self._state is None or not self._state["incremental"]:
            self.zoom = z                                # stale fine grids (or a K = 1 fit): refit cold
            self._solve(shift_idx=self._i0)
        else:
            self._solve(state=self._state)
        return self

    def _inputs(self, image, zoom):
        """The frozen inputs, without the fine view when the readout does not use it."""
        xc, fine, bank, hw = self.inputs(image, zoom)
        return xc, (fine if "fine" in self.blocks else None), bank, hw

    def _solve(self, rng=None, shift_idx=None, state=None):
        z = self.zoom
        self.result, self._state = ridge_fit(lambda i: self._inputs(self.images[i], z), self.masks, self.metric,
                                             rng=rng, shift_idx=shift_idx, state=state, counters=self.counters,
                                             blocks=self.blocks)
        self._i0 = self._state["i0"]

    def release_support(self) -> None:
        """Free the kept support inputs (GPU memory); a later `add` then refits cold, with the same result."""
        self._state = None

    # -- the fitted readout --------------------------------------------------------------------------------------------
    @property
    def readout(self) -> dict:
        """The reported readout (blend weight a, lambdas, cut 1/2, weights w on [coarse | fine | priors | 1])."""
        if self.result is None:
            raise RuntimeError("not fitted")
        return self.result["blend1_ex"]

    @property
    def weights(self) -> np.ndarray:
        return self.readout["w"]

    @property
    def info(self) -> dict:
        r = self.readout
        return dict(k=len(self.masks), zoom=self.zoom, a=r["a"], variant=r["variant"], lam=r["lam"],
                    lam_bal=self.result["bal_mse"]["lam"], tau=r["tau"], heldout=r["heldout"],
                    gain=r.get("gain"), se=r.get("se"), **self.counters)

    # -- prediction ----------------------------------------------------------------------------------------------------
    def score(self, image) -> np.ndarray:
        """The score map s at native resolution (float32); the mask is s > 1/2."""
        if self.result is None:
            raise RuntimeError("not fitted")
        xc, fine, bank, hw = self._inputs(np.asarray(image), self.zoom)
        return logit_map(xc, fine, bank, hw, self.readout["w"])

    def predict(self, image) -> np.ndarray:
        """The foreground mask of an image: bool [H, W]."""
        image = np.asarray(image)
        lg = self.score(image)
        if lg.shape != image.shape[:2]:
            raise RuntimeError(f"score map {lg.shape} does not match the image {image.shape[:2]}")
        return lg > self.readout["tau"]
