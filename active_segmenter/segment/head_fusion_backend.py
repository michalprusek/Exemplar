"""Trainable-head backend WITH classical per-pixel priors fused at native resolution.

Same pipeline role as :class:`TrainableHeadBackend` (few-shot fit per round, EGL/BADGE acquisition
via ``grad_embedding``), but the head is :class:`DINOHeadFusion`: the decision is made at native
resolution over [upsampled DINOv3 semantics ⊕ classical native features]. Classical features (a
subset of the vendored HyperBank bank, frozen unless ``trainable_classical``) are computed once per
image and cached ON THE CAPPED TRAINING PATH (the native-tiled inference bank, used when
``thin_adaptive`` fires, is recomputed per call — not memoised), so training only runs the light
head. Targets the small-blob ceiling of the pure-grid head while keeping the whole active-learning +
few-shot machinery.
"""
from __future__ import annotations

import os

import numpy as np

from active_segmenter.propose import instances as inst
from active_segmenter.segment.base import LabeledExample, foreground_from_score
from active_segmenter.types import InstanceMask


def _dice(logits, target):
    p = logits.sigmoid()
    return 1 - (2 * (p * target).sum() + 1) / (p.sum() + target.sum() + 1)


def _crop_or_pad(a, shape):
    """Centre-crop or reflect-pad `a` back to `shape` after a zoom, so every augmented copy keeps the
    original geometry and the head never sees two different output sizes for one support image."""
    import numpy as np
    out = a
    for ax in (0, 1):
        n, m = out.shape[ax], shape[ax]
        if n > m:
            lo = (n - m) // 2
            out = out.take(range(lo, lo + m), axis=ax)
        elif n < m:
            pad = [(0, 0)] * out.ndim
            lo = (m - n) // 2
            pad[ax] = (lo, m - n - lo)
            out = np.pad(out, pad, mode="reflect")
    return out


def _fit_morphology(pred_masks, gt_masks, metric="fg_iou", max_r=3):
    """LEARN the opening/closing kernel by scoring every candidate against the target metric.

    WHY LEARNED AND NOT DERIVED. A first version read the radii off support-mask statistics -- the
    thinnest structure bounds the opening, the closest inter-object gap bounds the closing. Those are
    bounds on what is SAFE, not evidence about what HELPS, and the difference showed up immediately:
    on fat nuclei the derivation licensed radius 7, which removes nothing on the support and real
    structure on the next image. The bound had to be replaced by an arbitrary cap, and an arbitrary
    cap is a tuned constant wearing a derivation's clothes.

    So the kernel is selected the way the threshold sweep in C59 selected a threshold: enumerate the
    candidates, score each on the SUPPORT against the metric the dataset is actually scored on, keep
    the best. Three discrete parameters over a 4x4x2 grid, chosen by the same quantity the paper
    reports, rather than three continuous parameters chosen by a proxy.

    WHY IT MIGHT WORK WHERE THE THRESHOLD DID NOT. C60 measured that sweep as worth +0.003 and -0.002
    on the centreline datasets, because a threshold moves every pixel the same way: it buys overlap
    and cannot buy topology. Opening and closing are topological. A closing of radius one rejoins a
    centreline broken by a single pixel -- the error clDice punishes and Dice barely notices -- so
    this is the same selection mechanism pointed at the axis where the previous one had no purchase.

    WHAT THIS SELECTION IS AND IS NOT. The predictions are the fitted head's on its OWN support, so
    this is a training-set selection. That is acceptable here and would not be for the head itself:
    three discrete parameters cannot memorise eight images the way half a million weights could, and
    the errors being fixed -- speckle, one-pixel breaks -- are systematic rather than per-image. It
    is stated rather than hidden, and the identity candidate (0, 0, off) is always in the grid, so
    the rule can decline to act.
    """
    from active_segmenter.eval.metrics import foreground_iou, cldice

    score = cldice if metric == "cldice" else foreground_iou
    best, best_s = {"r_open": 0, "r_close": 0, "largest_only": False}, -1.0
    for r_o in range(0, max_r + 1):
        for r_c in range(0, max_r + 1):
            for lo in (False, True):
                rule = {"r_open": r_o, "r_close": r_c, "largest_only": lo}
                vals = []
                for pm, gm in zip(pred_masks, gt_masks):
                    vals.append(float(score(_apply_morphology(pm, rule), gm)))
                sc = float(np.mean(vals)) if vals else -1.0
                # Strict improvement to move off the identity: a tie should not buy an operation.
                if sc > best_s + 1e-9:
                    best, best_s = rule, sc
    best["selected_score"] = round(best_s, 4)
    return best


def _apply_morphology(mask, rule):
    """Opening, then closing, then optionally the largest component. Order matters: opening first so
    a closing cannot re-seal speckle that the opening was about to remove."""
    from scipy import ndimage as ndi

    m = np.asarray(mask, bool)
    if rule.get("r_open", 0) > 0:
        m = ndi.binary_opening(m, ndi.generate_binary_structure(2, 1),
                               iterations=int(rule["r_open"]))
    if rule.get("r_close", 0) > 0:
        m = ndi.binary_closing(m, ndi.generate_binary_structure(2, 1),
                               iterations=int(rule["r_close"]))
    if rule.get("largest_only") and m.any():
        lab, n = ndi.label(m)
        if n > 1:
            sizes = ndi.sum(m, lab, range(1, n + 1))
            m = lab == (int(np.argmax(sizes)) + 1)
    return m


def _bridge_weight(gt, band: int = 12):
    """Per-pixel weight, computed from the GROUND TRUTH alone, that is high where a false positive
    would MERGE two distinct objects and ~0 elsewhere.

    WHY THIS TERM EXISTS. clDice's skeleton-PRECISION factor needs the skeleton of the PREDICTION,
    which is an iterated min/max pooling and therefore not convex and not per-pixel -- it cannot
    survive a single closed-form solve. Its job, though, is to punish predicted structure that is not
    in the ground truth, and the most expensive instance of that is bridging: `foreground-is-the-
    bottleneck` measured that at equal foreground IoU, error that MERGES two objects costs 2-3x the
    AP of the same amount of error placed apart. So instead of penalising all spurious structure
    indirectly, this penalises the specific spurious structure that costs the most, directly.

    WHY IT IS CLOSED-FORM COMPATIBLE. The map depends only on `gt`, so the loss term is
    `<prediction, W>` -- LINEAR in the prediction. It can be added to a convex objective and solved
    once, unlike the factor it stands in for. It is not clDice's Tprec and this docstring should not
    be read as claiming it is; it is a stand-in for Tprec's ROLE, and whether that is enough is a
    question for a screen, not for an argument.

    Construction: label the foreground, take each background pixel's NEAREST component, and mark the
    pixels where a 3x3 neighbourhood sees more than one distinct nearest component. Those are the
    loci equidistant from two objects -- the skeleton by influence zones -- which is exactly where a
    merge happens. Restricted to a band of `band` pixels so that far-away background, where no
    plausible prediction could bridge anything, carries no weight.
    """
    from scipy import ndimage as ndi

    g = np.asarray(gt)
    fg = g > 0
    lab, n = (g.astype(np.int32), int(g.max())) if g.max() > 1 else ndi.label(fg)
    if n < 2:
        return np.zeros(fg.shape, np.float32)          # one object: nothing to bridge
    dist, (iy, ix) = ndi.distance_transform_edt(~fg, return_indices=True)
    nearest = lab[iy, ix]                              # each background pixel's closest object
    big = n + 1
    masked = np.where(fg, big, nearest)                # objects themselves must not create a line
    hi = ndi.maximum_filter(np.where(fg, 0, nearest), size=3)
    lo = ndi.minimum_filter(masked, size=3)
    lines = (~fg) & (hi > 0) & (lo < big) & (hi != lo)
    w = (lines & (dist <= band)).astype(np.float32)
    return ndi.uniform_filter(w, size=3) * 9.0 / 9.0   # 1px feather; keeps the band contiguous


def _skeleton_recall(logits, target_skel, eps: float = 1e-6):
    """Skeleton recall as a CONVEX-IN-THE-WEIGHTS term: `<S, softplus(-logits)> / sum(S)`.

    TAKES LOGITS, NOT PROBABILITIES, AND THAT IS THE WHOLE POINT. The previous form was
    `1 - <S, sigmoid(z)>/sum(S)` -- linear in the PREDICTION, which is what the surrounding
    comments meant by "convex", but the solve is over the WEIGHTS `w`, and `sigmoid` is neither
    convex nor concave (`d^2s/dz^2 = s(1-s)(1-2s)` changes sign at 0). Composed with a linear
    readout `z = Xw` the old term was therefore NOT convex in `w`, so the "one-solve objective"
    it belonged to did not actually admit one solve. Only `_lovasz_hinge` did, because it is the
    one term that already took `fgl` rather than `fgp`.

    WHY THIS FORM IS THE RIGHT FIX AND NOT A DIFFERENT LOSS. Since `-log t >= 1 - t` for t in
    (0,1], and `-log sigmoid(z) = softplus(-z)`, this term is a convex UPPER BOUND on the
    original `mean_S(1 - p)`. Minimising it minimises a majorant of the quantity that was
    intended -- the standard surrogate argument -- while being convex-nonincreasing composed
    with a linear function, hence convex in `w`. It is exactly clCE's `CE-T_prec` (Acebes et
    al., MICCAI 2024) and Skea-Topo's `Skeaw` (Liu et al., IJCAI 2024), i.e. the published
    member of the only convex family: a fixed nonnegative GT-derived weight on a per-pixel
    convex loss.

    SCALE CHANGED, AND IT NEEDS WATCHING. The old form was bounded in [0,1]; `softplus(-z)` is
    not -- at a confidently-negative skeleton pixel (z ~ -20) it contributes ~20, against a
    Lovasz hinge bounded near 1. With `wts["cldice"] ~ 0.5` on drive/hrf the thin term can
    therefore dominate the objective by an order of magnitude at initialisation, so a screen of
    this arm could easily measure a weight mis-scaling rather than the objective. Whoever runs
    it should log the per-term magnitudes on the first epoch rather than assume the balance.

    NO REPORTED NUMBER CAME THROUGH THIS PATH, but not for the reason an earlier draft of this
    docstring gave. `al_testbed.py`'s token parser indeed has no `convex`/`lovasz` token, but
    `scripts/screen_flag.py` reaches BOTH by `setattr(be, args.flag, on)` and already stages a
    `convex_loss` screen (targets drive/hrf, controls dsb2018/spheroidj). The path is reachable;
    it is simply that the registry holds no `convex_loss` result row yet.

    STILL DEGENERATE ALONE: a prediction that fattens everything has perfect skeleton recall.
    It is only meaningful beside a term that punishes over-prediction, which is why the convex
    objective pairs it with the Lovasz hinge rather than using it as the region anchor.
    """
    import math

    import torch.nn.functional as F
    s = target_skel.reshape(-1).float()
    z = logits.reshape(-1).float()
    # `_SP0` makes this term equal the OLD bounded one AT INITIALISATION: at z=0 the old form gave
    # `1 - sigmoid(0) = 0.5` and softplus(0) = ln2, so dividing by 2*ln2 reproduces 0.5 there. That
    # keeps `wts["cldice"]`'s calibration meaningful at the point it was chosen. It does NOT bound
    # the term -- a convex majorant of a bounded function cannot be bounded -- so the tail warning
    # above still stands; this only removes the gratuitous 1.4x offset at the calibration point.
    return (s * F.softplus(-z)).sum() / (s.sum() + eps) / (2.0 * math.log(2.0))


def _lovasz_grad(gt_sorted):
    """Gradient of the Lovasz extension of the Jaccard loss, w.r.t. the sorted errors."""
    import torch
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-8)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1].clone()
    return jaccard


def _lovasz_hinge(logits, target, max_px: int = 1_000_000):
    """Lovasz-hinge: the tight convex surrogate for 1 - IoU (Berman et al., CVPR 2018).

    WHY THIS TERM AND NOT ANOTHER REGION LOSS. Registry C58 measured that the representation is not
    the bottleneck -- an unsupervised PCA to 32 dims keeps 98-99.9% of the raw per-pixel AUC, and a
    SUPERVISED projection adds only 0.001-0.036 against a 0.103-0.223 gap. What the readout does
    badly is convert a good ranking into a good OVERLAP, and the variable that separates the datasets
    is foreground prevalence (1-9% on vessels against tens of percent on nuclei), which is exactly
    where the probabilistic optimum and the overlap optimum diverge. Lovasz optimises the overlap
    directly rather than through a per-pixel proxy, and it stays convex composed with a linear
    readout -- which is why C58 named it as the redirect.

    WHAT IT CANNOT FIX, stated up front. It is still a per-pixel rule, and clDice's argument (Shit et
    al., CVPR 2021) is that one missing pixel on a thin vessel costs ~0 overlap while severing a
    component. So this should move the seven overlap-scored datasets and do nothing for the three
    scored by centreline agreement -- which is the split we already measure against nnU-Net (+0.056
    overlap, -0.051 centreline). A screen that shows it helping the topology sets would be evidence
    the measurement is wrong, not that the loss is better than its authors claim.

    The sort is O(n log n) and the whole loss is 0.2% of a fit's device time, so cost is not the
    constraint; `max_px` only bounds the sort's memory on the largest fields, drawn uniformly so the
    Jaccard estimate stays unbiased.
    """
    import torch
    import torch.nn.functional as F

    lg = logits.reshape(-1).float()
    tg = target.reshape(-1).float()
    if lg.numel() > max_px:
        idx = torch.randperm(lg.numel(), device=lg.device)[:max_px]
        lg, tg = lg[idx], tg[idx]
    if tg.numel() == 0:
        return lg.sum() * 0.0
    signs = 2.0 * tg - 1.0
    errors = 1.0 - lg * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    grad = _lovasz_grad(tg[perm])
    return torch.dot(F.relu(errors_sorted), grad)


def _dice_bce(logits, target):
    import torch.nn.functional as F
    return F.binary_cross_entropy_with_logits(logits, target) + _dice(logits, target)


# ─── adaptive-loss MENU (each term ~unit-magnitude so the morphology weights are the sole control) ───
def _focal_bce(logits, target, gamma: float = 2.0):
    """Focal loss — down-weights easy pixels, engages on class imbalance (small/faint foreground)."""
    import torch
    import torch.nn.functional as F
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    return (torch.clamp(1 - pt, 0, 1).pow(gamma) * ce).mean()


def _tversky(logits, target, alpha: float = 0.3, beta: float = 0.7):
    """Tversky (β>α) — asymmetric FP/FN, penalises false-negatives harder → recovers thin/small structures."""
    p = logits.sigmoid()
    tp = (p * target).sum()
    fp = (p * (1 - target)).sum()
    fn = ((1 - p) * target).sum()
    return 1 - (tp + 1) / (tp + alpha * fp + beta * fn + 1)


def _signed_dt(label_map) -> np.ndarray:
    """Signed distance transform of a GT mask (negative inside fg, positive outside), normalised by the
    image diagonal → ~unit magnitude. Level-set field for the boundary loss (Kervadec et al.)."""
    from scipy import ndimage
    m = np.asarray(label_map) > 0
    if not m.any() or m.all():
        return np.zeros(m.shape, np.float32)
    din = ndimage.distance_transform_edt(m)
    dout = ndimage.distance_transform_edt(~m)
    diag = float(np.hypot(*m.shape)) + 1e-6
    return ((dout - din) / diag).astype(np.float32)


def _boundary_ls(prob, dt_signed):
    """Boundary (distance) loss — integral of the foreground PROBABILITY × signed-DT level set; pulls the
    contour to the GT boundary. Must be paired with a region loss (collapses to empty fg alone — Kervadec)."""
    return (prob * dt_signed).mean()


def _boundary_dou(logits, target):
    """Boundary Difference-over-Union loss (Sun et al., MICCAI 2023; ref github sunfan-bvb/BoundaryDoULoss),
    binary adaptation. A region-based boundary loss, self-scaling by target size, with NO auxiliary loss: it
    DOWN-WEIGHTS the interior intersection (via ``alpha``) so the gradient concentrates on the fg CONTOUR and
    on over-prediction — the monuseg fg-bleed the diagnostic pinned down (precision 0.68 vs recall 0.89).
    ``alpha = 1 − 2C/S`` (C = #boundary fg pixels via a 3×3 cross-conv, S = fg area), clamped ≤ 0.8: small
    objects → small/negative alpha → MORE boundary weight, large objects → 0.8. Loss =
    ``(z²+y²−2·inter)/(z²+y²−(1+alpha)·inter)`` with soft p (sigmoid). alpha is a shape descriptor (no grad)."""
    import torch
    import torch.nn.functional as F
    p = logits.sigmoid()
    t = target.to(p.dtype)
    eps = 1e-5
    with torch.no_grad():                                      # alpha is a size descriptor, not a grad path
        kernel = torch.tensor([[0., 1, 0], [1, 1, 1], [0, 1, 0]], device=t.device,
                              dtype=t.dtype).view(1, 1, 3, 3)
        conv = F.conv2d(t, kernel, padding=1) * t              # cross-neighbour count, on fg pixels only
        C = ((conv > 0) & (conv != 5)).sum().to(p.dtype)       # fg pixels NOT fully surrounded = boundary
        S = t.sum().clamp(min=1.0)
        alpha = torch.clamp(1.0 - 2.0 * (C + eps) / (S + eps), max=0.8)
    inter = (p * t).sum()
    y2 = (t * t).sum()
    z2 = (p * p).sum()
    return (z2 + y2 - 2 * inter + eps) / (z2 + y2 - (1.0 + alpha) * inter + eps)


def _mc_ce_dice(logits_c, target_idx):
    """MULTI-CLASS region anchor for >1 semantic class: cross-entropy + generalised (soft) multi-class Dice
    over the C channels. `logits_c`=[1,C,H,W], `target_idx`=[1,H,W] long class ids (0=bg). Replaces the
    binary Dice+BCE when the annotations carry multiple classes; morphology terms still act on the fg union."""
    import torch
    import torch.nn.functional as F
    C = logits_c.shape[1]
    ce = F.cross_entropy(logits_c, target_idx)
    p = logits_c.softmax(1)
    oh = F.one_hot(target_idx.clamp(0, C - 1), C).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3))
    dice = 1 - ((2 * inter + 1) / (p.sum((0, 2, 3)) + oh.sum((0, 2, 3)) + 1)).mean()
    return ce + dice


def _gray01(image) -> np.ndarray:
    a = np.asarray(image).astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    return (a - a.min()) / (np.ptp(a) + 1e-6)


def _color_channels(image, force: bool = False) -> dict:
    """Candidate single-channel views for support-driven contrast selection, each min-max normalised to
    [0, 1]: always ``gray``; for COLOUR images also per-channel ``R``/``G``/``B`` and H&E stain-deconvolution
    ``hematoxylin``/``eosin``.

    ``force=False`` (DETECTION mode) applies a per-image mono gate: an RGB image whose channels are ~equal
    returns only ``{'gray'}`` — used once to decide whether the DATASET is a colour modality.
    ``force=True`` (RETRIEVAL mode, used once the dataset is deemed colour) computes the full candidate set
    for ANY 3-channel image, so every image in a colour dataset exposes the SAME candidates (a near-gray tile
    just yields R≈G≈B≈gray) — the inter-channel-variation mono gate cannot fire per-image and skew a
    homogeneous-colour run. A genuinely 2-D (single-channel) image always returns only ``{'gray'}`` regardless
    of ``force``; selection requires a channel on EVERY support image, so a 2-D image in the support falls the
    dataset back to gray, while a 2-D image appearing only at inference (after a colour source was chosen)
    raises in ``_channel`` — a loud train/inference-inconsistency signal, never a silent gray swap."""
    a = np.asarray(image).astype(np.float32)
    out = {"gray": _gray01(image)}
    if a.ndim != 3 or a.shape[2] < 3:
        return out
    rgb = a[..., :3]
    if not force:                                                # detection: per-image mono gate
        chan_spread = float(np.abs(rgb - rgb.mean(axis=2, keepdims=True)).mean())
        if chan_spread / (float(np.ptp(rgb)) + 1e-6) < 0.01:     # <1% inter-channel variation → grayscale
            return out

    def n01(c):
        return ((c - c.min()) / (np.ptp(c) + 1e-6)).astype(np.float32)

    for i, name in enumerate(("R", "G", "B")):
        out[name] = n01(rgb[..., i])
    try:                                                          # H&E deconvolution (Ruifrok-Johnston)
        from skimage.color import rgb2hed
        hed = rgb2hed(rgb / (float(rgb.max()) + 1e-6))
        out["hematoxylin"] = n01(hed[..., 0])
        out["eosin"] = n01(hed[..., 1])
    except Exception:                                             # H&E candidates are optional extras;
        pass                                                      # gray/R/G/B always cover the gate
    return out


def _label(fg):
    from skimage.measure import label

    return label(np.asarray(fg))


def _crop_frac(t, a: float, b: float):
    """Crop the LAST dim of a tensor to the fractional range [a, b). None passes through.

    Fractions, not pixels, because a support item carries tensors at two different scales: the DINOv3
    grids are at patch resolution (42 wide) while the classical bank and the label are native (up to
    1536). Cropping both to the same fraction keeps them registered through the head's upsample; a
    shared pixel count would not. Guarantees at least one column so a thin strip cannot produce an
    empty tensor.
    """
    if t is None:
        return None
    w = t.shape[-1]
    i, j = int(round(a * w)), int(round(b * w))
    j = max(j, i + 1)
    return t[..., i:j]


def _split_item(it, a: float, b: float):
    """Crop one training item to the width fraction [a, b), leaving every non-spatial field alone."""
    X, C, Y, B, hw, Xf, tcl, aux, Dg = it
    aux2 = aux
    if aux is not None:
        wts, dt, yc = aux
        aux2 = (wts, _crop_frac(dt, a, b), _crop_frac(yc, a, b))
    Wn = _crop_frac(Y, a, b).shape[-1]
    return (_crop_frac(X, a, b), _crop_frac(C, a, b), _crop_frac(Y, a, b), _crop_frac(B, a, b),
            (hw[0], Wn), _crop_frac(Xf, a, b), tcl, aux2, _crop_frac(Dg, a, b))


def _boundary_target(label_map):
    """Object outline map (W2 target): ``find_boundaries(mode="inner")``, dilated by one iteration.

    Read the mode, not the intent: ``mode="inner"`` marks the inner rim of EVERY instance, against
    background as well as against a neighbour -- verified on a synthetic label map where an isolated
    square still receives a full 12-pixel outline. So this is not only the contact faces between
    touching instances, which is what the W2 lever was FOR; it is every object's outline, and the
    contact faces are the subset that matters for splitting. Binary GT is first connected-component
    labelled, so the head is live there too; it goes inactive only when that yields one component."""
    from scipy import ndimage
    from skimage.segmentation import find_boundaries

    lab = np.asarray(label_map)
    if int(lab.max()) <= 1:  # binary fg → derive touching-blob boundaries from connected comps
        lab = _label(lab > 0)
    if int(lab.max()) <= 1:
        return np.zeros(lab.shape, np.float32)
    b = find_boundaries(lab, mode="inner", background=0)
    b = ndimage.binary_dilation(b, iterations=1)
    return b.astype(np.float32)


# silhouette_score builds the FULL pairwise distance matrix, so it is O(n^2) in the number of radii.
_SCALE_SUBSAMPLE = 4000


def cluster_scales(radii, kmax: int = 4, sil_floor: float = 0.55):
    """Cluster 1-D object radii (px) in LOG space; return the sorted cluster centroids (px). The number of
    clusters -- hence the number of bank scales the scaleconf lever builds -- self-configures: it defaults
    to ONE scale (the geometric mean) and only splits into k>=2 when a k-means over log-radii is clearly
    separated (silhouette > ``sil_floor``). So a single-thickness morphology yields one scale and a genuinely
    multi-scale one yields several. ONE definition (module scope) so the lever and any probe agree.

    Two properties a reader should not have to infer:
      * the radii are SUBSAMPLED (deterministically) before clustering -- see ``_SCALE_SUBSAMPLE``;
      * the k-means path is the only one that honours ``sil_floor``. If sklearn is not installed the
        quantile fallback returns up to three scales with NO separability test at all.
    """
    # KEEP sub-pixel radii: the caller clamps centroids up to >=1.5, so filtering them out (an earlier
    # `x >= 1.0`) silently collapsed a thin-filament support to ONE COARSE scale instead of a fine one.
    r = np.asarray([float(x) for x in radii if x >= 0.3], np.float32)
    if r.size == 0:
        return [1.0]
    # `_support_scales` pools EVERY medial-axis pixel of all K masks -- ~10^5 radii on a vessel or
    # filament set. Handing that to an O(n^2) silhouette is the same native-resolution latency trap that
    # got the bank-unfreeze lever dropped (CLAUDE.md C13, "hrf hung >1h"), and thin vessels are exactly
    # this lever's target. A few thousand samples already fix the cluster structure of a 1-D
    # distribution; the fixed seed keeps the derived scales reproducible across runs.
    if r.size > _SCALE_SUBSAMPLE:
        r = np.random.default_rng(0).choice(r, _SCALE_SUBSAMPLE, replace=False)
    lr = np.log(r).reshape(-1, 1)
    best = [float(np.exp(lr.mean()))]                 # default: a single scale = geometric-mean radius
    if np.unique(np.round(r, 1)).size >= 2:
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except ImportError:
            # ONLY a genuinely absent sklearn falls back. Anything the clustering itself can raise
            # (convergence, memory, degenerate labels) now propagates: swallowing it would silently swap
            # the scale-selection ALGORITHM per draw, so one arm's mean would average seeds run under two
            # different rules -- and the old blanket `except Exception` blamed every such failure on
            # "sklearn unavailable", which is a diagnostic that sends the reader the wrong way.
            print("    [scaleconf] sklearn not installed → quantile-scale fallback "
                  "(NO separability gate on this path)", flush=True)
            best = [float(q) for q in np.quantile(r, [0.25, 0.5, 0.75]) if q >= 1.0] or best
        else:
            best_s = sil_floor                        # require clear separation to ADD scales
            for k in range(2, min(kmax, r.size - 1) + 1):
                km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(lr)
                if len(set(km.labels_)) < k:
                    continue
                s = float(silhouette_score(lr, km.labels_))
                if s > best_s:
                    best_s = s
                    best = [float(np.exp(c[0])) for c in km.cluster_centers_]
    # Merge centroids closer than 1.4x: 1.4 ~= sqrt(2) is half an octave of scale space, below which two
    # Frangi/LoG responses overlap so heavily they are not separable filters -- so a pair that close is
    # one scale split by cluster noise, not two DISTINCT scales.
    out = [sorted(best)[0]]
    for c in sorted(best)[1:]:
        if c / out[-1] < 1.4:
            out[-1] = float(np.sqrt(out[-1] * c))     # collapse to the geometric mean
        else:
            out.append(c)
    return sorted(round(c, 1) for c in out)


def channel_separability(support):
    """Per-support-image fg/local-bg Fisher separability of every candidate colour channel.

    Returns ``(scores, n_used)`` where ``scores[channel]`` is the PER-IMAGE list, in support order,
    and ``n_used`` counts the images that contributed. The per-image list matters: the decision rule
    currently reduces it to a median and compares against a fixed ``color_margin``, but whether that
    margin can be replaced by the sample's own noise is a question only answerable from the
    individual values. Extracted to module scope so ``scripts/gate_constants_probe.py`` can measure
    the rule the segmenter actually runs rather than re-deriving the separability by hand.

    The background is a DILATED RING around each fg structure, not the global background: the
    classical bank is a stack of LOCAL operators (Frangi ridge / Sauvola adaptive-threshold), so the
    channel that matters is the one with the best LOCAL fg-vs-surround contrast. A global background
    lets a far-field artefact dominate -- DRIVE's black FOV border made the bright red channel win
    spuriously over the textbook green vessel channel.
    """
    from scipy.ndimage import binary_dilation

    scores: dict = {}
    n_used = 0
    for ex in support:
        m = np.asarray(ex.label_map) > 0
        if not m.any() or m.all():
            continue
        n_used += 1
        ring = binary_dilation(m, iterations=8) & ~m           # local surround (~8 px), not global bg
        if ring.sum() < 10:                                    # fg fills the frame → fall back to global
            ring = ~m
        for name, c in _color_channels(ex.image, force=True).items():
            if c.shape != m.shape:                              # defensive: sizes should already match
                continue
            fg, bg = c[m], c[ring]
            # Variance floor. Every candidate arrives min-max normalised to [0,1] from
            # ``_color_channels.n01``, so ``c.max() - c.min()`` is ~1 and this term is in practice the
            # CONSTANT 1e-3 rather than anything channel-relative; the expression is kept in that form
            # only so it still degrades sensibly if an un-normalised channel is ever added.
            #
            # What it buys is a BOUND, not a reordering. Since the numerator cannot exceed the
            # normalised range, the ratio is capped near 1000, so a channel with almost no within-class
            # variance cannot return an astronomically confident score off a rounding-level difference
            # and hijack the selection. It does NOT make a near-flat channel score ~0: in the noiseless
            # limit every channel whose foreground and background sit at the two extremes ties at the
            # cap. That case does not arise in real data, where the variance term dominates and the
            # floor never binds, but it means this floor must not be relied on to rank channels.
            denom = max(float(np.sqrt(fg.var() + bg.var())),
                        1e-3 * (float(c.max()) - float(c.min())), 1e-6)
            scores.setdefault(name, []).append(abs(float(fg.mean()) - float(bg.mean())) / denom)
    return scores, n_used


def colour_gate(scores, n_used, margin):
    """Which channel should feed the classical bank? Returns ``(choice, ranked_medians)``.

    The ONE definition of the decision, for the same reason ``thin_gate`` is:
    ``scripts/gate_constants_probe.py`` is the evidence behind the paper's "no per-dataset tuning"
    claim, and it re-implemented this by hand. Only the MEASUREMENT
    (``channel_separability``) had been shared, so the colour probe still carried exactly the
    drift risk the size probe was fixed to remove.

    A channel is eligible only if it was scored on EVERY used support image, else its average is
    over a biased subset (rgb2hed silently failing on the hard tiles would inflate hematoxylin
    against gray). The MEDIAN is used so one degenerate image cannot dominate. ``gray`` wins ties
    and wins by default, so a monochrome modality is unaffected.

    ``margin`` is the fitted 1.05 constant. Replacing it with a noise-based criterion is measured,
    not yet adopted, because it cannot be a no-op at K=1 and the reported campaign includes K=1.
    """
    means = {k: float(np.median(v)) for k, v in scores.items() if len(v) == n_used}
    if not means:
        return "gray", means
    gray = means.get("gray", 0.0)
    best = max(means, key=means.get)
    choice = best if (best != "gray" and means[best] > gray * margin) else "gray"
    return choice, means


def thin_gate(mean_tubularity: float, mean_side: float, thin_ref: float, train_cap: int) -> bool:
    """Should the native-resolution classical bank be switched on for this support set?

    Lives at module scope, and is the ONE definition of the rule, because
    ``scripts/gate_constants_probe.py`` is the evidence behind the paper's "no per-dataset tuning"
    claim and used to re-implement this expression by hand. Two copies of a decision rule is two
    chances to drift, and drift here means the probe certifies a rule the segmenter does not run.

    ``train_cap`` is the head's own feature-resolution cap, NOT a tuned threshold: at or below it
    the native bank and the capped-trained head see the same image, so the distribution shift the
    size test exists to avoid is zero by construction.
    """
    return (mean_tubularity > thin_ref) and (mean_side <= train_cap)


def _tubularity(label_map, max_side: int = 512) -> float:
    """Per-image morphology descriptor ∈ [0,1] = 1 − SOLIDITY (area ⁄ convex-hull area), averaged
    PER CONNECTED COMPONENT (area-weighted), from the ANNOTATED shape. Drives the adaptive clDice weight:
    ~0 for COMPACT objects (blobs, round polyps, or scattered round nuclei — each fills its own hull →
    solidity≈1), ~1 for branching/sparse structures (a vessel tree occupies a small fraction of its hull).
    PER-COMPONENT (not global) is essential: global convex-hull over SCATTERED compact instances (dsb
    nuclei) is huge → low global solidity → would wrongly switch clDice ON and roughen the nuclei (the
    dsb −0.032 regression); per-component correctly sees each nucleus as round → clDice off. Chosen over
    skeleton-to-area, which conflates 'thin' with 'small/jagged' (ranked compact polyps ABOVE vessels).
    Computed on the GT (no grad); solidity is ~scale-invariant so the mask is downscaled. Edge case: a
    single STRAIGHT rod fills its hull → gated off; our thin data (hrf) is branching so this doesn't bite."""
    from skimage.measure import label as sklabel
    from skimage.morphology import convex_hull_image

    m = np.asarray(label_map)
    binary = m > 0
    if binary.sum() < 4:
        return 0.0
    if max(binary.shape) > max_side:                     # scale-invariant → downscale for a cheap hull
        from skimage.transform import resize
        s = max_side / max(binary.shape)
        hh, ww = max(1, int(binary.shape[0] * s)), max(1, int(binary.shape[1] * s))
        # preserve instance ids through the resize (nearest) so components stay separated
        m = np.round(resize(m.astype(np.float32), (hh, ww), order=0)).astype(np.int64)
        binary = m > 0
    comps = m if m.max() > 1 else sklabel(binary)        # use GT instance ids if present, else CC
    num = int(comps.max())
    if num < 1:
        return 0.0
    tot_w, acc = 0.0, 0.0
    for i in range(1, num + 1):
        c = comps == i
        a = float(c.sum())
        if a < 4:
            continue
        ys, xs = np.where(c)                             # CROP to the component's bbox before the hull:
        y0, y1 = ys.min(), ys.max() + 1                  # convex_hull_image tests EVERY grid pixel of its
        x0, x1 = xs.min(), xs.max() + 1                  # input (O(H·W) per component), so hulling on the
        try:                                             # full image is O(num·H·W) → pathological on dense
            hull = float(convex_hull_image(c[y0:y1, x0:x1]).sum())  # many-instance masks (monuseg: 27 min
        except Exception:                                # stall). Hull AREA is translation-invariant → the
            hull = float((y1 - y0) * (x1 - x0))          # crop gives an identical .sum(). qhull still fails
                                                         # on collinear/1-px-wide THIN comps → bbox-area
                                                         # proxy (dropping them would bias the gate compact).
        acc += a * (1.0 - a / max(hull, 1.0))            # area-weighted per-component non-compactness
        tot_w += a
    return float(np.clip(acc / tot_w, 0.0, 1.0)) if tot_w > 0 else 0.0


def _fg_faintness(image, label_map) -> float:
    """Foreground/background separation |mean(fg)−mean(bg)| on the gray image ∈ [0,1]: LOW = faint,
    low-contrast objects (retinal vessels 0.07) that CLAHE helps; HIGH = distinct objects (bright
    spheroids 0.64) where CLAHE amplifies noise and hurts. Drives the per-dataset adaptive-CLAHE gate
    (aggregated over the support set → a fixed strength applied at train AND inference). Needs the GT →
    computed at fit time only; the resulting strength persists for inference."""
    g = _gray01(image)
    m = np.asarray(label_map) > 0
    if not m.any() or not (~m).any():
        return 1.0
    return float(abs(g[m].mean() - g[~m].mean()))


def _mask_descriptors(label_map) -> dict:
    """Closed-form shape descriptors from a GT mask that drive the adaptive loss weights (RQ features):
    thinness (1−solidity, per-component → tubular vs compact), fg_frac (class imbalance), complexity
    (contour excess over a circle → jaggedness), inst_density (#components → touching/dense), mean_radius
    (distance-transform thickness). Cheap; computed per support image on the GT (no grad)."""
    from skimage.measure import label as sklabel, regionprops
    from scipy import ndimage

    m = np.asarray(label_map)
    binary = m > 0
    fg = float(binary.sum())
    z = dict(thinness=0.0, fg_frac=0.0, complexity=0.0, inst_density=0.0, mean_radius=0.0)
    if fg < 4:
        return z
    comps = m if int(m.max()) > 1 else sklabel(binary)
    dt = ndimage.distance_transform_edt(binary)
    props = [p for p in regionprops(comps) if p.area >= 4]
    if not props:
        return z
    # contour complexity = perimeter²/(4π·area) − 1  (0 for a disc, ↑ for jagged/elongated), area-weighted
    compl = sum(p.area * max((p.perimeter ** 2) / (4 * np.pi * p.area) - 1.0, 0.0) for p in props)
    compl /= sum(p.area for p in props)
    return dict(thinness=_tubularity(label_map), fg_frac=fg / binary.size,
                complexity=float(compl), inst_density=float(len(props)),
                mean_radius=float(dt[binary].mean()))


def _adaptive_weights(desc: dict, instance: bool, cfg: dict) -> dict:
    """Map mask descriptors → per-term CONTINUOUS loss weights, self-gating (→0 where a term doesn't fit
    the morphology). Fixed monotone ramps (established descriptor→weight family: cbDice/Skea-Topo/BSWL).
    Dice is the always-on anchor (a topology/boundary loss alone collapses to empty fg — Kervadec'19)."""
    def ramp(x, lo, hi):
        return float(np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0))

    tau = desc["thinness"]
    imbalance = ramp(cfg["phi_ref"] - desc["fg_frac"], 0.0, cfg["phi_ref"])   # small fg → ↑
    # NB focal/tversky are NOT gated by (1−thinness): measured hrf-only, suppressing them on thin vessels
    # HURT (0.598→0.596, p5e-20) — the imbalance FN-recovery helps thin faint fg too. Keep them imbalance-
    # driven only.
    w = {
        "dice": 1.0,                                                          # region anchor, always
        "focal": cfg["w_focal"] * imbalance,                                 # imbalance/faint fg
        "tversky": cfg["w_tversky"] * imbalance,                             # small-fg FN recovery
        # clDice (skeleton topology) is for TUBULAR structures — thin (low solidity) AND genuinely NARROW.
        # (1−solidity) alone conflates filaments with TOUCHING-NUCLEI clusters (irregular but thick: monuseg
        # solidity=0.99 yet DT-radius 5.4 vs filaments' 1–4 px), where clDice hurts. So also require a small
        # DT half-width — a filament is a few px wide; radius > ~6 px is a blob, not a tube.
        "cldice": (cfg["w_cldice"] * ramp(tau, cfg["tau_lo"], cfg["tau_hi"])
                   * (1.0 - ramp(desc["mean_radius"], cfg["r_lo"], cfg["r_hi"]))),  # thin AND narrow
        "boundary": cfg["w_boundary"] * ramp(desc["complexity"], cfg["c_lo"], cfg["c_hi"]) * (1.0 - tau),
    }                                                                        # jagged contours, NOT thin
    if instance:
        w["instance"] = cfg["w_instance"] * ramp(desc["inst_density"], cfg["k_lo"], cfg["k_hi"])
        # Boundary-DoU (Lever 2, applied only when the backend flag is set): fires on dense COMPACT touching
        # instances (blobs/nuclei) where fg over-prediction/bleed merges neighbours — NOT on thin/tubular
        # structures. Thin filaments/vessels fragment into MANY components → high inst_density, but boundary
        # sharpening is wrong there (they need clDice topology). Co-gate by (1−thinness) so microtubules/drive
        # (thin, multi-component, best_v2 already wins) are damped → protects the no-regression bar (review finding).
        w["bdou"] = cfg["w_bdou"] * ramp(desc["inst_density"], cfg["k_lo"], cfg["k_hi"]) * (1.0 - tau)
        # Precision-Tversky (Lever 3, flag-gated): same dense-compact gate — penalise the diffuse interior
        # over-prediction that merges touching nuclei (precision lever, mirror of the recall Tversky above).
        w["prec"] = cfg["w_prec"] * ramp(desc["inst_density"], cfg["k_lo"], cfg["k_hi"]) * (1.0 - tau)
    return w


# default calibration (refs measured on the panel via morph_descriptors.py; tunable per-backend).
# k_lo=1 → the instance-separation term is OFF for a single instance and turns on ONLY when the annotation
# holds MULTIPLE instances (density>1), scaling with how many/dense they are (k_hi=full engagement).
# r_lo/r_hi: DT half-width band for clDice — narrow (<4 px, filaments) full, wide (>6 px, blobs/nuclei) off;
# fixes clDice wrongly firing on touching-nuclei clusters (monuseg radius 5.4) that read non-compact.
_ADAPTIVE_LOSS_CFG = dict(phi_ref=0.15, w_focal=0.5, w_tversky=0.5, w_cldice=0.5, tau_lo=0.10,
                          tau_hi=0.65, w_boundary=0.3, c_lo=0.4, c_hi=3.0, w_instance=0.6, k_lo=1, k_hi=12,
                          r_lo=4.0, r_hi=6.0, w_bdou=0.5, w_prec=0.5)


def _detect_n_classes(support, cap: int = 8) -> int:
    """#semantic classes from the support masks. Conservative — must AVOID reading INSTANCE ids as classes:
    multi-class semantic labels carry the SAME small id set in EVERY image (class 3 is class 3 everywhere),
    whereas per-instance ids are arbitrary and their SET varies image to image (e.g. few-cell masks {1..6}
    vs {1..7}). So: any id beyond `cap` → instances (1); otherwise multi-class ONLY if the nonzero id set is
    IDENTICAL across all support images (else instances → 1). NB an earlier max-only variance check let
    ctc_u373 {1..6}/{1..7} (Δ=1) slip through as "7 classes" and corrupt the loss — the set test fixes it."""
    # EXEMPLAR IS A SEMANTIC SEGMENTER. Its output is a binary foreground map and every panel dataset
    # is scored that way, so auto-detecting "semantic classes" from support ids has no correct
    # POSITIVE case here -- only false ones. It had two, measured on tulen 2026-08-28 by reproducing
    # the harness's own draw: CTC-U373 read as 7 classes on 10/10 seeds at K=1 and 3/10 at K=4, FISBE
    # on 9/10 at K=1. The set-identity test below cannot catch them, because it assumes instance ids
    # VARY across images -- and CTC-U373 has the same seven cells in image after image, so the set is
    # stable at any K, while at K=1 the test is vacuous outright.
    #
    # The cost was not cosmetic: on CTC-U373 at K=1 the multi-class head scored 0.720 against 0.479
    # binary (p=3.8e-06, all ten seeds), because seven output channels on an instance-labelled mask
    # are per-instance supervision. That is exactly the information a semantic method must not use,
    # and the same method does NOT use it at K=8, so the K-scaling curve mixed two behaviours.
    #
    # Default is therefore binary. Opt back in only for a genuinely multi-class SEMANTIC dataset.
    if not os.environ.get("ASG_MULTICLASS_HEAD"):
        return 1
    idsets = []
    for ex in support:
        u = np.unique(np.asarray(ex.label_map))
        pv = tuple(int(x) for x in u[u > 0])
        if not pv:
            continue
        if pv[-1] > cap:                          # large ids → instances, not classes
            return 1
        idsets.append(pv)
    if not idsets:
        return 1
    if len(set(idsets)) == 1:                     # SAME id set in every image → genuine multi-class labels
        return idsets[0][-1]
    return 1                                       # id set varies across images → instances → binary


def _center_dist_target(label_map) -> np.ndarray:
    """Per-instance normalised CENTER-DISTANCE map ∈ [0,1] (1 at each instance's centre → 0 at its boundary)
    — the StarDist/micro-SAM-AIS regression target. Each instance's EDT is divided by its own max so all
    instances peak at 1 (scale-invariant); local maxima = one marker per instance for the seeded watershed."""
    from scipy import ndimage
    from skimage.measure import label as sklabel

    m = np.asarray(label_map)
    comps = m if int(m.max()) > 1 else sklabel(m > 0)
    out = np.zeros(comps.shape, np.float32)
    for i in range(1, int(comps.max()) + 1):
        mask = comps == i
        if mask.sum() < 2:
            continue
        d = ndimage.distance_transform_edt(mask)
        mx = float(d.max())
        if mx > 0:
            out[mask] = (d[mask] / mx).astype(np.float32)
    return out


def _soft_skeleton(p, iters: int = 6):
    """Differentiable soft skeleton (Shit et al. clDice): iterative soft-erode/open via min/max pool."""
    import torch.nn.functional as F

    def sero(x):
        return -F.max_pool2d(-x, 3, 1, 1)          # soft erosion

    def sdil(x):
        return F.max_pool2d(x, 3, 1, 1)            # soft dilation

    skel = F.relu(p - sdil(sero(p)))
    x = p
    for _ in range(iters):
        x = sero(x)
        skel = skel + F.relu(x - sdil(sero(x))) - skel * F.relu(x - sdil(sero(x)))
    return skel


def _soft_cldice(pred, target, eps: float = 1e-6, target_skel=None):
    """soft-clDice topology loss for tubular structures (1 - clDice).

    ``target_skel`` is the ground truth's soft skeleton, which does NOT change while the head trains.
    Recomputing it every step was costing a full six-iteration erode/open pass over the native-
    resolution label on every one of the (up to 500) epochs, for a tensor that is fixed. It is now
    computed once per support image and carried in, and because the augmentation is the dihedral
    group -- whose elements commute exactly with a 3x3 symmetric min/max pool -- the caller can
    apply the SAME `_d4` to the cached skeleton instead of re-skeletonising the augmented label.
    Verified exact: max |difference| is 0 over all eight D4 elements, square and non-square.
    """
    sp = _soft_skeleton(pred)
    st = _soft_skeleton(target) if target_skel is None else target_skel
    tprec = (sp * target).sum() / (sp.sum() + eps)   # skeleton-precision
    tsens = (st * pred).sum() / (st.sum() + eps)     # skeleton-sensitivity
    cldice = 2 * tprec * tsens / (tprec + tsens + eps)
    return 1 - cldice


def _blob_markers(fg, gray, min_sigma: float = 1.0, max_sigma: float = 12.0,
                  num_sigma: int = 6, threshold: float = 0.04):
    """Training-free SCALE-ADAPTIVE instance markers (W2). Multi-scale Laplacian-of-Gaussian blob
    detection on the intensity gives ONE seed per blob *regardless of crowding*, fixing the
    scale-blind fixed-``min_distance`` DT peaks that under-seed dense fields (dsb AP 0.06 on crowds).
    Nuclei are bright convex blobs; if the foreground is darker than the background we invert so LoG
    fires on them. Markers are restricted to the foreground; returns an int marker image or None
    (caller falls back to DT peaks). No training → dodges the Phase-J data-hungry learned boundary."""
    from skimage.feature import blob_log

    fg = np.asarray(fg, bool)
    if not fg.any():
        return None
    g = np.asarray(gray, np.float32)
    g = (g - g.min()) / (np.ptp(g) + 1e-6)
    if g[fg].mean() < g[~fg].mean() if (~fg).any() else False:
        g = 1.0 - g                                # make the objects bright for LoG
    try:
        blobs = blob_log(g * fg, min_sigma=min_sigma, max_sigma=max_sigma,
                         num_sigma=num_sigma, threshold=threshold)
    except Exception as e:
        # None → caller falls back to DT peaks. Surface it: a SYSTEMATIC blob_log failure would silently
        # report DT-peak instance-AP under an explicitly-selected instance_mode="blob".
        print(f"    [blob_markers] blob_log failed ({e!r}) → DT-peak fallback", flush=True)
        return None
    markers = np.zeros(fg.shape, int)
    n = 0
    for y, x, _s in blobs:
        yi, xi = int(round(y)), int(round(x))
        if 0 <= yi < fg.shape[0] and 0 <= xi < fg.shape[1] and fg[yi, xi]:
            n += 1
            markers[yi, xi] = n
    return markers if n > 0 else None


def _watershed_instances(fg, class_id, min_distance: int = 5, min_area: int = 30, boundary=None,
                         markers=None):
    """Marker-controlled watershed. Marker source, in priority: (1) explicit ``markers`` (W2
    scale-adaptive blob seeds — one per nucleus, splits crowds), (2) connected components of fg
    MINUS the learned inter-instance ``boundary``, (3) distance-transform peaks. Watershed then
    splits what connected-components would merge. One model, no SAM."""
    from scipy import ndimage
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    fg = np.asarray(fg, bool)
    if not fg.any():
        return []
    dt = ndimage.distance_transform_edt(fg)
    if markers is not None and int(np.asarray(markers).max()) > 0:   # W2: blob-detector seeds
        lab = watershed(-dt, np.asarray(markers), mask=fg)
    elif boundary is not None:                     # W2: learned-boundary-seeded markers
        core = fg & ~np.asarray(boundary, bool)
        markers = _label(core)
        lab = watershed(-dt, markers, mask=fg) if int(markers.max()) > 0 else _label(fg)
    else:
        coords = peak_local_max(dt, min_distance=min_distance, labels=fg, exclude_border=False)
        if len(coords) == 0:
            lab = _label(fg)
        else:
            markers = np.zeros(fg.shape, dtype=int)
            for i, (y, x) in enumerate(coords, 1):
                markers[y, x] = i
            lab = watershed(-dt, markers, mask=fg)
    out = []
    for i in range(1, int(lab.max()) + 1):
        m = lab == i
        if int(m.sum()) >= min_area:
            out.append(InstanceMask(mask=m, points=None, class_id=class_id,
                                    instance_id=len(out), score=1.0))
    return out


def _ridge_map(gray) -> np.ndarray:
    """Polarity-agnostic native-res ridge/valley response in [0, 1], HIGH on the thin membrane groove
    BETWEEN touching cells (works whether the interface is a DARK valley or a BRIGHT halo). Max over both
    Frangi polarities → catches either. This is the classical REPULSIVE (split) signal; DINO features can't
    supply it for same-class neighbours (identical features)."""
    from skimage.filters import frangi

    g = np.asarray(gray, np.float32)
    g = (g - g.min()) / (np.ptp(g) + 1e-6)
    try:
        r = np.maximum(frangi(g, black_ridges=True), frangi(g, black_ridges=False))
    except Exception as e:
        print(f"    [ridge_map] frangi failed ({e!r}) → zero ridge (watershed falls back to DT-only)",
              flush=True)
        return np.zeros_like(g)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    mx = float(r.max())
    if mx < 1e-6:                                          # flat / low-contrast → ridge is DEAD, not a real edge map
        print("    [ridge_map] ridge ~dead (flat/low-contrast image) → this image's watershed is DT-only, "
              "merge-veto loses its ridge term", flush=True)
        return np.zeros_like(g)
    return (r / (mx + 1e-6)).astype(np.float32)


# number of channels the expanded classical bank (`bank_extra`) appends (2 top-hat scales × 2 polarities
# + 1 texture). Kept as a module constant so the count is documented in one place; the head's n_classical
# still AUTO-propagates from the bank's actual output width at fit — this is only for readability/tracing.
_BANK_EXTRA_N = 5


def _extra_bank_features(gray) -> np.ndarray:
    """Curated EXTRA classical channels for the ``bank_extra`` HyperBank expansion (roadmap #8). Returns
    ``[k, H, W]`` float32 (``k == _BANK_EXTRA_N``) at the SAME spatial resolution as the passed gray:

      • white/black TOP-HAT at two small disk radii (3, 8) — enhance small BRIGHT / DARK structures on a
        slowly-varying background (nuclei, spots, faint blobs) that the ridge/blob banks under-serve;
      • one TEXTURE cue = local STD in a 7×7 window (cheap, robust; no dtype conversion) — a monuseg-style
        H&E histopathology signal.

    Each channel is per-channel Z-SCORED (mean 0 / unit std) to MATCH the vendored bank's scale — the bank
    normalises every sub-bank group with ``GroupNorm(num_groups=1, affine=False)`` (per-image, ~unit-std),
    so unit-std channels sit at parity under the 1×1 fusion and cannot dominate/vanish. Per-channel (not a
    single group-norm over all extras) because the extras are HETEROGENEOUS in magnitude (a top-hat ≈[0,0.3]
    vs a texture std) — one shared norm would let the largest swamp the rest. A DEGENERATE (constant) channel
    normalises to ALL ZEROS (never NaN); a final ``nan_to_num`` guards any residual non-finite value.

    Pure classical (skimage/scipy, no grad) — a fixed prior like the rest of the frozen bank."""
    from scipy.ndimage import uniform_filter
    from skimage.morphology import black_tophat, disk, white_tophat

    g = np.asarray(gray, np.float32)
    g = (g - g.min()) / (np.ptp(g) + 1e-6)               # [0,1], matching the bank's per-image input scale
    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    chans = []
    for radius in (3, 8):                               # two small scales (7×7 and 17×17 footprints)
        fp = disk(radius)
        chans.append(white_tophat(g, fp))              # small bright-on-dark structures
        chans.append(black_tophat(g, fp))              # small dark-on-bright structures
    win = 7                                            # local-std texture: sqrt(E[x²] − E[x]²), reflect pad
    mu = uniform_filter(g, win, mode="reflect")
    mu2 = uniform_filter(g * g, win, mode="reflect")
    chans.append(np.sqrt(np.clip(mu2 - mu * mu, 0.0, None)))
    out = np.stack(chans, 0).astype(np.float32)        # [k, H, W]
    for i in range(out.shape[0]):                      # per-channel z-score; constant channel → all-zeros
        c = out[i]
        s = float(c.std())
        out[i] = (c - float(c.mean())) / s if s > 1e-6 else 0.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _basin_mean_features(lab, feat_grid) -> dict:
    """Mean L2-normalised DINO feature per basin. Each NATIVE pixel is mapped to its grid cell and the basin
    feature is the area-weighted mean over the grid cells its pixels fall in → EVERY basin with >=1 pixel gets
    a feature (no small/dense instance is silently dropped, unlike NN-downsampling the native label map to the
    grid). Returns {label: unit_vector[D]}."""
    lab = np.asarray(lab).astype(int)
    fg = np.asarray(feat_grid, np.float32)
    G0, G1, D = fg.shape
    H, W = lab.shape
    gy = np.clip(np.arange(H) * G0 // max(H, 1), 0, G0 - 1)[:, None]
    gx = np.clip(np.arange(W) * G1 // max(W, 1), 0, G1 - 1)[None, :]
    cell = (gy * G1 + gx)                                  # [H,W] grid-cell index per native pixel (broadcast)
    cell = np.broadcast_to(cell, (H, W))
    Fflat = fg.reshape(G0 * G1, D)
    means = {}
    for i in np.unique(lab):
        if i == 0:
            continue
        v = Fflat[cell[lab == i]].mean(0)                  # area-weighted mean over the cells this basin covers
        means[int(i)] = v / (np.linalg.norm(v) + 1e-6)
    return means


def _affinity_merge(lab, feat_grid, ridge, merge_cos: float, merge_ridge: float):
    """DINO-affinity MERGE-VETO: merge two adjacent basins iff their mean-feature cosine is HIGH (clearly the
    same object) AND the classical ridge on their shared border is WEAK (no real membrane there). Affinity is
    used only to VETO a wrong split — never to make one (it cannot split same-class neighbours). Union-find
    over the region-adjacency graph; returns a relabelled map."""
    try:                                                       # RAG moved skimage.future.graph → skimage.graph
        from skimage.graph import RAG
    except ImportError:
        from skimage.future.graph import RAG

    lab = np.asarray(lab).astype(int)
    if int(lab.max()) < 2:
        return lab
    means = _basin_mean_features(lab, feat_grid)
    rag = RAG(lab)
    parent = {i: i for i in range(1, int(lab.max()) + 1)}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # Precompute the mean ridge on EVERY adjacent-label interface in ONE O(H·W) pass over 8-connected shifts
    # (a per-edge full-image dilation was O(edges·H·W) → minutes on monuseg's 400+ instances/image).
    maxlab = int(lab.max())
    stride = maxlab + 1
    keys_all, rv_all = [], []
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        sl = np.roll(lab, (dy, dx), axis=(0, 1))
        rr = np.roll(ridge, (dy, dx), axis=(0, 1))
        m = (lab != sl) & (lab > 0) & (sl > 0)
        if not m.any():
            continue
        lo = np.minimum(lab[m], sl[m]).astype(np.int64)
        hi = np.maximum(lab[m], sl[m]).astype(np.int64)
        keys_all.append(lo * stride + hi)
        rv_all.append(0.5 * (ridge[m] + rr[m]))
    border_mean = {}
    if keys_all:
        keys = np.concatenate(keys_all)
        uk, inv = np.unique(keys, return_inverse=True)
        rsum = np.bincount(inv, weights=np.concatenate(rv_all).astype(np.float64))
        rcnt = np.bincount(inv)
        border_mean = {int(k): rsum[j] / rcnt[j] for j, k in enumerate(uk)}
    for a, b in rag.edges:
        if a == 0 or b == 0 or a not in means or b not in means:
            continue
        cos = float(np.dot(means[a], means[b]))
        if cos <= merge_cos:
            continue
        br = border_mean.get(min(a, b) * stride + max(a, b), 1.0)  # no interface found → don't merge
        if br < merge_ridge:                                   # high affinity AND weak ridge → same instance
            parent[find(a)] = find(b)
    remap = {}
    lut = np.zeros(maxlab + 1, dtype=int)                      # vectorized relabel (O(H·W+N), not O(N·H·W))
    for i in range(1, maxlab + 1):
        r = find(i)
        remap.setdefault(r, len(remap) + 1)
        lut[i] = remap[r]
    return lut[lab]


def _affinity_watershed_instances(fg, ridge, feat_grid, class_id, *, r_star: float = 5.0,
                                  w_dt: float = 1.0, w_ridge: float = 1.0, merge_cos: float = 0.9,
                                  merge_ridge: float = 0.25, min_area: int = 30):
    """SAM-free touching-instance separation (novel): markers = DT peaks of the fg mask (≥ r_star apart);
    watershed over elevation ``-w_dt·DT_n + w_ridge·ridge`` (geometry splits, classical ridge sharpens the
    cut); then a DINO-affinity MERGE-VETO removes over-splits. No learned dense field → few-shot-safe."""
    from scipy import ndimage
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    fg = np.asarray(fg, bool)
    if not fg.any():
        return []
    dt = ndimage.distance_transform_edt(fg)
    dtn = dt / (dt.max() + 1e-6)
    r = np.asarray(ridge, np.float32)
    if r.shape != fg.shape:                                    # align ridge to native fg
        from skimage.transform import resize
        r = resize(r, fg.shape, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
    coords = peak_local_max(dt, min_distance=max(1, int(round(r_star))), labels=fg, exclude_border=False)
    if len(coords) == 0:
        lab = _label(fg)
    else:
        markers = np.zeros(fg.shape, int)
        for i, (y, x) in enumerate(coords, 1):
            markers[y, x] = i
        lab = watershed(-w_dt * dtn + w_ridge * r, markers, mask=fg)
    lab = _affinity_merge(lab, feat_grid, r, merge_cos, merge_ridge)
    out = []
    for i in range(1, int(lab.max()) + 1):
        m = lab == i
        if int(m.sum()) >= min_area:
            out.append(InstanceMask(mask=m, points=None, class_id=class_id,
                                    instance_id=len(out), score=1.0))
    return out


def _end_tangent(pix, end, r=5):
    """Unit tangent of a skeleton fragment at endpoint ``end``, pointing OUTWARD (away from the fragment
    body), from a PCA of the fragment pixels within radius ``r`` of ``end``. ``pix`` = Nx2 (y,x)."""
    d = pix - np.asarray(end)
    near = pix[(d[:, 0] ** 2 + d[:, 1] ** 2) <= r * r]
    if len(near) < 2:
        return np.zeros(2, np.float32)
    c = near - near.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    t = vt[0]
    body = near.mean(0) - np.asarray(end)                      # points from end toward the body
    if float(np.dot(t, body)) > 0:                            # flip so t points OUTWARD (away from body)
        t = -t
    return (t / (np.linalg.norm(t) + 1e-6)).astype(np.float32)


def _trace_filament_instances(fg, class_id, *, angle_max_deg: float = 40.0, width: float = 2.0,
                              min_len: int = 8, link_min: int = 3, margin: float = 0.05):
    """SAM-free CROSSING-filament separation (soft/layered): skeletonise the fg, split at junctions into
    fragments, LINK fragment ends through each junction by ORIENTATION CONTINUITY (straight-through: tangents
    near-opposite, within ``angle_max_deg`` of a straight line). Union linked fragments into filament traces;
    crossing pixels are owned by BOTH filaments (amodal — no hard partition). Uses PER-JUNCTION greedy matching
    with an ambiguity ``margin``: a >2-way (T) junction links its one collinear pair and ABSTAINS the stub, and
    a near-parallel crossing (two comparably-straight options) abstains rather than guessing.

    FIXED thresholds (NOT support-derived — a fixed-threshold heuristic). KNOWN LIMITATIONS: PCA-chord tangents
    bias at sharp bends; a sub-``link_min`` bridge fragment between two very close crossings can split one
    filament into two; genuine translucent overlap is out of scope. Returns [] (with a warning) when the fg is
    non-empty but no trace survives, rather than silently reporting zero. NOT yet benchmark-validated — the
    microtubule GT is semantic (clDice), so this is a reviewed capability, not an instance-AP result."""
    from collections import defaultdict
    from scipy import ndimage
    from skimage.morphology import skeletonize

    fg = np.asarray(fg, bool)
    if not fg.any():
        return []
    skel = skeletonize(fg)
    if not skel.any():
        return []
    k = np.ones((3, 3), int)
    deg = ndimage.convolve(skel.astype(int), k, mode="constant") - skel.astype(int)   # neighbour count
    junction = skel & (deg >= 3)
    jmap = ndimage.label(junction, structure=k)[0]
    frag = skel & ~ndimage.binary_dilation(junction, k)       # detach fragments at junctions (+1 px margin)
    lab, n = ndimage.label(frag, structure=k)
    if n == 0:
        print("    [trace] fg non-empty but skeleton is all-junction (compact blob) → 0 traces", flush=True)
        return []
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # per fragment: end pixels adjacent to a junction + their OUTWARD tangents (skip degenerate zero tangents)
    ends = []                                                 # (frag_label, end_yx, outward_tangent)
    for L in range(1, n + 1):
        pix = np.argwhere(lab == L)
        if len(pix) < link_min:
            continue
        nb = ndimage.convolve((lab == L).astype(int), k, mode="constant")[tuple(pix.T)] - 1
        for p in pix[nb <= 1]:                                # skeleton endpoints of this fragment
            win = junction[max(0, p[0] - 2):p[0] + 3, max(0, p[1] - 2):p[1] + 3]
            if win.any():                                     # this end abuts a junction
                t = _end_tangent(pix, p)
                if np.any(t):
                    ends.append((L, tuple(p), t))
    # bucket each end by its NEAREST junction blob → never link across the wrong junction (M3)
    by_junc = defaultdict(list)
    for idx, (L, e, t) in enumerate(ends):
        w = jmap[max(0, e[0] - 2):e[0] + 3, max(0, e[1] - 2):e[1] + 3]
        labs = w[w > 0]
        if labs.size:
            by_junc[int(np.bincount(labs).argmax())].append(idx)
    # per-junction greedy straight-through matching WITH abstention (T-junctions + near-parallel; O(ends_j^2))
    cos_thr = -np.cos(np.deg2rad(angle_max_deg))              # opposite tangents → dot ≈ −1
    for idxs in by_junc.values():
        cand = []
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ia, ib = idxs[a], idxs[b]
                if find(ends[ia][0]) == find(ends[ib][0]):
                    continue
                dot = float(np.dot(ends[ia][2], ends[ib][2]))
                if dot < cos_thr:                            # near-collinear enough to be a candidate
                    cand.append((dot, ia, ib))
        cand.sort()                                          # straightest (most negative) first
        used = set()
        for pos, (dot, ia, ib) in enumerate(cand):
            if ia in used or ib in used:
                continue
            alt = min([d for q, (d, x, y) in enumerate(cand)
                       if q != pos and (x in (ia, ib) or y in (ia, ib))], default=1.0)
            if dot > alt - margin:                           # a comparably-straight alternative → ambiguous → abstain
                continue
            parent[find(ends[ia][0])] = find(ends[ib][0])
            used.add(ia)
            used.add(ib)
    # group fragments → traces; dilate to width; share the junction pixels a trace passes through (amodal)
    groups = defaultdict(list)
    for L in range(1, n + 1):
        groups[find(L)].append(L)
    w2 = max(1, int(round(width)))
    struct = np.ones((2 * w2 + 1, 2 * w2 + 1), bool)         # odd, centred (was even for non-int width — M2)
    out = []
    for members in groups.values():
        m = np.isin(lab, members)
        if int(m.sum()) < min_len:
            continue
        touchj = ndimage.binary_dilation(m, np.ones((5, 5), bool)) & junction   # reach the +2px-detached junctions
        mask = ndimage.binary_dilation(m | touchj, struct) & fg
        if mask.any():
            out.append(InstanceMask(mask=mask, points=None, class_id=class_id,
                                    instance_id=len(out), score=1.0))
    if not out and fg.any():
        print("    [trace] fg non-empty but no trace survived (dense/compact field) → 0 instances", flush=True)
    return out



def _d4(t, k: int, flip: bool):
    """One element of the dihedral group of the square, applied to a [B,C,H,W] tensor.

    Augmentation here acts on the CACHED FEATURE GRID, not on the image, so the only transforms
    available are those that commute with the frozen encoder: the eight symmetries of the square.
    Arbitrary rotations would require re-encoding and would interpolate features, which is not the
    same thing as the features of an interpolated image.

    No channel permutation is needed, and that is a property of this bank rather than a general
    truth. Directional quantities never reach its output: it emits Lx^2+Ly^2, the Hessian trace and
    determinant, Frangi from Hessian EIGENVALUES, Hu invariants and monogenic energy -- all
    rotation-invariant combinations. A bank exposing raw Ix, Iy or Ixy would need those channels
    permuted and negated under the same transform, and rotating it without that would train the head
    on physically impossible inputs.
    """
    import torch                      # this module imports torch per-function, not at module scope

    if t is None:
        return None
    if k:
        t = torch.rot90(t, k, dims=(-2, -1))
    return torch.flip(t, (-1,)) if flip else t


class HeadFusionBackend:
    def __init__(self, device: str | None = None, hidden: int = 256, proj_dim: int = 32,
                 epochs: int = 60, lr: float = 1e-3, max_side: int = 1536,
                 trainable_classical: bool = False, instance_mode: str = "watershed",
                 tile_classical: bool = False, tile: int = 1024, guided_fuse: bool = False,
                 boundary_head: bool = False, contrast_norm: bool = False, cldice: bool = False,
                 fine_scales: bool = False, scale_fusion: bool = False, encoder=None,
                 fine_max_grid: int = 160, upsampler=None, batch_size: int = 4, amp: bool = True,
                 stem: str = "wide", depth: int = 2, hidden_film: int = 64, eca: bool = False,
                 early_stop: bool = False, patience: int = 25,
                 min_rel_improve: float = 0.002, weight_decay: float = 0.0,
                 dropout: float = 0.0, se: int = 0, mix: bool = False, bilinear: int = 0,
                 banknorm: bool = False, aug: str = "flip", znorm: bool = False,
                 classical_up: bool = False, classical_lowres_side: int = 768,
                 grad_checkpoint: bool = False, cldice_adaptive: bool = False,
                 channels_last: bool = False, compile_head: bool = False,
                 ridge_init: bool = False,
                 lovasz: bool = False, convex_loss: bool = False,
                 n_aug: int = 0, aug_elastic: float = 0.0, morph_post: bool = False,
                 val_tile: float = 0.0, val_loo: bool = False, sgd_poly: bool = False,
                 tau_cal: bool = False, morph_orient: bool = False,
                 cldice_base: float = 0.5, cldice_thin_ref: float = 0.35,
                 clahe_adaptive: bool = False, clahe_faint_ref: float = 0.3,
                 adaptive_loss: bool = False, adaptive_loss_eps: float = 0.02, n_classes: int = 0,
                 dist_head: bool = False, thin_adaptive: bool = False, thin_ref: float = 0.4,
                 thin_max_side: int | None = None, color_adaptive: bool = False,
                 color_margin: float = 1.05,
                 bank_unfreeze_adaptive: bool = False, competitive_gate: bool = False,
                 corr_prior: bool = False, film: bool = False, bank_extra: bool = False,
                 n_proto: int = 1, boundary_dou: bool = False, prec_loss: bool = False,
                 dino_only: bool = False, bank_only: bool = False,
                 dino_scale: str = "both", rgb_feat: bool = False,
                 bankselect: bool = False, scaleconf: bool = False):
        self.device = device or "cpu"
        self.rgb_feat = rgb_feat  # A/B (simplification test): append the raw RGB channels as native
        # features instead of the closed-form colour-channel selection; the head/gate weight them itself.
        # LEVER (bankselect): apply the SAME support-driven Fisher separability used for colour-channel
        # selection to the CLASSICAL BANK channels — keep the fg/bg-separable priors, zero the rest — so one
        # rule governs input+bank instead of per-filter gates. DEFAULT OFF (composable); zeros (not drops)
        # the discarded channels, so shape/head-size are unchanged and the change is a pure add-on.
        self.bankselect = bankselect
        self._bank_keep = None          # boolean keep-mask over bank channels, set once at fit from support
        self.bankselect_ratio = 0.3     # keep channels with median support Fisher >= ratio * best channel's
        # LEVER (scaleconf): set the classical bank's Frangi/Sauvola/LoG SCALES from the support masks' own
        # object radii, instead of the fixed grid. We cluster the medial-axis distance-transform radii of the
        # support foreground and use ONE scale per cluster centroid — so the NUMBER and VALUES of scales
        # self-configure to the morphology (thin vessels get small σ, big blobs large σ), and a mis-scaled
        # filter (which returns nothing, or crashes on tiny images) is made to match the object it detects.
        # It also subsumes thin_adaptive's fine-Sauvola window choice — but NOT its other effect
        # (`tile_classical`), which thin_adaptive still controls.
        # TWO CONFOUNDS, because the A/B arm is NOT a pure scale swap and a verdict must not be read as one:
        #   * UNITS — the baseline passes Frangi σ = (1,2,4,8,16) ABSOLUTE px but leaves LoG at HyperBank's
        #     default (0.001…0.016), which `_resolve_sigma` reads as a FRACTION of the reference size. So
        #     baseline LoG scales WITH the image while scaleconf's LoG is absolute: the arm changes LoG's
        #     semantics, not only its values.
        #   * WIDTH — one centroid means one Frangi σ and one LoG σ instead of five each, so the bank (and
        #     with it the fusion head's classical input width, auto-derived via `_n_classical`) drops from
        #     ~35 channels to as few as ~15. Part of any delta is therefore a capacity change.
        # DEFAULT OFF.
        self.scaleconf = scaleconf
        self._bank_scales = None         # list of centroid radii (px), set once at fit from support
        # SENSITIVITY scan (ASG_LOSS_SCALE): global multiplier on the adaptive auxiliary loss weights
        # (clDice/boundary/Tversky/…) relative to the region anchor; 1.0 = the self-config heuristic. Used
        # to measure how flat the loss-weight landscape is around the heuristic (defends "no per-dataset tuning").
        self.loss_scale = float(os.environ.get("ASG_LOSS_SCALE", "1.0"))
        self.dino_only = dino_only  # ABLATION: zero the classical prior bank → DINO-features-only head
        # ABLATION, the mirror of dino_only: zero the DINO grids so the head reads the classical bank
        # alone. Shapes are preserved exactly as dino_only preserves them, so the head keeps its
        # 32+32+35 concatenation and its parameter count, and the comparison is between two inputs to
        # ONE architecture rather than between two architectures. It closes the asymmetric cell of the
        # 2x2: features under our head, priors under a random forest, both under our head, and now
        # priors under our head.
        self.bank_only = bool(bank_only)
        # ABLATION over the two DINO scales: "both" = coarse⊕fine scale-fusion (default best_v2); "coarse" =
        # coarse whole-image grid only; "fine" = native-tiled fine grid only (coarse dropped). coarse/fine
        # force scale_fusion off (single DINO branch); _dino_x() supplies the chosen grid as the head input.
        self.dino_scale = dino_scale
        self.hidden, self.proj_dim = hidden, proj_dim
        self.epochs, self.lr = epochs, lr
        self.max_side = max_side  # cap native side for tractable training (keeps most crumb detail)
        self.trainable_classical = trainable_classical  # unfreeze the classical filter params
        self.fine_scales = fine_scales  # W1: add sub-pixel classical scales for sub-10px crumbs
        self.instance_mode = instance_mode  # "watershed" | "cc" | "blob" (scale-adaptive markers)
        self.tile_classical = tile_classical  # compute the classical bank on native-res tiles (no cap)
        self.tile = tile
        # Cheaper-than-tiling uncap: compute the bank at a low cap then image-GUIDED-upsample to native
        # (inference only). Trades exactness for ~cap²/native² of the Frangi/LoG compute.
        self.classical_up = classical_up
        self.classical_lowres_side = classical_lowres_side
        # Accuracy-EXACT memory lever: recompute the native-res head activations in backward instead of
        # storing them (identical gradients, global dice intact) → frees the O(out_hw²) activation term
        # that caps trainable resolution. Enables TRAINING at native res so the head co-adapts to native
        # detail (inference res must MATCH training res — a bank-res mismatch is what made tiling regress).
        self.grad_checkpoint = grad_checkpoint
        # THROUGHPUT LEVERS, both off by default so the reported configuration is unchanged
        # until each is measured. The head is 0.5M parameters but runs at NATIVE resolution over
        # ~1M pixels and tens of channels, so the fit is memory-bandwidth-bound rather than
        # compute-bound -- which is the regime where a channels-last layout and kernel fusion
        # pay, and where neither changes what is computed.
        self.channels_last = channels_last
        self.compile_head = compile_head
        self.ridge_init = ridge_init
        # Support-derived DECISION THRESHOLD. The head is trained on an adaptive loss but read out
        # at logit 0, i.e. p>0.5, which is the Bayes rule for accuracy and NOT for Dice or clDice.
        # Koyejo et al. (NeurIPS 2014, Thm 2) put the optimum for the linear-fractional family at a
        # threshold, and registry C61 measured a swept threshold worth +0.113 clDice on DRIVE -- on
        # the closed-form arm, whose baseline sits far below this one, so whether it survives here
        # is exactly what the screen must answer rather than assume.
        self.tau_cal = tau_cal
        self._tau = 0.0
        # Ten EXTRA prior channels: white and black top-hats at 3/7/15 px, structure-tensor
        # orientation as (sin 2theta, cos 2theta), its coherence, and a median residual. Top-hats
        # and orientation are the classical vessel and filament detectors the current bank lacks,
        # which is where the whole nnU-Net deficit sits (centreline -0.042, overlap already level).
        self.morph_orient = morph_orient
        # Swap the region ANCHOR for the Lovasz hinge; every other term keeps its adaptive weight.
        self.lovasz = lovasz
        # THE ONE-SOLVE OBJECTIVE. Three terms, each convex IN THE WEIGHTS -- not merely in the
        # prediction, which is a weaker and, on its own, useless property: the solve is over `w`,
        # and a term that is linear in `p = sigmoid(Xw)` is generally NOT convex in `w`. The first
        # version of this objective got that wrong for two of its three terms; both now consume
        # logits through softplus. Sum of convex => convex => a single solve is the global optimum:
        #   Lovasz(logits, GT)          convex surrogate for 1 - IoU; supplies the pressure
        #                               against over-prediction that recall alone lacks
        #   lam_thin * SkelRecall       <S, softplus(-z)>/sum(S), S from the GT skeleton (fixed)
        #                               = a convex upper bound on clDice's Tsens
        #   lam_bridge * <W, softplus(z)>   W depends only on GT -- stands in for clDice's Tprec,
        #                               which needs skeleton(pred) and cannot be convex at all
        # This is NOT clDice. It is an objective built from the terms that survive one solve,
        # and whether the substitution costs accuracy is a question for a screen.
        # WHAT NO VERSION OF THIS CAN DO, so it is not silently claimed: connectivity (b0) is not
        # computable by ANY local decomposable term -- two images differing by one moved pixel can
        # share every 2x2 AND 3x3 window multiset, area, perimeter and Euler characteristic while
        # differing in b0 -- and connectivity-constrained MAP is NP-hard even with unary terms only
        # (Vicente, Kolmogorov & Rother, CVPR 2008, Thm 1). These terms bias the fit; they do not
        # enforce topology.
        self.convex_loss = convex_loss
        # Re-encoded augmentation: copies per support image, and the elastic strength.
        self.n_aug = int(n_aug)
        # nnU-Net analogues (both OFF in the reported configuration).
        #
        # `val_tile` > 0 holds out a strip of every support image and early-stops on the FOREGROUND
        # IoU measured there, instead of on the training-loss plateau. The plateau rule's own
        # docstring concedes it detects convergence rather than overfitting, because every support
        # pixel is trained on; this supplies the held-out signal it says it lacks, and it works at
        # K=1, where the strip comes out of the single image. The head is 1x1 convolutions over a
        # frozen backbone plus one 3x3 in the upsampler's low-res coefficient net, so its receptive
        # field is small and the strip's interior is genuinely unseen by the loss.
        #
        # `sgd_poly` swaps AdamW at a constant rate for nnU-Net's own optimiser: SGD with Nesterov
        # momentum 0.99 and the polynomial schedule lr*(1-e/E)^0.9. Read out of the installed
        # nnunetv2 (nnUNetTrainer: initial_lr 1e-2, weight_decay 3e-5, momentum 0.99, PolyLRScheduler),
        # not paraphrased from the paper.
        self.val_tile = float(val_tile)
        if not 0.0 <= self.val_tile < 0.5:
            raise ValueError(f"val_tile={val_tile}: a held-out strip must be under half the image")
        # LEAVE-ONE-OUT VALIDATION, the honest form of the same idea. `val_tile` cuts a strip out of
        # EVERY support item, which at K=1 means training on 75% of the only image there is; measured
        # on the C84 screen that costs -0.117 on DSB2018 and -0.035 on MoNuSeg, and it inflates the
        # seed spread twentyfold, because the strip is small enough to be a noisy stopping signal in
        # its own right. Holding out one whole support MASK instead costs 1/K of the data (12.5% at
        # K=8, against 25% of every image) and yields a genuinely independent case -- which is what
        # nnU-Net's fold split actually is, and what a strip out of the same image is not.
        #
        # It is undefined at K=1, and that is the real answer rather than a limitation to engineer
        # around: with one annotated image there is no held-out data, and the strip experiment is the
        # evidence that pretending otherwise costs more than it returns.
        self.val_loo = bool(val_loo)
        if self.val_loo and self.val_tile > 0:
            raise ValueError("val_loo and val_tile are two answers to the same question; pick one")
        self.sgd_poly = bool(sgd_poly)
        self.aug_elastic = float(aug_elastic)
        # Support-derived morphological post-processing; the rule is computed at fit time.
        self.morph_post = bool(morph_post)
        self._morph_rule = None
        self._tau = 0.0
        self._wbridge_cache: dict = {}
        # ADAPTIVE clDice: instead of a fixed per-dataset weight (canonical clDice α∈[0,0.5], the manual
        # switch we want to remove), scale the topology term PER IMAGE by a thinness descriptor of the
        # annotated shape (skeleton-to-area ratio) → ~0 on compact blobs (where clDice roughens smooth
        # boundaries), full on thin/branching structures (where it helps). Region loss stays always-on at
        # full weight (a topology term alone collapses to empty fg — Kervadec'19). Fixed morphology-derived
        # weight, NOT learned: learned multi-loss weights (Kendall&Gal/GradNorm) overfit / don't transfer
        # at K≈8. Deriving a per-IMAGE loss-term weight from a global GT shape descriptor is novel.
        self.cldice_adaptive = cldice_adaptive
        self.cldice_base = cldice_base        # topology weight at full engagement (thin structures)
        self.cldice_thin_ref = cldice_thin_ref  # (unused; kept for back-compat)
        # ADAPTIVE CLAHE: strength derived PER-DATASET from the support set's fg/bg faintness (CLAHE is
        # preprocessing applied at inference too → no per-image GT there, so gate at the dataset level).
        # Faint objects (vessels) → full CLAHE (+0.045 on hrf); distinct objects (spheroids/crumbs) → off
        # (CLAHE amplified noise → the rozpad regression). Computed once at first fit, then fixed.
        self.clahe_adaptive = clahe_adaptive
        self.clahe_faint_ref = clahe_faint_ref
        self._clahe_strength = None           # set once from the support morphology at fit
        # ADAPTIVE-LOSS constructor: build the loss per support image from its mask morphology — Dice anchor
        # + morphology-gated {focal, tversky, clDice, boundary, instance}; terms with weight < eps are SKIPPED
        # (not computed) → filaments get Dice+clDice, blobs Dice+focal+tversky, touching-inst adds a term.
        self.adaptive_loss = adaptive_loss
        self.adaptive_loss_eps = adaptive_loss_eps
        # MULTI-CLASS: 0 = auto-detect from the support annotations (binary unless masks carry consistent
        # small class ids), >1 = forced C-class. When >1 the region anchor becomes multi-class CE+Dice and
        # the head outputs C channels; morphology-gated terms still act on the foreground union.
        self.n_classes = n_classes
        self._n_cls = 1
        # DT-regression instance head — predicts a per-instance normalised center-distance map; its peaks
        # seed a watershed (StarDist/micro-SAM-AIS pattern). Few-shot-friendlier than the boundary-BCE head.
        self.dist_head = dist_head
        # ADAPTIVE THIN-STRUCTURE gate: the resolution levers (native-res tiled classical bank + finer
        # Sauvola scales) recover 1–3 px vessels/filaments that the max_side cap destroys (HRF 0.22→0.41,
        # H-A), but applied UNIFORMLY they regress blobs (seam artefacts / over-fine bank) — which is why
        # tiling was "superseded". So GATE them on the support morphology: turn native classical ON only
        # when the annotated shapes are thin (mean per-component tubularity > thin_ref). Blobby support →
        # keep the cheap cap (no regression). Decided once per dataset at first fit, mirroring CLAHE.
        self.thin_adaptive = thin_adaptive
        self.thin_ref = thin_ref
        # native-classical is ONLY safe on SMALL-thin structures (microtubules ~630 px: +0.018). On HUGE-thin
        # images (hrf 3504 px, 5.2× downscale) the native-tiled bank vs capped-trained head is a catastrophic
        # distribution shift (hrf 0.607→0.469) — so ALSO require the support images to be small.
        #
        # This bound USED to be a hardcoded 1500, and that number was placed between HRF (3504 px) and the
        # rest of the panel because HRF fell. A threshold positioned to separate one dataset from the others
        # is per-dataset tuning however it is routed, it contradicts the "no per-dataset tuning" claim, and
        # it promises nothing at all about the next dataset.
        #
        # The comment above already names the real mechanism: the shift exists because the classical bank
        # runs at NATIVE resolution while the head was trained on features capped at ``max_side``. What
        # matters is therefore whether that cap downscales the image AT ALL, not any absolute pixel count.
        # So the bound is the training cap itself: at or below it, "native" and "capped" are the same image
        # and the shift is zero BY CONSTRUCTION. No fitted number survives, and the rule states what happens
        # on a dataset nobody has seen. That the fitted 1500 landed within 2.4% of the architectural 1536 is
        # evidence it was approximating exactly this quantity; ``scripts/gate_constants_probe.py`` confirms
        # the two rules make the IDENTICAL decision on all 15 registry datasets, so this is a change of
        # justification and generality, not of measured behaviour.
        #
        # ``None`` = derive from the cap. An explicit value is still honoured, for ablation only.
        self.thin_max_side = thin_max_side
        self._thin_active = None
        # ADAPTIVE COLOUR CHANNEL: the classical bank was grayscale-only, discarding the most discriminative
        # pixel signal on stained/colour modalities (nuclei = the hematoxylin channel in H&E; retinal vessels
        # peak-contrast in green). Self-configuring: at fit, pick the single channel (gray / R / G / B /
        # hematoxylin / eosin) that maximises fg/bg Fisher separability over the K support masks, and feed the
        # WHOLE bank (Frangi/Sauvola/structure/identity) that channel. Defaults to 'gray' unless a colour
        # channel beats it by `color_margin` → monochrome data (fluorescence/phase-contrast) stays
        # byte-identical (no regression). Decided ONCE per dataset at first fit, like the CLAHE/thin gates.
        self.color_adaptive = color_adaptive
        self.color_margin = color_margin
        self._contrast_source = None
        # TUBULARITY-GATED bank-unfreeze: unfreeze the ~dozen Frangi/Sauvola bank params ONLY when the support
        # is thin/tubular (mean tubularity > thin_ref) — helps filaments (microtubules +0.031, scales with K)
        # but is neutral-to-slightly-negative on blobs/vessels, so gate it (same descriptor as thin_adaptive).
        self.bank_unfreeze_adaptive = bank_unfreeze_adaptive
        # EXPANDED classical bank (roadmap #8): append a MODEST curated set of extra native priors
        # (white/black top-hat at 2 small scales + a local-std texture cue, `_extra_bank_features`) to the
        # frozen HyperBank output — small bright/dark structures + histopathology texture the ridge/blob
        # banks under-serve. Kept modest (K≈8 → too many channels overfit the 1×1 fusion). Default False ⇒
        # the bank is byte-for-byte the `head_fusion_best` bank and n_classical is unchanged. This changes
        # only WHAT is in the bank; the thin_adaptive size-gate that decides WHEN the native bank is used is
        # untouched. n_classical AUTO-propagates from the actual bank output width at fit (see `_classical`).
        self.bank_extra = bank_extra
        # COMPETITIVE GATE: per-pixel softmax competition over the fusion groups (coarse/fine/classical) in the
        # head — a confident group dominates instead of a static linear sum. Zero-init → starts at parity.
        self.competitive_gate = competitive_gate
        # CORRESPONDENCE PRIOR: append a support-derived cos(feat,fg_proto)−cos(feat,bg_proto) channel to the
        # classical priors (self-configuring fg signal; targets the fg bottleneck). Prototypes built at fit.
        self.corr_prior = corr_prior
        # SUPPORT-CONDITIONED FiLM: a tiny hypernetwork in the head maps the SAME support fg/bg prototypes
        # (s=[fg_proto⊕bg_proto]) → per-channel (γ,β) that modulate the penultimate before the 1×1 classifier,
        # so the fusion self-adapts to the dataset from the K support masks. Zero-init final layer → identity at
        # init (starts at parity, sharpens from the labels). Reuses the corr_prior prototype machinery.
        self.film = film
        self._fg_proto = None
        self._bg_proto = None
        # MULTI-PROTOTYPE correspondence (Lever 1): when n_proto>1 the corr channel uses k-means
        # centroids of the support fg/bg patch features with a MAX-POOLED cosine (max_k cos − max_j cos)
        # instead of the single averaged prototype (diluted on appearance-varied fg, e.g. H&E nuclei).
        # n_proto==1 → centroid stacks stay None → byte-identical single-prototype channel (parity).
        # The mean prototypes above are STILL built (FiLM conditions on them, unchanged).
        self.n_proto = n_proto
        self._fg_protos = None
        self._bg_protos = None
        if n_proto > 1 and not corr_prior:   # centroids ONLY feed the corr channel — else silently unused
            raise ValueError("n_proto>1 requires corr_prior=True (multi-prototype feeds the corr channel)")
        # BOUNDARY-DoU loss term (Lever 2): a self-scaling boundary loss added to the adaptive menu, gated by
        # instance-density × COMPACTNESS (1−thinness) — fires on dense COMPACT touching blobs (nuclei) where fg
        # over-prediction/bleed merges neighbours (the monuseg diagnostic: precision 0.68 vs recall 0.89), and is
        # damped on thin/tubular structures (filaments/vessels, which need clDice topology, not boundary
        # sharpening). Default False → the loss is byte-identical to best_v2.
        self.boundary_dou = boundary_dou
        # PRECISION-Tversky (Lever 3): a Tversky term with α>β (penalise FALSE POSITIVES harder than false
        # negatives) gated to dense COMPACT instances — the mirror of the existing recall-favouring Tversky
        # (which is imbalance-gated). Targets the monuseg diagnostic's DIFFUSE INTERIOR over-prediction
        # (precision 0.68 vs recall 0.89): reducing fg bleed un-merges touching nuclei → lifts instance-AP.
        # Novel framing: the constructor self-configures the FP/FN asymmetry from MORPHOLOGY. Default False.
        self.prec_loss = prec_loss
        self._color_dataset = False   # set at fit: is this a colour modality (decides per-image candidate set)
        # SAM-FREE affinity-watershed instance decoder (instance_mode="affinity"): geometry (computed DT of
        # the fg mask) + classical ridge split touching cells; a DINO-affinity merge-veto removes over-splits.
        # Two scalars calibrated from the K support at fit (median instance radius + same-class merge floor).
        self._inst_r = None
        self._inst_merge_cos = None
        self.guided_fuse = guided_fuse  # C: light separable guided-fusion block before the classifier
        self.boundary_head = boundary_head  # W2: learned inter-instance boundary → split touching
        self.contrast_norm = contrast_norm  # W1: CLAHE-style local contrast norm before the bank
        self.cldice = cldice                # thin-structure: soft-clDice topology loss term
        self.scale_fusion = scale_fusion and dino_scale == "both"  # coarse/fine ablation → single DINO branch
        self.encoder = getattr(encoder, "enc", encoder)  # raw encoder for the fine (native-tiled) branch
        self.fine_max_grid = fine_max_grid  # cap fine feature-grid side (memory)
        self.upsampler = upsampler          # learned upsampler for coarse/fine (None=bilinear)
        # "wide" reproduces every reported number; "lean" makes the first convolution 1x1,
        # which removes 76% of the head's parameters (see DINOHeadFusion). Must be A/B'd.
        self.stem = stem
        self.depth = depth              # hidden layers in the stem (2 = reported)
        self.hidden_film = hidden_film  # FiLM hypernet width (64 = reported)
        self.eca = eca                  # ECA channel attention on the raw backbone grid
        # TRAINING BUDGET AND WHAT REPLACES IT. With weight_decay=0, no dropout and only a
        # horizontal flip, the fixed 60-step budget IS this head's regulariser: the only thing
        # stopping 3.1M parameters from memorising 8 images is that training ends early. The logged
        # loss curve shows that budget also BINDS (still descending at 60, and at 120), so it is
        # simultaneously under-training and over-regularising, which is why it cannot simply be
        # raised. `early_stop` lets the fit run to convergence instead, and `weight_decay` supplies
        # the explicit regularisation that the truncation was doing implicitly -- the two belong
        # together, and a low parameter count (see `stem`) is the third leg.
        self.early_stop = early_stop
        # A plateau test comparing two windows of `patience` needs 2*patience steps before it
        # can fire at all. With the reported 60-step budget and patience 25 that leaves ten
        # steps in which to trigger, so `es` alone would return a null that measures the
        # window rather than early stopping. Fail rather than produce that null.
        if early_stop and epochs < 3 * patience:
            raise ValueError(f"early_stop with epochs={epochs} and patience={patience}: the "
                             f"plateau test cannot fire before step {2 * patience}, so this "
                             f"would measure the window. Raise the budget or lower patience.")
        self.patience = patience              # window length for the plateau test, in steps
        self.min_rel_improve = min_rel_improve  # relative loss gain per window below which we stop
        self.weight_decay = weight_decay      # >0 switches Adam -> AdamW
        self.dropout = dropout                # channel dropout in the stem
        self.se = se                          # SE reduction ratio (0 = off)
        self.mix = mix                        # nonlinear 1x1 mixer after the concat
        self.bilinear = bilinear              # low-rank cross-source interaction rank
        # BANK CHANNEL SCALE. The 35 classical channels reach only a 1x1 linear classifier, with no
        # normalisation anywhere on that path -- unlike the DINOv3 path, which has GroupNorm inside
        # its stem. Measured on DRIVE, their per-channel standard deviations span 7.6e-4 to 1.7e0,
        # a factor of 2229. At init the high-variance channels therefore dominate entirely and the
        # weakest contribute ~2000x less signal, which the classifier would have to undo by learning
        # weights three orders of magnitude apart -- in 60 full-batch steps.
        #
        # The scale is derived ONCE from the SUPPORT set rather than per image. Per-image
        # standardisation would equalise the channels but destroy cross-image comparability: "this
        # image has strong ridges overall" is real information the bank carries in its magnitudes.
        # A support-derived divisor equalises the channels while leaving each image's relative
        # magnitude intact, and needs no learned parameters -- the same closed-form, support-driven
        # shape as the colour rule.
        if aug not in ("flip", "d4"):
            raise ValueError(f"unknown aug {aug!r}; expected 'flip' (reported) or 'd4'")
        self.aug = aug
        # INPUT INTENSITY NORMALISATION, the single lever nnU-Net's own ablations credit most.
        # nnU-Net z-scores every image by statistics derived from the training data; on the chosen
        # channel we do the analogue from the SUPPORT foreground: subtract the median foreground
        # intensity and divide by its IQR, so the classical bank sees the object at a consistent
        # level across images whose staining or illumination differs (H&E, retinal fundus). Derived
        # once at fit, applied identically to support and query; zero learned parameters, the same
        # closed-form, support-driven shape as the colour rule. Distinct from banknorm, which scales
        # the bank OUTPUT channels -- this scales the bank INPUT.
        self.znorm = znorm
        self._znorm = None                    # (centre, scale) on the chosen channel, from support
        self.banknorm = banknorm
        self._bank_scale = None
        self.batch_size = batch_size        # minibatch SGD: bounds GPU mem (no OOM) + more grad steps
        self.amp = amp                      # bf16 autocast + bf16 input storage (½ mem, ~2× speed)
        self.tile_batch = 6                 # fine-branch: tiles encoded per model forward (speed)
        self.head = None
        self._bank = None
        self._n_classical = None
        self._ccache = {}  # classical features per image (id-keyed) — reused across AL-loop refits
        self._fcache = {}  # fine feature grids per image (id-keyed)
        # Constructor values of the flags that the support-derived gates MUTATE in place (thin_adaptive
        # flips tile_classical/fine_scales; bank_unfreeze_adaptive flips trainable_classical).
        # Snapshotted so `reset_support_state` can restore them for a fresh support draw.
        # corr_prior/film are ALSO latched off (permanently) when a support draw is degenerate, and
        # `fit` guards prototype construction on `(corr_prior or film)` — so without restoring them one
        # unlucky draw silently disables FiLM for every LATER seed and the reported mean mixes two
        # different methods. They belong in the snapshot exactly like the tiling flags.
        self._ctor_flags = dict(tile_classical=self.tile_classical, fine_scales=self.fine_scales,
                                trainable_classical=self.trainable_classical,
                                corr_prior=self.corr_prior, film=self.film)

    def reset_support_state(self) -> None:
        """Drop EVERY piece of state derived from a previous support draw, so the next ``fit`` re-runs
        the self-configuration from scratch. Leaves the object indistinguishable from a freshly
        constructed one, except for ``_fcache`` (see below).

        Why this exists: each self-configuring gate latches on ``is None`` and is therefore decided at
        the FIRST ``fit`` only. A benchmark that builds the backend once and loops over seeds (the
        multi-draw protocol) consequently ran seeds 1..N with seed 0's colour channel, CLAHE strength,
        thin gate, affinity calibration and FiLM prototypes. That understates the across-seed variance
        of the self-configuration step and makes the method self-configure from the FIRST draw rather
        than from each draw. Callers that re-fit on a NEW support set must call this first. An active-
        learning loop that deliberately GROWS one support set must NOT — there the carry-over is intended.

        Cache handling is not symmetric: ``_bank`` bakes in ``fine_scales`` (Sauvola window set),
        ``trainable_classical`` (``requires_grad``) and, with the scaleconf lever, ``_bank_scales``; and
        ``_ccache`` keys on the contrast channel but NOT on ``_clahe_strength``, ``_bank_keep``
        (bankselect zeroing is applied before an entry is stored) or ``_bank_scales``, so both can serve
        values inconsistent with a re-derived configuration and are dropped. ``_fcache`` holds fine DINO grids that depend only on the encoder and the image, so
        it is kept — it is by far the expensive one and cannot go stale here.
        """
        self.head = None
        self._n_cls = 1
        self._n_classical = None          # re-derived from the bank output width at the next fit
        self._clahe_strength = None
        self._thin_active = None
        self._contrast_source = None
        self._color_dataset = False
        self._fg_proto = self._bg_proto = None
        self._fg_protos = self._bg_protos = None
        self._inst_r = self._inst_merge_cos = None
        self._bank_keep = None            # re-derive the bankselect Fisher channel selection from each draw
        self._bank_scale = None           # re-derive the per-channel bank scale from each draw
        self._znorm = None                # re-derive the support intensity normalisation each draw
        self._bank_scales = None          # re-derive the scaleconf bank scales from each draw (rebuilds _bank)
        for name, value in self._ctor_flags.items():   # undo the gates' in-place side effects
            setattr(self, name, value)
        self._bank = None                              # rebuilt under the restored flags
        self._ccache.clear()

    def _support_scales(self, support):
        """LEVER (scaleconf): derive the bank's Frangi/Sauvola/LoG scales from the support masks' own object
        radii. Takes the medial-axis distance transform (the DT ridge = local object half-thickness) of each
        support foreground, pools the radii, and clusters them (``cluster_scales``) into centroids -- one bank
        scale per centroid. Sets ``self._bank_scales`` (px); ``_bank_module`` then builds the bank at them."""
        radii = []
        # Resolve the radius DEFINITION once, before the loop. Deciding it per image (a try/except
        # inside the loop) meant a failure on SOME masks silently pooled medial-axis half-widths with
        # EDT top-quartile radii -- systematically different quantities, the EDT one skewing larger --
        # so the clusters, and hence every bank scale, would follow which images happened to fail.
        # Now the whole support shares one definition and a genuine per-mask failure propagates.
        try:
            from skimage.morphology import medial_axis
        except ImportError:
            medial_axis = None
            print("    [scaleconf] skimage medial_axis not installed → EDT top-quartile radius for the "
                  "WHOLE support (coarser, but one consistent definition)", flush=True)
        for ex in support:
            m = np.asarray(ex.label_map) > 0
            if not m.any() or m.all():                           # degenerate mask: no radius to read
                continue
            if medial_axis is not None:
                skel, dist = medial_axis(m, return_distance=True)
                radii.extend(dist[skel].tolist())
            else:
                from scipy.ndimage import distance_transform_edt
                dt = distance_transform_edt(m)
                pos = dt[dt > 0]
                if pos.size:
                    radii.extend(dt[dt >= np.percentile(pos, 75)].tolist())
        if not radii:
            print("    [scaleconf] no usable support masks → keeping the default fixed bank scales", flush=True)
            return
        # Clamp to a sane band. LOWER 1.5 is the load-bearing one: HyperBank._resolve_sigma reads a sigma
        # BELOW 1.0 as a FRACTION of the reference size, so a support-derived centroid of 0.9 would not
        # be a fine scale at all -- it would silently become sigma ~= 0.9*672 ~= 600 px and blow up
        # reflect-pad. cluster_scales legitimately returns sub-pixel centroids on filaments, so do NOT
        # lower this below 1.0 to chase thin structures; 1.5 keeps a margin above that switch.
        # Upper 64 (4x the fixed grid's max σ=16) lets scaleconf adapt to fairly large
        # objects while capping the pathological case: a σ≈400 Frangi on a huge spheroid is a ~3000-tap kernel
        # AND is wasted (Frangi suppresses blob-like responses by design), and big objects are already the
        # DINO backbone's domain — the classical bank earns its keep on the fine detail the coarse grid misses.
        self._bank_scales = sorted({round(float(np.clip(s, 1.5, 64.0)), 1) for s in cluster_scales(radii)})
        print(f"    [scaleconf] {len(self._bank_scales)} scale cluster(s) from support object radii "
              f"→ bank scales {self._bank_scales} px (n_radii={len(radii)})", flush=True)

    def _bank_module(self):
        if self._bank is None:
            from active_segmenter.segment.hyperbank_bank import HyperBank

            if self.scaleconf and self._bank_scales:             # LEVER: scales from the support object radii
                if self.fine_scales:                             # thin_adaptive requested the fine Sauvola set;
                    print("    [scaleconf] note: thin_adaptive fine_scales overridden by support-derived windows",
                          flush=True)                            # make the override observable (no silent conflict)
                c = [float(x) for x in self._bank_scales]        # cluster centroid radii (px)
                sauv = tuple(sorted({max(3, int(2 * round(x)) | 1) for x in c}))  # window ≈ object diameter (odd)
                self._bank = HyperBank(
                    frangi_sigmas=tuple(c),                      # Frangi σ ≈ structure half-width = radius
                    sauvola_windows=sauv, struct_sigmas=(2.0, 8.0),
                    use_log=True, log_sigmas=tuple(c),           # LoG σ ≈ blob radius
                    use_morph_orient=self.morph_orient,
                ).to(self.device).eval()
            else:
                # W1: finer scales for sub-10px faint crumbs. NB the bank reads a Frangi σ ≤ 1.0 as a
                # FRACTION of min(H,W) (a giant kernel), so a "finer" Frangi <1 blows up reflect-pad —
                # 2.0 is already the finest absolute Frangi scale. The viable finer knob is a tighter
                # Sauvola binarization window (absolute px) for the smallest crumbs.
                sauvola = (7, 15, 51, 151) if self.fine_scales else (15, 51, 151)
                self._bank = HyperBank(
                    frangi_sigmas=(1.0, 2.0, 4.0, 8.0, 16.0), sauvola_windows=sauvola,
                    struct_sigmas=(2.0, 8.0), use_log=True,
                    use_morph_orient=self.morph_orient,
                ).to(self.device).eval()
            for p in self._bank.parameters():
                p.requires_grad_(self.trainable_classical)
        return self._bank

    def _choose_contrast_source(self, support) -> None:
        """Pick the channel maximising mean fg/LOCAL-bg Fisher separability over the K support masks (decided
        ONCE per dataset at fit, like the CLAHE/thin gates). Stays ``gray`` unless a colour channel beats it
        by ``color_margin`` → monochrome data is unaffected. Self-configuring; no per-dataset hardcoding.

        The background is a DILATED RING around each fg structure, not the global background: the classical
        bank is a stack of LOCAL operators (Frangi ridge / Sauvola adaptive-threshold), so the channel that
        matters is the one with the best LOCAL fg-vs-surround contrast. A global background lets a far-field
        artefact dominate — e.g. DRIVE's black FOV border made the bright red channel win spuriously over the
        textbook green vessel channel."""
        # Decide MODALITY per-dataset (not per-image): colour iff a majority of support images are genuinely
        # colour (detection mode). Then retrieve candidates with force=True for EVERY image so gray and the
        # colour channels are averaged over the SAME image set (unbiased) and the chosen source is available
        # on every image at inference (no per-image mono classification can crash a homogeneous run).
        n_color = sum(1 for ex in support if len(_color_channels(ex.image)) > 1)
        self._color_dataset = n_color > len(support) / 2
        if not self._color_dataset:
            self._contrast_source = "gray"
            print("    [color_adaptive] dataset is monochrome → source=gray", flush=True)
            return
        scores, n_used = channel_separability(support)
        if n_used == 0:                                            # every support mask all-fg/all-bg
            self._contrast_source = "gray"
            print("    [color_adaptive] no usable support masks (all fg/bg) → source=gray", flush=True)
            return
        # A channel is eligible only if scored on EVERY used support image — else its average is over a biased
        # subset (e.g. rgb2hed silently failing on the hard tiles would inflate hematoxylin vs gray). Use the
        # MEDIAN (not mean) so one degenerate image can't dominate. gray/R/G/B always cover all images.
        self._contrast_source, means = colour_gate(scores, n_used, self.color_margin)
        ranked = ", ".join(f"{k}:{v:.2f}" for k, v in sorted(means.items(), key=lambda kv: -kv[1]))
        print(f"    [color_adaptive] fg/bg separability(median) {{{ranked}}} → source={self._contrast_source}",
              flush=True)

    def _calibrate_instance(self, support) -> None:
        """Calibrate the affinity-watershed decoder from the K support masks (decided once at fit): the
        median instance RADIUS (marker separation / DT-peak spacing) and the same-class MERGE floor — the
        90th-pct cosine between adjacent GT instances' mean DINO features, so the merge-veto only fuses
        basins MORE similar than typical true-different-instance neighbours. Falls back to safe defaults on
        degenerate support. No dense field is learned.

        These TWO scalars are the only support-DERIVED state. The decoder is not self-configuring end to
        end: ``_affinity_watershed_instances`` also takes ``merge_ridge`` (the ridge height above which the
        veto refuses to merge), ``w_dt``/``w_ridge`` and ``min_area``, and ``predict`` passes none of them,
        so they stay at their fixed module defaults on every dataset. Describe the decoder as
        support-calibrated in its marker spacing and merge floor, NOT as fully self-configured."""
        try:
            from skimage.graph import RAG
        except ImportError:
            from skimage.future.graph import RAG

        radii, across_cos, ninst = [], [], []
        for ex in support:
            lm = np.asarray(ex.label_map)
            ninst.append(int(lm.max()))
            if lm.max() < 1:
                continue
            areas = np.bincount(lm.ravel().astype(np.int64))  # per-instance areas (O(H·W), robust to sparse ids)
            radii.extend(((areas[1:][areas[1:] > 0]) / np.pi) ** 0.5)
            if int(lm.max()) >= 2 and getattr(ex, "feat_grid", None) is not None:
                means = _basin_mean_features(lm.astype(int), ex.feat_grid)
                try:
                    for a, b in RAG(lm.astype(int)).edges:
                        if a > 0 and b > 0 and a in means and b in means:
                            across_cos.append(float(np.dot(means[a], means[b])))
                except Exception:
                    pass
        # guard: the affinity decoder needs PER-INSTANCE support masks. A binary/semantic mask reads as ONE
        # giant "instance" → r* = sqrt(total_fg_area/π) → every test image collapses to ~1 object (the same
        # class of bug as the _detect_n_classes regression CLAUDE.md records). Fail loud, don't silently ruin AP.
        if not radii or float(np.median(ninst)) <= 1.0:
            # SEMANTIC/binary support (drive/hrf/microtubules/…): the affinity instance decoder is not
            # applicable and is NEVER scored on these (the metric is fg-IoU/clDice via foreground(), not
            # predict()). SKIP calibration gracefully so a UNIVERSAL method (affinity on) still runs here. A
            # genuine misconfig (instance-AP dataset fed binary masks) still fails loud later — predict()
            # raises "uncalibrated" because _inst_r stays None.
            print(f"    [affinity] binary/semantic support (median max-label={np.median(ninst) if ninst else 0:.0f})"
                  f" → instance decoder inactive here (not scored on this dataset)", flush=True)
            return
        # clamp r* to a sane pixel band (CRITICAL: a support/test scale shift otherwise silently mis-seeds)
        self._inst_r = float(np.clip(np.median(radii), 2.0, 60.0)) if radii else 5.0
        if len(across_cos) >= 5:
            self._inst_merge_cos = float(np.clip(np.percentile(across_cos, 90), 0.85, 0.995))
        else:  # too few touching support instances to calibrate → CONSERVATIVE (rarely merge; over-seg beats
            self._inst_merge_cos = 0.97   # under-seg for AP). Loud so an uncalibrated veto is visible.
            print("    [affinity] WARNING: <5 adjacent support pairs → merge-veto UNCALIBRATED, using "
                  "conservative merge_cos=0.97", flush=True)
        print(f"    [affinity] calibrated r*={self._inst_r:.1f}px merge_cos={self._inst_merge_cos:.3f} "
              f"({len(radii)} instances, {len(across_cos)} adjacent pairs, ~{int(np.median(ninst))}/img)",
              flush=True)

    def _build_prototypes(self, support) -> None:
        """Build unit fg/bg feature prototypes from the K support masks (correspondence prior): the L2-normed
        mean of the grid patch features under the grid-downsampled fg / bg. Disables the prior on degenerate
        support. Prints the fg-vs-bg prototype cosine — a quick indicator of how separable the modality is."""
        import torch
        from skimage.transform import resize

        fg_sum, bg_sum, nf, nb = None, None, 0, 0
        fg_stack, bg_stack = [], []                                       # patches for multi-prototype k-means
        for ex in support:
            fgrid = np.asarray(ex.feat_grid, np.float32)                 # [G,G,D] unit
            G0, G1, D = fgrid.shape
            m = resize(np.asarray(ex.label_map) > 0, (G0, G1), order=0,
                       preserve_range=True, anti_aliasing=False).reshape(-1) > 0.5
            F = fgrid.reshape(-1, D)
            if m.any():
                fg_sum = F[m].sum(0) if fg_sum is None else fg_sum + F[m].sum(0); nf += int(m.sum())
                if self.n_proto > 1:
                    fg_stack.append(F[m])
            if (~m).any():
                bg_sum = F[~m].sum(0) if bg_sum is None else bg_sum + F[~m].sum(0); nb += int((~m).sum())
                if self.n_proto > 1:
                    bg_stack.append(F[~m])
        if fg_sum is None or bg_sum is None:
            print("    [corr_prior] degenerate support (no fg or bg patches) → prior/FiLM DISABLED", flush=True)
            self.corr_prior = False
            self.film = False            # degenerate prototypes → build the head without FiLM (→ parity)
            return
        fg, bg = fg_sum / max(nf, 1), bg_sum / max(nb, 1)
        fg, bg = fg / (np.linalg.norm(fg) + 1e-6), bg / (np.linalg.norm(bg) + 1e-6)
        self._fg_proto, self._bg_proto = torch.from_numpy(fg).float(), torch.from_numpy(bg).float()
        if self.n_proto > 1 and fg_stack and bg_stack:                   # multi-prototype corr channel (Lever 1)
            from active_segmenter.segment.multiproto import kmeans_protos
            fgp = kmeans_protos(np.concatenate(fg_stack), self.n_proto)
            bgp = kmeans_protos(np.concatenate(bg_stack), self.n_proto)
            self._fg_protos = torch.from_numpy(fgp).float()
            self._bg_protos = torch.from_numpy(bgp).float()
            print(f"    [corr_prior] multi-prototype: {fgp.shape[0]} fg + {bgp.shape[0]} bg centroids "
                  f"(n_proto={self.n_proto})", flush=True)
        print(f"    [corr_prior] prototypes built (fg {nf} / bg {nb} patches); fg·bg cos={float(np.dot(fg, bg)):.3f} "
              f"(lower = more separable)", flush=True)

    def _fit_znorm(self, support) -> None:
        """Centre and scale for the chosen input channel, from the SUPPORT foreground.

        Median and inter-quartile range of the foreground intensities, pooled across the K support
        images -- robust to the outliers a plain mean/std would chase in a few masks. nnU-Net derives
        the analogue from a whole dataset; here the K masks are all there is, and pooling their
        foreground pixels is the closest support-only equivalent.
        """
        vals = []
        for ex in support:
            ch = self._raw_channel(ex.image)
            m = np.asarray(ex.label_map) > 0
            if m.shape != ch.shape:
                from skimage.transform import resize
                m = resize(m, ch.shape, order=0, preserve_range=True) > 0.5
            if m.any():
                vals.append(np.asarray(ch)[m].ravel())
        if not vals:
            self.znorm = False                 # degenerate support: leave the input untouched
            return
        v = np.concatenate(vals)
        centre = float(np.median(v))
        iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
        # A near-constant foreground gives IQR ~0; dividing by it would blow noise up into the
        # largest input the bank sees. Fall back to unit scale there rather than amplify.
        self._znorm = (centre, iqr if iqr > 1e-4 else 1.0)
        print(f"    [znorm] support foreground centre={centre:.3f} iqr={self._znorm[1]:.3f}", flush=True)

    def _raw_channel(self, image) -> np.ndarray:
        """The chosen channel WITHOUT znorm -- used by _fit_znorm to measure the raw statistics."""
        src = self._contrast_source or "gray"
        if src == "gray":
            return _gray01(image)
        ch = _color_channels(image, force=self._color_dataset)
        return ch[src] if src in ch else _gray01(image)

    def _apply_znorm(self, arr):
        """z-score the chosen channel by the support-derived (centre, iqr). Identity when off."""
        if self._znorm is None:
            return arr
        c, s = self._znorm
        return ((np.asarray(arr, np.float32) - c) / s)

    def _channel(self, image) -> np.ndarray:
        """The single channel fed to the classical bank: the support-chosen contrast source (default gray).
        Raises on a train/inference mismatch (chosen source unavailable for this image) rather than silently
        falling back to gray — a silent channel swap between fit and inference is the tiling-regression bug class."""
        src = self._contrast_source or "gray"
        if src == "gray":
            return _gray01(image)
        ch = _color_channels(image, force=self._color_dataset)     # colour dataset → full set on every image
        if src not in ch:
            raise RuntimeError(f"color_adaptive picked source={src!r} at fit but this image lacks it "
                               f"(available {sorted(ch)}) — train/inference channel mismatch")
        return ch[src]

    def _append_rgb(self, feats, image):
        """A/B simplification test: append the raw R, G, B channels (native, [0,1]), resized to the
        classical-feature grid, as extra native features so the head/gate weight colour themselves,
        instead of the closed-form colour-channel selection. Mono images replicate grayscale."""
        import torch
        ch = _color_channels(image, force=True)
        chans = ([ch[k] for k in ("R", "G", "B")] if all(k in ch for k in ("R", "G", "B"))
                 else [_gray01(image)] * 3)
        hw = feats.shape[-2:]
        outs = []
        for c in chans:
            t = torch.from_numpy(np.asarray(c, np.float32))[None, None].to(feats.device)
            if t.shape[-2:] != hw:
                t = torch.nn.functional.interpolate(t, size=hw, mode="bilinear", align_corners=False)
            outs.append(t)
        return torch.cat([feats] + outs, dim=1)

    def _classical(self, image, grad: bool = False, inference: bool = False):
        """Classical per-pixel feature stack → [1, C, h, w] tensor. Training uses a capped native
        res (``max_side``) for tractability; ``inference=True`` with ``tile_classical`` computes the
        bank at FULL native resolution tile-by-tile (Improvement A — no res cap → thin vessels
        survive), reusing the res-agnostic head. ``grad=True`` trains the filter params."""
        import torch

        tiled = inference and self.tile_classical and max(np.asarray(image).shape[:2]) > self.tile
        gup = inference and self.classical_up and not tiled  # cheap low-res bank + guided upsample
        # Key includes the chosen channel so a per-seed re-selection cannot serve stale features. It does
        # NOT include `_bank_keep` (bankselect) or `_bank_scales` (scaleconf), which also change these
        # maps — correctness there rests entirely on `reset_support_state` clearing `_ccache` between
        # draws, so that clearing is load-bearing. Add them to the key if either can ever change WITHIN
        # a single fit.
        cache_key = ((id(image), self._contrast_source)
                     if (not grad and not tiled and not gup) else None)  # capped path is stable
        if cache_key is not None and cache_key in self._ccache:
            import torch
            return self._ccache[cache_key].to(self.device)
        bank = self._bank_module()
        g = self._channel(image)  # support-chosen contrast channel (default grayscale; color_adaptive picks)
        if self.contrast_norm:  # W1: CLAHE local-contrast enhancement for faint low-contrast objects
            # adaptive: blend by the per-dataset strength (0=off on distinct objects, 1=full on faint)
            s = self._clahe_strength if (self.clahe_adaptive and self._clahe_strength is not None) else 1.0
            if s > 0:
                try:
                    from skimage.exposure import equalize_adapthist
                    gc = equalize_adapthist(g, clip_limit=0.01).astype(np.float32)
                    g = gc if s >= 1.0 else ((1.0 - s) * g + s * gc).astype(np.float32)
                except Exception as e:
                    # CLAHE was GATED ON for this dataset (faint objects) — a silent skip would feed the
                    # head an unnormalised input it wasn't trained on. Surface it (train/inference mismatch
                    # is the class of bug that made tiling regress) rather than swallowing it.
                    print(f"    [clahe] SKIPPED (strength={s:.2f}): {e!r} — proceeding unnormalised",
                          flush=True)
        # z-score LAST, after CLAHE: CLAHE is a contrast enhancement that needs a [0,1] input, and
        # z-norm is a standardisation of the result. Reversing them fed CLAHE a z-scored channel it
        # rejects, silently disabling it and confounding the znorm A/B with "znorm minus CLAHE".
        g = self._apply_znorm(g)
        h, w = g.shape
        if tiled:
            feats = self._classical_tiled(bank, g)
        else:
            cap = self.classical_lowres_side if gup else self.max_side  # gup computes at a lower cap
            scale = min(1.0, cap / max(h, w))
            t = torch.from_numpy(g)[None, None].to(self.device)
            if scale < 1.0:
                t = torch.nn.functional.interpolate(t, scale_factor=scale, mode="bilinear",
                                                    align_corners=False)
            ctx = torch.enable_grad() if grad else torch.no_grad()
            with ctx:
                feats = bank.feature_maps(t)
            if gup and feats.shape[-2:] != (h, w):  # image-GUIDED upsample of the cheap bank → native
                from active_segmenter.segment.upsamplers import fast_guided_upsample
                guide = torch.from_numpy(g)[None, None].to(self.device)
                with torch.no_grad():
                    feats = fast_guided_upsample(feats, guide)
        if self.bank_extra:                          # roadmap #8: curated extra classical priors (top-hat + texture)
            feats = self._append_extra(feats, g)     # → n_classical AUTO-picks up the wider count just below
        if self.rgb_feat:                            # A/B: raw RGB channels as native features (vs colour selection)
            feats = self._append_rgb(feats, image)   # → n_classical AUTO-picks up the +3 count below
        if self._n_classical is None:
            self._n_classical = int(feats.shape[1])
        if self.dino_only:                               # ABLATION: DINO features only — zero the classical
            import torch                                 # bank so it carries no information (shapes preserved;
            feats = torch.zeros_like(feats)              # head is trained + tested on zeros = pure DINO head)
        if self.bankselect and self._bank_keep is not None:   # LEVER: zero the low-Fisher bank channels (soft
            if len(self._bank_keep) != feats.shape[1]:        # discard; shape preserved so the head is unchanged).
                raise RuntimeError(                           # post-reset the mask is always re-derived for the
                    f"bankselect keep-mask width {len(self._bank_keep)} != bank width {feats.shape[1]} — "
                    f"a should-never-happen invariant violation; refusing to silently fall back to baseline")
            keep = torch.as_tensor(self._bank_keep, device=feats.device, dtype=feats.dtype)
            feats = feats * keep.view(1, -1, 1, 1)
        if self._bank_scale is not None:
            # BEFORE the cache write, which is the only correct order. `_classical` has TWO exits --
            # this one and the cache-hit return above -- and dividing after the write meant cached
            # maps stayed raw while fresh ones came back scaled. `_fit_bank_scale` itself populates
            # the cache for every support image while the scale is still None, so the head then
            # trained on raw maps and predicted on scaled ones, and even disagreed with itself
            # between the first and second call on one query. Scaling here makes the cache hold what
            # every consumer should see; `_fit_bank_scale` drops the raw entries it created.
            feats = feats / self._bank_scale.to(feats.dtype).view(1, -1, 1, 1)
        if cache_key is not None:
            self._ccache[cache_key] = feats.detach().cpu()  # cache on CPU (grows with pool, not GPU)
        return feats

    def _fit_bank_scale(self, support) -> None:
        """Per-channel divisor for the classical bank, derived once from the support masks.

        Runs BEFORE any query is scored and is stored on the backend, so support and query maps are
        divided by the same constants -- a scale re-derived per image would make the two
        incomparable. Uses the median of the per-image standard deviations rather than the mean, so
        one atypical support image cannot set the scale for the draw.
        """
        import torch

        sds = []
        for ex in support:
            with torch.no_grad():
                C = self._classical(ex.image)
            sds.append(C.float().std(dim=(0, 2, 3)).cpu())
        med = torch.stack(sds).median(dim=0).values
        # A channel that is constant across the whole support carries no information for this draw;
        # dividing by its ~0 std would amplify pure numerical noise into the largest input the
        # classifier sees. Leave those channels alone instead.
        self._bank_scale = torch.where(med > 1e-6, med, torch.ones_like(med)).to(self.device)
        # Computing the scale required reading every support image through `_classical`, which cached
        # them UNSCALED because the scale did not exist yet. Those entries must go, or training reads
        # raw maps from the cache while inference recomputes scaled ones.
        self._ccache.clear()
        r = float(med[med > 1e-6].max() / med[med > 1e-6].min()) if (med > 1e-6).any() else 1.0
        print(f"    [banknorm] per-channel scale from {len(support)} support image(s); "
              f"raw std spread {r:.0f}x -> 1x", flush=True)

    def _bank_separability(self, support):
        """LEVER (bankselect): Fisher-select the classical bank channels from the support masks -- the same
        FORM of fg-vs-local-bg separability as the colour-channel selection (``channel_separability``),
        extended from the input channel to the whole bank, so one rule covers both instead of per-filter
        gates. It is a SEPARATE implementation, not a shared helper, and two divergences are deliberate and
        must not be read as "identical to the colour probe":
          * the 8-iteration background ring is dilated in the CLASSICAL-MAP grid, which ``_classical`` may
            have downscaled (``max_side``), so on a large image it is a wider PHYSICAL surround than the
            colour probe's, which dilates at native resolution;
          * the ``1e-3 * range`` variance floor is copied from that function, but the justification given
            there ("every candidate arrives min-max normalised, so the term is in practice a constant")
            does NOT carry over: bank channels are not [0,1]-normalised, so here the floor is genuinely
            channel-relative.
        Keeps channels within ``bankselect_ratio`` of the most-separable channel (median over support) and
        zeros the rest. Runs while ``_bank_keep`` is still None so the probe reads UNFILTERED maps; it then
        clears the classical cache the probe warmed with those unfiltered maps so training re-filters."""
        from scipy.ndimage import binary_dilation
        per = []
        for ex in support:
            maps = self._classical(ex.image)                          # [1, Cc, h, w], unfiltered (keep is None)
            C = np.asarray(maps[0].detach().cpu().float(), np.float32)     # [Cc, h, w]
            m = np.asarray(ex.label_map) > 0
            if m.shape != C.shape[1:]:
                from skimage.transform import resize
                m = resize(m.astype(np.float32), C.shape[1:], order=0, mode="edge") > 0.5
            if not m.any() or m.all():
                continue
            ring = binary_dilation(m, iterations=8) & ~m              # local surround, matching the colour probe
            if ring.sum() < 10:
                ring = ~m
            sc = np.empty(C.shape[0], np.float32)
            for c in range(C.shape[0]):
                fg, bg = C[c][m], C[c][ring]
                denom = max(float(np.sqrt(fg.var() + bg.var())),
                            1e-3 * (float(C[c].max()) - float(C[c].min())), 1e-6)
                sc[c] = abs(float(fg.mean()) - float(bg.mean())) / denom
            per.append(sc)
        if not per:
            # FAIL LOUD, like the width-mismatch guard below. Printing and returning would leave
            # `_bank_keep` None, so `_classical` skips filtering entirely and this arm writes
            # BASELINE-IDENTICAL scores under the name `..._bankselect` -- the A/B would then compare
            # baseline against baseline and report "no effect", which is indistinguishable from a real
            # negative result and would go into the CONFIG-REGISTRY as one.
            raise RuntimeError("[bankselect] no usable support masks (every mask is all-fg or all-bg), "
                               "so no channel separability can be measured; refusing to run this arm "
                               "unfiltered under the bankselect name")
        med = np.median(np.stack(per, 0), axis=0)                     # [Cc] median separability over support
        # A NaN channel (Frangi/LoG on a degenerate or near-constant image) would poison the whole
        # selection: the threshold becomes NaN, EVERY comparison is False, and argmax() returns the NaN
        # index -- so the "never empty the bank" line would keep exactly the one broken channel and zero
        # all the good ones, with only a WARNING printed. Exclude non-finite channels from both the
        # threshold and the argmax instead.
        finite = np.isfinite(med)
        if not finite.any():
            raise RuntimeError("[bankselect] every bank channel scored a non-finite separability; "
                               "refusing to select a bank from it")
        if not finite.all():
            print(f"    [bankselect] {int((~finite).sum())} non-finite channel(s) excluded: "
                  f"{np.flatnonzero(~finite).tolist()}", flush=True)
        med = np.where(finite, med, -np.inf)
        top = float(med.max())
        if top <= 0:
            # Nothing separates fg from bg (e.g. a constant support image). The ratio rule would then
            # admit nothing and the argmax line would collapse the bank to one ARBITRARY channel.
            raise RuntimeError("[bankselect] no bank channel separates foreground from background on "
                               "this support (max median Fisher <= 0); refusing to collapse the bank")
        keep = med >= max(self.bankselect_ratio * top, 1e-9)
        keep[int(med.argmax())] = True                               # never empty the bank
        self._bank_keep = keep
        self._ccache.clear()                                         # probe warmed the cache UNFILTERED → drop it
        kept = np.flatnonzero(keep).tolist()
        print(f"    [bankselect] kept {int(keep.sum())}/{len(keep)} bank channels "
              f"(Fisher >= {self.bankselect_ratio:.2f} x max); kept={kept} median_sep={np.round(med, 2).tolist()}",
              flush=True)
        if int(keep.sum()) <= max(1, len(keep) // 8):                 # collapse from a small (K~8) median → flag it
            print(f"    [bankselect] WARNING: kept only {int(keep.sum())}/{len(keep)} channels — low-confidence "
                  f"selection from a small support; verify it is not run-noise", flush=True)

    def _append_extra(self, feats, gray):
        """Concatenate the curated EXTRA classical channels (``bank_extra``) onto the ``[1, C, h, w]`` bank
        output. The extras are computed at feats' OWN spatial resolution (resize the post-CLAHE gray to
        match) so their top-hat/texture footprints act at the SAME resolution the bank's absolute-px
        operators do — uniform across the capped / tiled / guided-upsample branches. The extras carry no
        grad (fixed priors like the frozen bank), so this is a plain concat in ``feats``'s dtype/device."""
        import torch
        from skimage.transform import resize

        fh, fw = int(feats.shape[-2]), int(feats.shape[-1])
        g = np.asarray(gray, np.float32)
        if g.shape != (fh, fw):
            g = resize(g, (fh, fw), order=1, mode="edge", anti_aliasing=True).astype(np.float32)
        ex = _extra_bank_features(g)                                     # [k, fh, fw] z-scored numpy
        ext = torch.from_numpy(ex)[None].to(feats.device, feats.dtype)   # [1, k, fh, fw]
        return torch.cat([feats, ext], dim=1)

    def _classical_tiled(self, bank, gray, overlap: int = 192):
        """Compute the classical bank at FULL native res on overlapping tiles, keep tile centres
        (filters have up to ~150 px footprint → overlap avoids seams). → [1, C, H, W]."""
        import torch

        h, w = gray.shape
        step = self.tile - overlap
        out = None
        ys = list(range(0, max(1, h - overlap), step)) or [0]
        xs = list(range(0, max(1, w - overlap), step)) or [0]
        with torch.no_grad():
            for y in ys:
                for x in xs:
                    y1, x1 = min(y + self.tile, h), min(x + self.tile, w)
                    y0, x0 = max(0, y1 - self.tile), max(0, x1 - self.tile)
                    t = torch.from_numpy(gray[y0:y1, x0:x1])[None, None].to(self.device)
                    f = bank.feature_maps(t)  # [1, C, th, tw]
                    if out is None:
                        out = torch.zeros((1, f.shape[1], h, w), device=self.device, dtype=f.dtype)
                    # write the interior (drop overlap/2 margins except at image border)
                    my0 = y0 + (overlap // 2 if y0 > 0 else 0)
                    mx0 = x0 + (overlap // 2 if x0 > 0 else 0)
                    out[:, :, my0:y1, mx0:x1] = f[:, :, my0 - y0:, mx0 - x0:]
        return out

    def _fine_feat(self, image) -> np.ndarray:
        """Fine (native-resolution) DINOv3 feature grid for the continuous scale-fusion branch: tile
        the image into encoder-resolution windows, encode each at NATIVE scale (no downscaling), and
        stitch the per-tile patch grids into one full-image grid (overlaps averaged). Capped to
        ``fine_max_grid`` per side for memory; id-cached across AL refits. This is the *mechanism* that
        obtains native detail — the crop-vs-context balance is then learned continuously by the head."""
        import torch
        from active_segmenter.acquire.crop_tiles import tile_grid

        key = id(image)
        if key in self._fcache:
            return self._fcache[key]
        a = np.asarray(image)
        h, w = a.shape[:2]
        res = int(getattr(self.encoder.cfg, "resolution", 672))
        ps = int(getattr(self.encoder.cfg, "patch_stride", 16))
        hg, wg = max(1, h // ps), max(1, w // ps)
        positions = list(tile_grid(h, w, res, 0.25))
        crops = []
        for (y, x) in positions:
            c = a[y:y + res, x:x + res]
            ph, pw = res - c.shape[0], res - c.shape[1]
            if ph or pw:
                c = np.pad(c, [(0, ph), (0, pw)] + ([(0, 0)] if a.ndim == 3 else []), mode="reflect")
            crops.append(c)
        grids = []                                                       # encode tiles in BATCHES
        for i in range(0, len(crops), max(1, self.tile_batch)):
            grids.extend(np.asarray(self.encoder.extract_batch(crops[i:i + self.tile_batch], res),
                                    np.float32))
        acc = cnt = None
        for (y, x), g in zip(positions, grids):
            gh, gw = g.shape[:2]
            if acc is None:
                acc = np.zeros((hg, wg, g.shape[-1]), np.float32)
                cnt = np.zeros((hg, wg, 1), np.float32)
            gy0, gx0 = min(y // ps, hg - 1), min(x // ps, wg - 1)
            gy1, gx1 = min(gy0 + gh, hg), min(gx0 + gw, wg)
            acc[gy0:gy1, gx0:gx1] += g[:gy1 - gy0, :gx1 - gx0]
            cnt[gy0:gy1, gx0:gx1] += 1.0
        grid = acc / np.maximum(cnt, 1e-6)
        m = max(hg, wg)
        if m > self.fine_max_grid:                                       # cap grid side (memory)
            t = torch.from_numpy(grid.transpose(2, 0, 1))[None]
            t = torch.nn.functional.interpolate(t, scale_factor=self.fine_max_grid / m,
                                                mode="bilinear", align_corners=False)
            grid = t[0].permute(1, 2, 0).contiguous().numpy()
        self._fcache[key] = grid
        return grid

    def _dino_x(self, image, feat_grid):
        """The DINO grid fed to the head as its coarse input X. dino_scale=='fine' swaps in the native-tiled
        fine grid (coarse dropped) — the ONLY-fine ablation; otherwise the passed coarse feat_grid."""
        if self.bank_only:                       # ABLATION: bank alone — zero the coarse DINO grid
            g = self._fine_feat(image) if (self.dino_scale == "fine" and self.encoder is not None) \
                else np.asarray(feat_grid)
            return np.zeros_like(np.asarray(g, np.float32))
        if self.dino_scale == "fine" and self.encoder is not None:
            return self._fine_feat(image)
        return np.asarray(feat_grid)

    def _fine_tensor(self, image):
        import torch

        if not self.scale_fusion or self.encoder is None:
            return None
        f = self._fine_feat(image)
        if self.bank_only:                       # ABLATION: zero the fine grid too, shape preserved
            f = np.zeros_like(np.asarray(f, np.float32))
        return torch.from_numpy(f.transpose(2, 0, 1))[None].to(self.device)

    def _ridge_init_classifier(self, head, items):
        """Solve the final 1x1 classifier in closed form, then let SGD start from there.

        WHY THIS IS EXACT AT INIT, NOT AN APPROXIMATION OF THE MODEL. The competitive gate and the
        FiLM modulation are both zero-initialised to an exact identity, so before the first optimiser
        step the head is a fixed feature extractor followed by one 1x1 convolution -- a linear model
        over the penultimate features. Its optimal weights under a logistic loss are therefore
        computable directly, and this is the same estimator (`closed_form.fit_irls`) the in-context
        experiments in this repository use.

        WHY IT SHOULD SAVE TIME. Profiling says a fit is `setup + E * epoch_cost`, and on the large
        fields E dominates: spheroidj runs about 274 epochs at 1.58 s while one epoch costs 1.58 s,
        so cutting E is worth roughly three times what cutting the epoch cost is worth. Starting the
        classifier at its closed-form optimum starts the loss curve lower, and the plateau rule --
        which compares a 25-epoch window against the previous one -- fires earlier.

        WHAT IT IS NOT. Six attempts in this repository to REPLACE the trained head with a
        closed-form readout all failed (registry C55-C59); a linear model on these features is not
        competitive on its own. This is initialisation, not replacement: SGD still runs, still trains
        the stem and the gate, and still stops on the same rule. The measurement to trust is the
        screen, not this argument.
        """
        import torch
        from active_segmenter.segment.closed_form import fit_irls, standardise

        rows_x, rows_y = [], []
        rng = np.random.default_rng(0)   # subsample only; the backend carries no seed
        head.eval()
        # Inside the SAME autocast context the training loop uses. The stored tensors are bf16 when
        # amp is on, so calling the head outside it puts bf16 activations against fp32 parameters.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            for (X, C_or_img, Y, B, hw, Xf, tcl, aux, Dg) in items:
                C = C_or_img if not self.trainable_classical else self._classical(C_or_img)
                _, z = head.forward_with_penultimate(X, C, hw, Xf)     # [1, d, H, W]
                d = z.shape[1]
                zf = z[0].reshape(d, -1).float().T                     # [HW, d]
                yf = Y.reshape(-1).float()
                n = zf.shape[0]
                # Subsample: the closed-form solve does not need every pixel, and the whole point is
                # to be cheap next to an epoch. Proportional, never class-balanced -- a balanced draw
                # would miscalibrate the intercept, which is the term the 0.5 threshold rests on.
                take = min(n, 20000)
                idx = rng.choice(n, take, replace=False) if take < n else np.arange(n)
                rows_x.append(zf[idx].cpu().numpy())
                rows_y.append(yf[idx].cpu().numpy())
        head.train()

        Xa = np.concatenate(rows_x, 0).astype(np.float64)
        ya = (np.concatenate(rows_y, 0) > 0.5).astype(np.float64)
        if ya.min() == ya.max():                 # single-class support: nothing to solve
            return False
        Xs, stats = standardise(Xa)
        res = fit_irls(Xs, ya, lam=1.0, n_steps=15, normalize_penalty=True)
        w = np.asarray(res.w, np.float64)
        # Undo the standardisation so the weights apply to the RAW penultimate features the head
        # actually produces: z_std = (z - mu)/sd, so w_raw = w/sd and b_raw = b - sum(w*mu/sd).
        mu, sd = np.asarray(stats["mu"]), np.asarray(stats["sd"])
        w_raw = w[:-1] / np.maximum(sd, 1e-8)
        b_raw = float(w[-1] - float((w[:-1] * mu / np.maximum(sd, 1e-8)).sum()))
        with torch.no_grad():
            cl = head.classifier
            if cl.weight.shape[0] != 1:          # multi-class head: leave it alone
                return False
            cl.weight.copy_(torch.tensor(w_raw, dtype=cl.weight.dtype,
                                         device=cl.weight.device).view(1, -1, 1, 1))
            if cl.bias is not None:
                cl.bias.fill_(b_raw)
        return True

    def _ensure_head(self, in_dim):
        from active_segmenter.segment.head_fusion import DINOHeadFusion

        if self.head is None:
            self.head = DINOHeadFusion(in_dim, self.hidden, max(1, self._n_cls), self.proj_dim,
                                       self._n_classical, guided_fuse=self.guided_fuse,
                                       boundary_head=self.boundary_head,
                                       scale_fusion=self.scale_fusion,
                                       upsampler=self.upsampler, dist_head=self.dist_head,
                                       competitive_gate=self.competitive_gate,
                                       corr_prior=self.corr_prior, film=self.film,
                                       stem=self.stem, depth=self.depth,
                                       hidden_film=self.hidden_film,
                                       eca=self.eca, dropout=self.dropout,
                                       se=self.se, mix=self.mix,
                                       bilinear=self.bilinear).to(self.device)
            if self.channels_last:
                self.head = self.head.to(memory_format=torch.channels_last)
            if self.compile_head:
                # `dynamic=True`: the native output size differs per image AND per augmentation
                # (an odd quarter-turn transposes it), so a static-shape compile would recompile
                # on nearly every step and lose more than it gains.
                self.head = torch.compile(self.head, dynamic=True)
            if (self.corr_prior or self.film) and self._fg_proto is not None:
                self.head.set_prototypes(self._fg_proto, self._bg_proto,
                                         self._fg_protos, self._bg_protos)
        return self.head

    def _augment_support(self, support):
        """Expand the support set with re-ENCODED augmented copies, once, before the fit.

        WHY RE-ENCODING RATHER THAN AUGMENTING FEATURES. The training loop already augments, but it
        applies `_d4` to the FEATURE tensors, and that is only valid because the dihedral group
        commutes with a patch grid. Elastic warps, rescaling and intensity changes do not commute
        with anything -- a warped image has different patch contents, not permuted ones -- so the
        only way to get them is to warp the IMAGE and put it back through the encoder. That is why
        this method exists and why the loop's augmentation cannot be extended to cover it.

        WHY IT SHOULD PAY. The head is 0.5M parameters fitted on eight images: a data-starved regime
        by construction, and the one nnU-Net escapes with a heavy augmentation pipeline. Profiling
        puts one encoder forward at 0.15-0.22 s, so `n_aug=4` over eight masks is about 7 s of
        encoding against a 600 s fit -- roughly 1% for a five-fold increase in effective support.

        The label is warped by the SAME transform with nearest-neighbour interpolation, so a thin
        vessel is never interpolated into existence or out of it. Intensity jitter is applied to the
        image only. Anything that would empty a mask is discarded rather than trained on.
        """
        import numpy as _np
        from scipy import ndimage as ndi

        if self.n_aug <= 0:
            return support
        # The backend has no `seed` attribute -- determinism of the fit comes from the global
        # torch seed the caller sets -- so the augmentation is seeded from the SUPPORT CONTENT.
        # That makes it deterministic for a given support set (the same eight masks always get
        # the same eight augmentations, so a re-run reproduces) while still differing between
        # draws, which a fixed constant would not.
        h = 0
        for ex in support:
            lb = _np.asarray(ex.label_map)
            h = (h * 1000003 + int(lb.sum()) + lb.size) % (2 ** 61 - 1)
        rng = _np.random.default_rng(h)
        out = list(support)
        for ex in support:
            im0 = _np.asarray(ex.image)
            lb0 = _np.asarray(ex.label_map)
            for k in range(self.n_aug):
                # scale + rotation as one affine, elastic as a smoothed displacement field
                sc = float(rng.uniform(0.85, 1.18))
                ang = float(rng.uniform(-25, 25))
                a = ndi.rotate(im0, ang, axes=(0, 1), reshape=False, order=1, mode="reflect")
                l = ndi.rotate(lb0, ang, axes=(0, 1), reshape=False, order=0, mode="reflect")
                if abs(sc - 1.0) > 1e-3:
                    zoom = [sc, sc] + ([1] * (a.ndim - 2))
                    a = ndi.zoom(a, zoom, order=1)
                    l = ndi.zoom(l, [sc, sc], order=0)
                    a = _crop_or_pad(a, im0.shape)
                    l = _crop_or_pad(l, lb0.shape)
                if self.aug_elastic > 0:
                    h, w = l.shape[:2]
                    sig = max(h, w) / 16.0
                    dy = ndi.gaussian_filter(rng.normal(0, 1, (h, w)), sig) * self.aug_elastic * sig
                    dx = ndi.gaussian_filter(rng.normal(0, 1, (h, w)), sig) * self.aug_elastic * sig
                    yy, xx = _np.mgrid[:h, :w]
                    crd = _np.stack([yy + dy, xx + dx])
                    l = ndi.map_coordinates(l, crd, order=0, mode="reflect")
                    if a.ndim == 2:
                        a = ndi.map_coordinates(a, crd, order=1, mode="reflect")
                    else:
                        a = _np.stack([ndi.map_coordinates(a[..., c], crd, order=1, mode="reflect")
                                       for c in range(a.shape[-1])], -1)
                if (l > 0).sum() < 8:            # an augmentation that erased the object teaches nothing
                    continue
                af = a.astype(_np.float32)
                af = _np.clip(af * float(rng.uniform(0.9, 1.1)) +
                              float(rng.uniform(-0.04, 0.04)) * (af.max() - af.min()), 
                              af.min(), af.max())
                a = af.astype(im0.dtype) if im0.dtype != _np.float32 else af
                fg = self.encoder.extract(a) if self.encoder is not None else ex.feat_grid
                out.append(LabeledExample(a, fg, l))
        return out

    def fit(self, support: list[LabeledExample], warm: bool = False,
            epochs: int | None = None) -> None:
        """Fit the head on the support masks.

        ``warm`` is the interactive-annotation path. In a HITL loop the user adds one mask at a time
        and the model refits after each, so the previous head -- fitted on a support set that
        differs by ONE image -- is a far better starting point than random init, and most of the
        60-step budget is spent re-deriving what it already knew. Warm start keeps those weights and
        runs ``epochs`` steps from there.

        What warm start does NOT keep is the prototypes: a new mask changes what foreground looks
        like, and FiLM and the correspondence channel read them, so they are re-derived. The colour
        choice IS kept, deliberately -- flipping the input channel mid-session would invalidate every
        cached classical map and make the tool's behaviour jump for a reason the user cannot see.
        """
        # Re-encoded augmentation BEFORE anything reads the support: the colour decision, the
        # prototypes and the bank are all derived from this list, so augmenting later would
        # leave them describing a different support set than the head trains on.
        if self.n_aug > 0 and not warm:
            n0 = len(support)
            support = self._augment_support(support)
            print(f'    [aug] support {n0} -> {len(support)} re-encoded copies', flush=True)
        import torch
        import torch.nn.functional as F

        if warm and self.head is not None:
            self._fg_proto = self._bg_proto = None      # re-derive from the enlarged support
            self._fg_protos = self._bg_protos = None

        if not support:
            # never-trained → foreground()/predict() will return empty. Warn so an empty-support call
            # is distinguishable from a genuine empty prediction (else it reads as "confidently empty").
            print("    [head_fusion] fit() called with EMPTY support — head left untrained (empty masks)",
                  flush=True)
            return
        if self.color_adaptive and self._contrast_source is None:  # per-dataset colour channel from support
            self._choose_contrast_source(support)                  # MUST precede the first _classical call
        if self.znorm and self._znorm is None:                     # support-foreground intensity normalisation
            self._fit_znorm(support)                               # on the chosen channel, before any _classical
        if self.instance_mode == "affinity" and self._inst_r is None:  # SAM-free affinity-watershed decoder
            self._calibrate_instance(support)
        if (self.corr_prior or self.film) and self._fg_proto is None:  # support fg/bg prototypes (corr_prior + FiLM)
            self._build_prototypes(support)
        if self.bank_unfreeze_adaptive and not self.trainable_classical:  # gate bank-unfreeze on support thinness
            mt = float(np.mean([_tubularity(ex.label_map) for ex in support]))
            if mt > self.thin_ref:
                self.trainable_classical = True                          # unfreeze Frangi β/γ + Sauvola k
                if self._bank is not None:                               # AL-refit: bank may already be built frozen
                    for p in self._bank.parameters():
                        p.requires_grad_(True)
            print(f"    [bank_unfreeze] mean_tubularity={mt:.3f} ({'>' if mt > self.thin_ref else '<='}{self.thin_ref})"
                  f" → classical bank {'UNFROZEN' if self.trainable_classical else 'frozen'}", flush=True)
        if self.clahe_adaptive and self._clahe_strength is None:  # per-dataset CLAHE from support faintness
            mf = float(np.mean([_fg_faintness(ex.image, ex.label_map) for ex in support]))
            self._clahe_strength = float(np.clip(
                (self.clahe_faint_ref - mf) / max(self.clahe_faint_ref, 1e-6), 0.0, 1.0))
        if self.thin_adaptive and self._thin_active is None:  # per-dataset native-classical gate on thinness
            # mean per-component tubularity of the annotated shapes (same descriptor as adaptive clDice):
            # ~0.1–0.3 for round blobs/nuclei, ~0.6–0.9 for vessels/filaments. Must be decided BEFORE the
            # first `_classical` call so `fine_scales` reaches the (lazily-built, per-dataset) bank.
            mt = float(np.mean([_tubularity(ex.label_map) for ex in support]))
            ms = float(np.mean([max(np.asarray(ex.image).shape[:2]) for ex in support]))
            # The size bound is the head's OWN training cap, not a fitted constant: below it the native
            # bank and the capped-trained head see the same image, so the distribution shift this gate
            # exists to avoid is zero by construction. See the __init__ comment on thin_max_side.
            cap = self.max_side if self.thin_max_side is None else self.thin_max_side
            self._thin_active = thin_gate(mt, ms, self.thin_ref, cap)  # small-thin only
            if self._thin_active:                             # small-thin support → recover native-res detail
                self.tile_classical = True                    # native-res classical bank at inference (H-A)
                self.fine_scales = True                       # finer Sauvola window for sub-10px structures
            src = "train-cap" if self.thin_max_side is None else "override"
            print(f"    [thin_adaptive] mean_tubularity={mt:.3f}(>{self.thin_ref}) mean_side={ms:.0f}"
                  f"(<={cap} {src}) → native_classical={'ON' if self._thin_active else 'off'}", flush=True)
        if self.scaleconf and self._bank_scales is None and not self.dino_only:  # LEVER: bank scales from support
            self._support_scales(support)     # sets _bank_scales BEFORE the first _bank_module / _classical build
        if self.bankselect and self._bank_keep is None and not self.dino_only:  # LEVER: Fisher-select the bank
            # MUST follow the contrast-source / CLAHE / thin_adaptive setup above: the probe runs `_classical`,
            # which depends on all of them, so the kept channels are measured on the same maps training sees.
            # (dino_only zeros the whole bank → bankselect is inert; skip the probe to avoid a misleading log.)
            self._bank_separability(support)
        if self.banknorm and self._bank_scale is None:
            self._fit_bank_scale(support)
        # multi-class: K = #foreground classes (forced n_classes, or auto-detected; binary otherwise).
        # The softmax head needs K+1 channels (background + K classes); binary stays a single sigmoid channel.
        kfg = self.n_classes if self.n_classes > 1 else (
            _detect_n_classes(support) if self.adaptive_loss else 1)
        mc = kfg > 1
        self._n_cls = (kfg + 1) if mc else 1
        # Inputs stay on GPU (they fit — ~6 GB for K16), but backward runs PER MINIBATCH so only
        # self.batch_size items' activation graphs are alive at once. The full-batch OOM came from the
        # ACTIVATION graph of all K items (not the inputs), so this bounds memory (no OOM at K16/large
        # images) with NO CPU↔GPU transfer overhead, and gives higher K more gradient steps (n/bs per
        # epoch) — also fixing the fixed-epoch underfit.
        idt = torch.bfloat16 if self.amp else torch.float32   # store held inputs in bf16 → ½ input mem
        items = []
        for ex in support:
            fg = np.asarray(self._dino_x(ex.image, ex.feat_grid), np.float32)
            X = torch.from_numpy(fg.transpose(2, 0, 1))[None].to(self.device, idt)  # [1,D,G,G]
            C0 = self._classical(ex.image)                                    # frozen probe: hw + Cc
            hw = tuple(C0.shape[-2:])
            Y = torch.from_numpy((np.asarray(ex.label_map) > 0).astype(np.float32))[None, None].to(self.device)
            if Y.shape[-2:] != hw:
                Y = (F.interpolate(Y, size=hw, mode="bilinear", align_corners=False) > 0.5).float()
            B = None
            if self.boundary_head:
                B = torch.from_numpy(_boundary_target(ex.label_map))[None, None].to(self.device)
                if B.shape[-2:] != hw:
                    B = (F.interpolate(B, size=hw, mode="bilinear", align_corners=False) > 0.5).float()
            Xf = self._fine_tensor(ex.image)                                  # fine native branch (or None)
            if Xf is not None:
                Xf = Xf.to(idt)
            # adaptive clDice weight from the annotated shape's non-compactness (per image, not learned)
            tcl = (self.cldice_base * _tubularity(ex.label_map)
                   if (self.cldice and self.cldice_adaptive) else self.cldice_base)
            aux = None
            if self.adaptive_loss:  # per-image loss-term weights + signed-DT (only if boundary term engaged)
                wts = _adaptive_weights(_mask_descriptors(ex.label_map), True, _ADAPTIVE_LOSS_CFG)
                dt = None
                if wts["boundary"] >= self.adaptive_loss_eps:
                    dt = torch.from_numpy(_signed_dt(ex.label_map))[None, None].to(self.device)
                    if dt.shape[-2:] != hw:
                        dt = F.interpolate(dt, size=hw, mode="bilinear", align_corners=False)
                yc = None
                if mc:                                        # class-index target [1,H,W] for multi-class CE
                    yc = torch.from_numpy(np.asarray(ex.label_map).astype(np.int64))[None].to(self.device)
                    if yc.shape[-2:] != hw:
                        yc = F.interpolate(yc[None].float(), size=hw, mode="nearest")[0].long()
                aux = (wts, dt, yc)
            Dg = None
            if self.dist_head:                            # per-instance center-distance regression target
                Dg = torch.from_numpy(_center_dist_target(ex.label_map))[None, None].to(self.device)
                if Dg.shape[-2:] != hw:
                    Dg = F.interpolate(Dg, size=hw, mode="bilinear", align_corners=False)
            items.append((X, ex.image if self.trainable_classical else C0.to(idt), Y, B, hw, Xf, tcl, aux, Dg))
        # HELD-OUT STRIP (nnU-Net analogue). Take a strip out of every support item and keep it for
        # validation. The side ALTERNATES with the item index: a strip always cut from the same edge
        # would systematically hold out one part of the field of view, and on datasets with any
        # spatial structure (a vignette, a gradient, a coverslip edge) that is a biased validation
        # set rather than a held-out sample of the same distribution.
        vitems = []
        if self.val_loo:
            # One whole item held out. Deterministic (the last), so a re-fit of the same support set
            # validates on the same case; at K=1 there is nothing to hold out and the plateau rule
            # takes over, which is stated rather than silently degraded.
            if len(items) >= 2:
                vitems = [items[-1]]
                items = items[:-1]
            elif os.environ.get("ASG_LOG_LOSS"):
                print("    [head] val_loo at K=1: nothing to hold out -- plateau rule applies",
                      flush=True)
        if self.val_tile > 0:
            # THE STRIP MUST CONTAIN FOREGROUND. The first version cut a fixed edge and measured IoU
            # on whatever landed there; on SpheroidJ, one central spheroid per image, that is a strip
            # of pure background, so the IoU was 0 at every epoch, "no improvement" fired immediately
            # and the fit stopped at 26 steps for 0.213 instead of 0.80. nnU-Net has the same hazard
            # and answers it with foreground oversampling (0.33 of every batch forced to contain a
            # label); the analogue here is to CHOOSE the side that carries foreground, and to decline
            # to validate an item where neither does rather than invent a signal from an empty strip.
            f = self.val_tile
            tr, kept = [], 0
            for it in items:
                Y = it[2]
                tot = float((Y > 0.5).float().sum())
                cand = []
                for lo, hi, tlo, thi in ((1.0 - f, 1.0, 0.0, 1.0 - f), (0.0, f, f, 1.0)):
                    vy = _crop_frac(Y, lo, hi)
                    vfg = float((vy > 0.5).float().sum())
                    # Enough foreground to score, and not so much that training loses most of it.
                    ok = vfg >= 16 and (tot - vfg) >= 0.5 * tot
                    cand.append((vfg if ok else -1.0, lo, hi, tlo, thi))
                best = max(cand)
                if best[0] < 0:
                    tr.append(it)                    # no usable strip: this item is trained on whole
                    continue
                _, lo, hi, tlo, thi = best
                tr.append(_split_item(it, tlo, thi))
                vitems.append(_split_item(it, lo, hi))
                kept += 1
            items = tr
            if not vitems and os.environ.get("ASG_LOG_LOSS"):
                print("    [head] val_tile: no support item has a strip carrying foreground on "
                      "either side -- falling back to the training-loss plateau rule", flush=True)
        in_dim = int(np.asarray(support[0].feat_grid).shape[-1])
        head = self._ensure_head(in_dim)
        params = list(head.parameters())
        if self.trainable_classical:
            params += [p for p in self._bank_module().parameters() if p.requires_grad]
        if self.sgd_poly:
            # nnU-Net's own optimiser, read out of the installed nnunetv2 rather than the paper:
            # SGD(lr, momentum=0.99, nesterov=True) with PolyLRScheduler, exponent 0.9.
            opt = torch.optim.SGD(params, lr=self.lr, momentum=0.99, nesterov=True,
                                  weight_decay=self.weight_decay)
        else:
            # AdamW rather than Adam when a decay is asked for: Adam's coupled L2 interacts with the
            # per-parameter step size, which is exactly the regime where it behaves unlike weight decay.
            opt = (torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
                   if self.weight_decay > 0 else torch.optim.Adam(params, lr=self.lr))
        if self.convex_loss:
            # lam_bridge FROM THE SUPPORT MASKS, not a tuned constant -- the anti-bridging term
            # should engage where objects are dense enough to merge and vanish where there is
            # nothing to merge. The signal is the fraction of support pixels the bridge map
            # marks at all: a field of well-separated spheroids marks almost none, a field of
            # touching nuclei marks a visible band around every neighbour pair. Scaled so a
            # typical dense field lands near 1 and a sparse one near 0, which keeps this an
            # axis of the same closed-form self-configuration as the loss ramps and the colour
            # channel rather than another knob.
            fracs = []
            for (_X, _C, Y, *_rest) in items:
                w = self._wbridge_cache.get(len(fracs))
                if w is None:
                    wnp = _bridge_weight(Y[0, 0].detach().cpu().numpy())
                    w = torch.from_numpy(wnp)[None, None].to(Y.device, Y.dtype)
                    self._wbridge_cache[len(fracs)] = w
                fracs.append(float((w > 0).float().mean()))
            # Saturating ramp rather than a clipped linear one: a hard clip at 40x reached 1.0
            # by 2.5% marked area, so every dense field got the same weight and the axis
            # carried no information above that. 1 - exp(-f/0.03) is zero at zero, monotone,
            # bounded, and still separates 3% marked (0.65) from 16% (0.99).
            f_mean = sum(fracs) / max(len(fracs), 1)
            self._lam_bridge = float(1.0 - np.exp(-f_mean / 0.03))
            print(f"    [convex_loss] bridge-marked support fraction "
                  f"{f_mean:.4f} -> lam_bridge={self._lam_bridge:.3f}",
                  flush=True)
        if self.ridge_init:
            self._ridge_init_classifier(head, items)
        head.train()
        # GRADIENT ACCUMULATION: mathematically identical to full-batch (ONE optimizer step per epoch,
        # gradient averaged over all n items), but the backward runs per minibatch so only self.batch_size
        # items' activation graphs are alive at once → bounds memory (no OOM at K16/large images) with the
        # SAME step count and ~same speed as the old full-batch loop (the OOM was the full-batch activation
        # graph, not the inputs). Per-minibatch shuffle only affects the flip-aug draw order.
        bs, n = max(1, self.batch_size), len(items)
        # j -> soft skeleton of that image's UNaugmented label. The label is fixed for the whole
        # fit, so this is computed once per support image instead of once per epoch; at 500
        # epochs on a 1000px vessel field that is the difference between one six-iteration
        # erode/open pass and five hundred of them, for a tensor that never changes.
        _yskel_cache: dict = {}
        n_epochs = int(epochs) if epochs else self.epochs
        # TRAINING BUDGET DIAGNOSTIC. There is no validation split -- all K support masks are
        # trained on -- so neither overfitting nor undertraining can be detected from a held-out
        # signal. The training loss at least says whether the optimiser has stopped moving by the
        # fixed epoch budget; a curve still descending at the last step means the budget binds.
        # None, not [], when nothing tracks it: an all-zero curve of the right length looks
        # measured, and the question it exists to answer is exactly whether the loss is flat.
        track_loss = bool(self.early_stop or os.environ.get("ASG_LOG_LOSS"))
        self._loss_curve = [] if track_loss else None
        self._stopped_at = None
        self._val_curve = [] if vitems else None
        _best_val, _best_state, _since_best = -1.0, None, 0
        for _ep in range(n_epochs):
            if self.sgd_poly:                                   # PolyLRScheduler: lr*(1-e/E)^0.9
                for g in opt.param_groups:
                    g["lr"] = self.lr * (1.0 - _ep / float(n_epochs)) ** 0.9
            perm = torch.randperm(n).tolist()
            opt.zero_grad()
            epoch_loss = 0.0
            for s in range(0, n, bs):
                loss = 0.0
                for j in perm[s:s + bs]:
                    X, C_or_img, Y, B, hw, Xf, tcl, aux, Dg = items[j]
                    # Fixed for the whole fit, so skeletonise once and cache on the item.
                    yskel = _yskel_cache.get(j)
                    if yskel is None:
                        yskel = _yskel_cache[j] = _soft_skeleton(Y).detach()
                    # Bridge map: depends only on the label, so once per image like the skeleton.
                    wbr = self._wbridge_cache.get(j)
                    if wbr is None and self.convex_loss:
                        wnp = _bridge_weight(Y[0, 0].detach().cpu().numpy())
                        wbr = self._wbridge_cache[j] = torch.from_numpy(wnp)[None, None].to(Y.device, Y.dtype)
                    C = self._classical(C_or_img, grad=True) if self.trainable_classical else C_or_img
                    adt = aux[1] if aux is not None else None
                    ayc0 = aux[2] if aux is not None else None
                    # AUGMENTATION GROUP. The reported configuration draws from {identity, hflip},
                    # two of the eight symmetries of the square. Microscopy has no canonical
                    # orientation, so the full dihedral group is available and exact -- no
                    # interpolation, unlike an arbitrary-angle rotation. `aug="d4"` uses all eight.
                    kk = int(torch.randint(0, 4, (1,))) if self.aug == "d4" else 0
                    ff = bool(torch.rand(1) < 0.5)
                    if kk or ff:
                        Xa, Ca, Ya = _d4(X, kk, ff), _d4(C, kk, ff), _d4(Y, kk, ff)
                        Ba, Xfa, dta = _d4(B, kk, ff), _d4(Xf, kk, ff), _d4(adt, kk, ff)
                        Dga = _d4(Dg, kk, ff)
                        yskela = _d4(yskel, kk, ff)   # exact: D4 commutes with the pooling
                        wbra = _d4(wbr, kk, ff) if wbr is not None else None
                        # ayc0 is [B,H,W] (class indices), so its spatial dims are the last two as
                        # well; _d4 addresses them negatively for exactly this reason.
                        ayc = _d4(ayc0, kk, ff)
                        # An odd number of quarter-turns transposes the image, and `hw` is the native
                        # output size the head upsamples to. Leaving it unswapped would silently
                        # resample every augmented sample to the wrong aspect ratio.
                        hw_a = (hw[1], hw[0]) if kk % 2 else hw
                    else:
                        Xa, Ca, Ya, Ba, Xfa, dta, ayc, Dga = X, C, Y, B, Xf, adt, ayc0, Dg
                        yskela = yskel
                        wbra = wbr
                        hw_a = hw
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                        if self.grad_checkpoint:  # recompute native-res penultimate in backward (exact)
                            from torch.utils.checkpoint import checkpoint
                            z = checkpoint(head._penultimate, Xa, Ca, hw_a, Xfa, use_reentrant=False)
                        else:
                            z = head._penultimate(Xa, Ca, hw_a, Xfa)
                        fg_logit = head.classifier(z)
                        b_logit = head.boundary(z) if head.boundary is not None else None
                        d_logit = head.dist(z) if head.dist is not None else None
                    fgl = fg_logit.float()                        # [1,C,H,W] (mc) or [1,1,H,W] (binary)
                    if self.adaptive_loss and aux is not None:    # construct the loss from mask morphology
                        wts, eps = aux[0], self.adaptive_loss_eps
                        if self.loss_scale != 1.0:                # SENSITIVITY: scale the adaptive aux weights
                            wts = {k: (v * self.loss_scale if isinstance(v, (int, float)) else v)
                                   for k, v in wts.items()}
                        if self._n_cls > 1:                       # MULTI-CLASS region anchor (CE+Dice over C)
                            loss = loss + _mc_ce_dice(fgl, ayc)
                            fgp = 1.0 - fgl.softmax(1)[:, :1]     # foreground = P(not background)
                        else:                                     # BINARY region anchor (Dice+BCE)
                            if self.convex_loss:
                                # ONE-SOLVE OBJECTIVE. Every term is convex in the WEIGHTS, which is
                                # the property that matters and which the first version of this block
                                # did not have: it fed `fgp = sigmoid(fgl)` to the skeleton and bridge
                                # terms, and those are linear in the PREDICTION but not convex in `w`,
                                # because sigmoid is neither convex nor concave. Both now take the
                                # LOGITS through softplus, which is a convex upper bound on what each
                                # was computing (`-log t >= 1-t`), so the sum is genuinely convex.
                                fgp = fgl.sigmoid()
                                loss = loss + _lovasz_hinge(fgl, Ya)
                                if wts["cldice"] >= eps:      # thinness ramp, already support-derived
                                    loss = loss + wts["cldice"] * _skeleton_recall(fgl, yskela)
                                if wbra is not None and float(wbra.sum()) > 0:
                                    # anti-bridging, same convexification: penalise softplus(+z) on the
                                    # merge loci rather than sigmoid(z). Upper-bounds `mean_W(p)` since
                                    # `-log(1-t) >= t`, and is convex-nondecreasing composed with linear.
                                    # Divided by 2*ln2 for the same reason as `_skeleton_recall`: at
                                    # z=0 the old term contributed sigmoid(0)=0.5, so this matches it
                                    # at the point where `_lam_bridge` was calibrated. Unbounded above
                                    # by construction; that is inherent to a convex majorant.
                                    import math as _math

                                    import torch.nn.functional as _F
                                    loss = loss + self._lam_bridge * (
                                        wbra * _F.softplus(fgl)).sum() / wbra.sum() / (
                                        2.0 * _math.log(2.0))
                            else:
                                loss = loss + (_lovasz_hinge(fgl, Ya) if self.lovasz else _dice_bce(fgl, Ya))
                                fgp = fgl.sigmoid()
                            # EVERY term below is skipped under `convex_loss`, and that is a
                            # CORRECTNESS requirement, not a preference. focal is convex in p but not
                            # in the logit (sigma(-z)^gamma has an inflection); Tversky, Boundary-DoU
                            # and the precision-Tversky are ratios; none is convex in `w`. Without
                            # these guards the "one-solve objective" silently carried four non-convex
                            # terms, so a screen of it would have measured something other than the
                            # thing its own docstring describes.
                            if not self.convex_loss and wts["focal"] >= eps:   # binary imbalance terms
                                loss = loss + wts["focal"] * _focal_bce(fgl, Ya)
                            if not self.convex_loss and wts["tversky"] >= eps:
                                loss = loss + wts["tversky"] * _tversky(fgl, Ya)
                            if (not self.convex_loss and self.boundary_dou
                                    and wts.get("bdou", 0) >= eps):          # Boundary DoU: sharpen fg
                                loss = loss + wts["bdou"] * _boundary_dou(fgl, Ya)  # contour / curb over-pred
                            if (not self.convex_loss and self.prec_loss
                                    and wts.get("prec", 0) >= eps):     # precision Tversky α>β: curb
                                loss = loss + wts["prec"] * _tversky(fgl, Ya, alpha=0.7, beta=0.3)  # interior bleed
                        # THIS BLOCK SITS OUTSIDE THE binary/multi-class SPLIT, which is how the thin
                        # term came to be applied TWICE under `convex_loss`: once as the convex
                        # `_skeleton_recall` above and again here as `_soft_cldice`, at the SAME
                        # `wts["cldice"]` weight -- and `_soft_cldice` soft-skeletonises the PREDICTION,
                        # the one operation the whole design says cannot be convex. The guard makes the
                        # convex arm exactly the three terms its docstring lists.
                        if not self.convex_loss and wts["cldice"] >= eps:  # thin/tubular topology (fg union)
                            loss = loss + wts["cldice"] * _soft_cldice(fgp, Ya, target_skel=yskela)
                        if not self.convex_loss and wts["boundary"] >= eps and dta is not None:
                            loss = loss + wts["boundary"] * _boundary_ls(fgp, dta)  # jagged contours
                        if wts.get("instance", 0) >= eps and b_logit is not None:  # touching-instance split
                            loss = loss + wts["instance"] * F.binary_cross_entropy_with_logits(b_logit.float(), Ba)
                    else:
                        loss = loss + (_lovasz_hinge(fgl, Ya) if self.lovasz else _dice_bce(fgl, Ya))
                        if self.cldice and tcl > 0:        # topology term, per-image adaptive weight tcl
                            loss = loss + tcl * _soft_cldice(fgl.sigmoid(), Ya, target_skel=yskela)
                        if b_logit is not None:
                            loss = loss + F.binary_cross_entropy_with_logits(b_logit.float(), Ba)
                    if d_logit is not None and Dga is not None:  # DT-regression instance head (center-dist)
                        wd = (aux[0].get("instance", 0.0) if (self.adaptive_loss and aux is not None) else 1.0)
                        if wd >= self.adaptive_loss_eps:   # skip on single-instance images (adaptive gate)
                            loss = loss + wd * F.mse_loss(d_logit.float().sigmoid(), Dga)
                if track_loss:
                    epoch_loss += float(loss.detach()) / n   # host sync: only when used
                (loss / n).backward()            # accumulate (÷n = full-batch average)
            opt.step()                           # ONE step per epoch == full-batch
            if track_loss:
                self._loss_curve.append(float(epoch_loss))
            if vitems:
                # Stop on the METRIC on held-out pixels, not on a proxy loss over trained ones.
                # nnU-Net validates every epoch too (num_val_iterations_per_epoch=50); it then runs
                # a fixed budget, whereas keeping the best state means the returned head is the one
                # that scored best rather than the one the last step happened to leave behind.
                with torch.no_grad():
                    inter = union = 0.0
                    for (Xv, Cv, Yv, _Bv, hwv, Xfv, _t, _a, _D) in vitems:
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                            zv = head._penultimate(Xv, Cv, hwv, Xfv)
                            pv = (head.classifier(zv)[:, :1].float().sigmoid() > 0.5).float()
                        yv = (Yv > 0.5).float()
                        u = float(((pv + yv) > 0).float().sum())
                        if u <= 0:
                            continue          # nothing predicted and nothing labelled: no evidence
                        inter += float((pv * yv).sum())
                        union += u
                    viou = inter / union if union > 0 else float("nan")
                if viou != viou:              # every strip vacuous this epoch: no signal to act on
                    continue
                self._val_curve.append(viou)
                if viou > _best_val + 1e-6:
                    _best_val, _since_best = viou, 0
                    _best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
                else:
                    _since_best += 1
                    if self.early_stop and _since_best >= self.patience:
                        self._stopped_at = len(self._val_curve)
                        break
            # PLATEAU STOP, on the TRAINING loss. There is no held-out signal to stop on -- all K
            # support masks are trained on -- so this detects CONVERGENCE, not overfitting, and is
            # not pretending otherwise: overfitting is handled by weight decay and by keeping the
            # head small. Compares two adjacent windows rather than two points, because the flip
            # augmentation makes a single step's loss noisy enough to trip a point-wise test early.
            if self.early_stop and not vitems and self._loss_curve is not None \
                    and len(self._loss_curve) >= 2 * self.patience:
                w = self.patience
                prev = sum(self._loss_curve[-2 * w:-w]) / w
                cur = sum(self._loss_curve[-w:]) / w
                if prev - cur < self.min_rel_improve * max(abs(prev), 1e-8):
                    self._stopped_at = len(self._loss_curve)
                    break
        if vitems and _best_state is not None:
            head.load_state_dict(_best_state)          # return the best head, not the last one
            if os.environ.get("ASG_LOG_LOSS"):
                print(f"    [head] val-strip IoU {self._val_curve[0]:.4f} -> best {_best_val:.4f} "
                      f"over {len(self._val_curve)} epochs"
                      + (f" (stopped at {self._stopped_at})" if self._stopped_at else ""), flush=True)
        if os.environ.get("ASG_LOG_LOSS"):
            c = self._loss_curve
            # Print the ends and the tail slope: a budget that binds shows a tail still descending,
            # a budget past convergence shows it flat or rising.
            tail = (c[-1] - c[-min(10, len(c))]) if len(c) > 1 else 0.0
            why = (f"plateau at {self._stopped_at}" if self._stopped_at
                   else f"budget exhausted at {n_epochs}")
            print(f"    [head] loss {c[0]:.4f} -> {c[-1]:.4f} over {len(c)} steps ({why}); "
                  f"last-10 change {tail:+.5f} ({'still descending' if tail < -1e-4 else 'flat/rising'})",
                  flush=True)
        head.eval()
        if self.morph_post:
            # Learn the kernel from the FITTED head's own predictions on the support, scored on the
            # metric this dataset is reported by. The thinness gate that turns on the centreline
            # loss is the same signal that says which metric that is, so this consumes a decision
            # the pipeline already makes rather than adding one.
            # Which metric this dataset is reported by, from the same tubularity descriptor the
            # adaptive loss uses to decide whether the centreline term engages at all. A
            # morphology chosen against the wrong metric would optimise the wrong thing.
            tub = float(np.median([_tubularity(np.asarray(e.label_map)) for e in support]))
            met = "cldice" if tub >= self.thin_ref else "fg_iou"
            preds = [np.asarray(self.foreground(e.image, e.feat_grid)) for e in support]
            gts = [np.asarray(e.label_map) for e in support]
            self._morph_rule = _fit_morphology(preds, gts, metric=met)
            print(f"    [morph] metric={met} rule={self._morph_rule}", flush=True)
        if self.tau_cal:
            self._tau = self._calibrate_tau(support)

    def _calibrate_tau(self, support):
        """Pick the readout threshold from the support masks alone.

        WHY THIS IS NOT METRIC PEEKING. Which metric to optimise is decided by the SAME tubularity
        descriptor that already decides whether the centreline loss term engages, so this consumes a
        decision the pipeline makes anyway rather than adding one, and nothing here sees the test
        split. The threshold is chosen IN-SAMPLE on the K support images: a proper leave-one-out
        would need K refits of the head, and one scalar fitted against eight images of pixels is a
        mild enough fit that the screen -- which scores on held-out test images -- can only be made
        conservative by it, never flattered.

        The shift is applied in LOGIT space BEFORE the resize, because `p > tau` is not
        `z > logit(tau)` once a thresholded grid is upsampled (registry C61 records that trap).
        """
        import numpy as np

        from active_segmenter.eval import metrics

        tub = float(np.median([_tubularity(np.asarray(e.label_map)) for e in support]))
        use_cldice = tub >= self.thin_ref
        grids = []
        for e in support:
            lg, _ = self._native_logits(e.image, e.feat_grid)
            grids.append((lg, np.asarray(e.label_map) > 0))

        def score_at(tau):
            vals = []
            for lg, gt in grids:
                m = foreground_from_score(lg, gt.shape, thresh=float(tau))
                vals.append(metrics.cldice(m, gt) if use_cldice
                            else metrics.foreground_iou(m, gt))
            return float(np.mean(vals))

        taus = np.linspace(-2.0, 2.0, 21)
        scores = [score_at(x) for x in taus]
        best = float(taus[int(np.argmax(scores))])
        print(f"    [tau_cal] metric={'cldice' if use_cldice else 'fg_iou'} "
              f"tau={best:+.2f} support score {scores[10]:.4f} -> {max(scores):.4f}", flush=True)
        return best

    def _native_logits(self, image, feat_grid):
        import torch

        idt = torch.bfloat16 if self.amp else torch.float32
        fg = np.asarray(self._dino_x(image, feat_grid), np.float32).transpose(2, 0, 1)[None]
        X = torch.from_numpy(fg).to(self.device, idt)
        C = self._classical(image, inference=True).to(idt)  # native tiled if tile_classical
        hw = tuple(C.shape[-2:])
        Xf = self._fine_tensor(image)               # fine native branch (scale-fusion) or None
        if Xf is not None:
            Xf = Xf.to(idt)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            out = self.head(X, C, hw, Xf)
        if out.shape[1] > 1:            # multi-class → foreground score = logsumexp(fg classes) − bg logit
            logits = torch.logsumexp(out[0, 1:].float(), 0) - out[0, 0].float()
        else:
            logits = out[0, 0].float()
        return logits.cpu().numpy(), hw

    def foreground(self, image, feat_grid, class_id: int = 1) -> np.ndarray:
        if self.head is None:
            return np.zeros(np.asarray(image).shape[:2], bool)
        logits, _ = self._native_logits(image, feat_grid)
        m = foreground_from_score(logits, np.asarray(image).shape[:2], thresh=self._tau)
        if self.morph_post and self._morph_rule is not None:
            m = _apply_morphology(m, self._morph_rule)
        return m

    def foreground_prob(self, image, feat_grid, class_id: int = 1) -> np.ndarray:
        """Foreground PROBABILITY (sigmoid) resized to image resolution — needed by the crop
        pipeline to blend overlapping crops before thresholding (a hard mask can't be feathered)."""
        from skimage.transform import resize

        tgt = np.asarray(image).shape[:2]
        if self.head is None:
            return np.zeros(tgt, np.float32)
        logits, _ = self._native_logits(image, feat_grid)
        prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
        if prob.shape != tgt:
            prob = resize(prob, tgt, order=1, mode="edge", anti_aliasing=False).astype(np.float32)
        return prob

    def _native_boundary(self, image, feat_grid):
        import torch
        from skimage.transform import resize

        if self.head is None or self.head.boundary is None:
            return None
        fg = np.asarray(self._dino_x(image, feat_grid), np.float32).transpose(2, 0, 1)[None]
        X = torch.from_numpy(fg).to(self.device)
        C = self._classical(image, inference=True)
        with torch.no_grad():
            _, b_logit = self.head.forward_fg_boundary(X, C, C.shape[-2:])
        bp = b_logit[0, 0].sigmoid().cpu().numpy().astype(np.float32)
        hw = np.asarray(image).shape[:2]
        return resize(bp, hw, order=1, mode="edge", anti_aliasing=False) > 0.5

    def _native_dist(self, image, feat_grid):
        """Predicted per-instance center-distance map (DT-regression head), resized to native res."""
        import torch
        from skimage.transform import resize

        if self.head is None or self.head.dist is None:
            return None
        fg = np.asarray(self._dino_x(image, feat_grid), np.float32).transpose(2, 0, 1)[None]
        X = torch.from_numpy(fg).to(self.device)
        C = self._classical(image, inference=True)
        Xf = self._fine_tensor(image)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            z = self.head._penultimate(X, C.to(X.dtype), C.shape[-2:], Xf.to(X.dtype) if Xf is not None else None)
            d = self.head.dist(z)[0, 0].float().sigmoid().cpu().numpy().astype(np.float32)
        hw = np.asarray(image).shape[:2]
        return resize(d, hw, order=1, mode="edge", anti_aliasing=False)

    def _dist_markers(self, image, feat_grid, fg):
        """One marker per instance = local maxima of the predicted center-distance (StarDist/AIS decoding)."""
        from skimage.feature import peak_local_max
        dmap = self._native_dist(image, feat_grid)
        if dmap is None:
            return None
        coords = peak_local_max(dmap * fg, min_distance=5, threshold_abs=0.3)
        markers = np.zeros(np.asarray(fg).shape, int)
        for i, (y, x) in enumerate(coords, 1):
            markers[y, x] = i
        return markers if markers.max() > 0 else None

    def predict(self, image, feat_grid, class_id: int = 1) -> list[InstanceMask]:
        if self.head is None:
            return []
        fg = self.foreground(image, feat_grid, class_id)  # native bool
        if self.dist_head and self.head is not None and self.head.dist is not None:  # DT-regression markers
            markers = self._dist_markers(image, feat_grid, fg)
            if markers is not None:
                return _watershed_instances(fg, class_id, markers=markers)
        if self.instance_mode == "trace":          # SAM-free crossing-filament tracer (soft/layered, amodal)
            return _trace_filament_instances(fg, class_id)
        if self.instance_mode == "affinity":       # SAM-free: geometry+ridge split, DINO-affinity merge-veto
            if self._inst_r is None or self._inst_merge_cos is None:
                raise RuntimeError("affinity predict on an UNCALIBRATED backend — fit()/_calibrate_instance "
                                   "did not run before predict")
            ridge = _ridge_map(self._channel(image))
            return _affinity_watershed_instances(fg, ridge, feat_grid, class_id,
                                                 r_star=self._inst_r, merge_cos=self._inst_merge_cos)
        if self.instance_mode == "blob":           # W2: scale-adaptive blob-detector markers
            markers = _blob_markers(fg, _gray01(image))
            return _watershed_instances(fg, class_id, markers=markers)
        if self.instance_mode == "watershed":
            return _watershed_instances(fg, class_id, boundary=self._native_boundary(image, feat_grid))
        lab = _label(fg)
        return [InstanceMask(mask=lab == i, points=None, class_id=class_id, instance_id=i - 1,
                             score=1.0) for i in range(1, int(lab.max()) + 1) if (lab == i).sum() >= 50]

    def score_map(self, image, feat_grid, class_id: int = 1) -> np.ndarray:
        # No per-pixel score head here: acquisition for this backend is the weight-coupled grad_embedding
        # (EGL/BADGE), and inference uses foreground()/predict(). A zeros stub would silently masquerade as
        # a real uncertainty map (→ effectively-random acquisition), so fail loud instead.
        raise NotImplementedError(
            "HeadFusionBackend has no score_map; use grad_embedding (acquisition) or foreground()/predict().")

    def grad_embedding(self, image, feat_grid) -> np.ndarray:
        """Weight-coupled acquisition signal (EGL/BADGE) over the fused native penultimate."""
        import torch

        if self.head is None:  # penultimate is [2·proj_dim (scale-fusion) or proj_dim] ⊕ classical, + bias
            d = (2 * self.proj_dim if self.scale_fusion else self.proj_dim) + (self._n_classical or 0) + 1
            # The bilinear lever APPENDS interaction channels to the penultimate, so this
            # untrained-head fallback has to widen with it. Otherwise the pre-fit acquisition
            # vector is SHORTER than the post-fit one, which corrupts k-means++ seeding in the
            # AL loop silently rather than raising -- and the AL loop is the deployment path.
            if self.bilinear:
                n_src = (2 if self.scale_fusion else 1) + 1
                d += self.bilinear * (n_src * (n_src - 1) // 2)
            return np.zeros(d, np.float32)
        idt = torch.bfloat16 if self.amp else torch.float32
        fg = np.asarray(self._dino_x(image, feat_grid), np.float32).transpose(2, 0, 1)[None]
        X = torch.from_numpy(fg).to(self.device, idt)
        C = self._classical(image).to(idt)
        Xf = self._fine_tensor(image)   # EGL must see the SAME scale-fused penultimate as inference
        if Xf is not None:
            Xf = Xf.to(idt)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            logits, z = self.head.forward_with_penultimate(X, C, C.shape[-2:], Xf)
            logits, z = logits.float(), z.float()
            p = logits.sigmoid()
            y = (p > 0.5).float()
            resid = (p - y)[0, 0]
            zz = z[0]
            gw = (zz * resid[None]).reshape(zz.shape[0], -1).mean(1)
            gb = resid.reshape(-1).mean().reshape(1)
            g = torch.cat([gw, gb])
        return g.cpu().numpy().astype(np.float32)
