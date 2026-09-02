# Exemplar — full method specification

**Configuration reported in the paper:**

```
head_fusion_best_nobank_noloss_nocolor_flat_es_ep500_wd4_do5_mix_noaux
```

**Simplified 2026-09-01.** The method reported in the paper no longer configures itself from the
support masks: the objective is fixed for every dataset, the bank reads grayscale, and the auxiliary
outline head is gone. Section 5 below documents what those three levers did and what removing them
cost, because the code still contains them and it must be unambiguous that they are dormant.

This document is the complete specification of that configuration: every component, every constant,
and the training paradigm. It is written from the source, not from the paper, and every number below
was read out of the code or measured by instantiating it. Where the codebase contains a lever that
the reported configuration does **not** use, that is stated explicitly rather than omitted, because
the repository contains many such levers and it must be unambiguous which ones are live.

Reference implementation: `active_segmenter/segment/head_fusion_backend.py` (the backend),
`active_segmenter/segment/head_fusion.py` (the head module), `active_segmenter/segment/hyperbank_bank.py`
(the classical bank), `scripts/al_testbed.py` (`make_backend`, which parses the configuration string).

---

## 1. Task setup

The model is given **K annotated masks** (the *support set*; K = 8 unless stated otherwise) drawn
from one dataset, and must segment the remaining images of that same dataset. The task is
personalisation to specific object instances, not generalisation to an unseen semantic class: the
support and the query images come from the same acquisition, the same modality and the same object
type.

A **separate model instance is fitted per support set**. Nothing is shared across datasets and
nothing is meta-trained: there is no pre-training stage of our own, and no weights carry over from
one dataset to the next. The only thing that is fixed across all eleven datasets is the *code path* —
the architecture, the constants below, and the rules that read the support masks.

The output is a **semantic foreground probability map at native image resolution**, thresholded to a
binary mask. Instance separation exists in the codebase as post-processing (§9) but is not part of
what the paper reports.

---

## 2. Stage 1 — frozen backbone

| item | value |
|---|---|
| model | `facebook/dinov3-vitl16-pretrain-lvd1689m` (DINOv3 ViT-L/16) |
| parameters | 303,129,600, **all frozen**, never receive a gradient |
| feature dimension | 1024 |
| patch stride | 16 px |
| encoder input resolution | 672 px (a multiple of 16, giving a 42×42 patch grid) |
| layer used | last hidden state (`layer = -1`) |
| tokens dropped | 5 prefix tokens (CLS + 4 register tokens) |
| feature super-resolution | **off** (`superres_factor = 1`) |
| layer fusion | **off** (single layer, not a concatenation of blocks) |

### 2.1 Two reading scales

A 16-pixel patch grid is too coarse for small objects, so the backbone is read **twice** and the two
readings are concatenated (`dino_scale = "both"`, `scale_fusion = True`):

- **Coarse scale.** The whole image is resized to 672 px and encoded once. This gives semantic
  context over the entire field but at a stride of `native_side × 16 / 672` native pixels — 10 px on
  a 448-px field, 83 px on HRF's 3504-px field.
- **Fine scale.** The image is tiled into 672-px windows at native resolution, each window is
  encoded, and the resulting patch grids are stitched. This gives a stride of a flat 16 native
  pixels regardless of field size. The stitched fine grid is capped at 160×160 (`fine_max_grid`);
  beyond that the tiling is coarsened so cost stays bounded.

Both scales are 1024-dimensional. Features are cached on disk, keyed by a hash of the image bytes
plus the full encoder configuration, so repeated runs over the same images do not re-encode.

---

## 3. Stage 2 — frozen classical prior bank

Deep patch features discard structure narrower than a patch. In parallel with the backbone, a bank
of **35 classical filter responses computed at native resolution** is stacked. The bank is **frozen**
(`trainable_classical = False`); it receives no gradient and its scales are fixed constants, not read
from the support.

Verified by instantiation:

| family | channels | detail |
|---|---|---|
| Frangi vesselness | 10 | 5 scales σ ∈ {1, 2, 4, 8, 16} px, **polarity-split** (bright-on-dark and dark-on-bright) |
| Laplacian-of-Gaussian | 10 | 5 scales, 2 polarities |
| Sauvola adaptive threshold | 6 | windows {15, 51, 151} px, each with its complement |
| Structure-tensor eigenvalues | 4 | λ₁ and λ₂ at σ ∈ {2, 8} px |
| Gradient magnitude | 3 | 3 scales |
| Normalised intensity | 2 | the chosen channel and its complement |
| **total** | **35** | |

The bank originates in our own prior work, where it *was* the segmenter and was trained end to end.
Here it is frozen and used as an input source only.

**Not used in the reported configuration:** `morph_orient` (+10 top-hat / orientation channels),
`bankselect` (Fisher-selecting bank channels from the support), `scaleconf` (deriving the Frangi /
Sauvola / LoG scales from support object radii), `banknorm` (support-derived per-channel scaling),
`bank_extra` (an expanded curated bank).

---

## 4. Stage 3 — the trainable head

This is the **only** part that is fitted, and it is fitted from scratch on each support set.

### 4.1 Architecture

```
coarse DINO [1024, 42, 42] ─┐
                            ├─ shared stem (body) ─→ proj ─→ 32 ch ─┐
fine DINO   [1024, gf, gf] ─┘  (weights shared)                     │
                                                                    ├─ concat → 99 ch
classical bank [35, H, W] ──────────────────────────────────────────┘   (native res)
                                                                    │
                                        GuidedUp lifts the 2×32 feature channels to native res
                                                                    │
                                            fuse: Conv2d(99→99, 1×1) + GELU
                                                                    │
                                    ┌───────────────────────────────┴──────────────────┐
                          classifier: Conv2d(99→1, 1×1)                boundary: Conv2d(99→1, 1×1)
                             (foreground logit)                      (auxiliary, see §4.4)
```

- **Stem (`body`).** `stem = "flat"`, `depth = 2`, `hidden = 256`. Two layers, each
  `Conv2d(·, 256, kernel 1×1) → GroupNorm(8 groups) → GELU → Dropout2d(0.5)`. "Flat" means **no
  spatial convolution at all** — a per-position channel mixer. The default in the codebase is a 3×3
  first layer ("wide"), which costs 2.36 M parameters; the flat stem was adopted because
  self-attention has already mixed spatially and the upsampler re-introduces spatial structure from
  the native priors afterwards. The same stem weights process both scales.
- **`proj`.** `Conv2d(256 → 32, 1×1)` — the semantic embedding handed to the fusion.
- **`up` (GuidedUp).** An edge-guided upsampler in the guided-filter family. It lifts each 32-channel
  feature map from its patch grid to native resolution through a **learned per-channel gain applied
  to an edge map reduced from the classical priors**. It is **zero-initialised**, so at step 0 it is
  exactly bilinear interpolation and departs from it only along edges the priors mark.
- **`fuse`.** `Conv2d(99 → 99, 1×1) + GELU` (this is the `mix` token). Without it the readout would
  be a single linear layer over the concatenation; with it, the head is a two-layer per-pixel MLP and
  can express interactions between the three sources.
- **`classifier`.** `Conv2d(99 → 1, 1×1)` — one foreground logit per native pixel.

### 4.2 Parameter budget (measured by instantiation)

| module | parameters |
|---|---|
| `body` (stem) | 329,216 |
| `proj` | 8,224 |
| `up` (GuidedUp) | 10,628 |
| `fuse` | 9,900 |
| `classifier` | 100 |
| **total trainable** | **358,068** |

The auxiliary outline head's `Conv2d(99 → 1, 1×1)` accounted for the other 100 parameters until it
was removed; the figure was 358,168 while it was present.

That is **0.12 %** of the frozen backbone.

### 4.3 What is deliberately absent

Two standard ways of weighting the three sources adaptively were implemented and measured on the
full panel, and both were **removed**:

- **Competitive per-pixel gate** (`cgate`): a softmax over the three fusion groups, zero-initialised
  to parity. Worth 0.000 on the ablation mean.
- **Support-conditioned FiLM** (`film`): a hypernetwork producing (γ, β) from support foreground and
  background prototypes, parity at initialisation. Worth +0.004.

Over all eleven datasets the pair is worth +0.003 at K = 8 (p = 0.084), +0.004 at K = 4 (p = 0.037)
and −0.002 at K = 1. The gain does not hold its sign across support sizes and the pair costs 29 % of
the trainable parameters, so the fusion stays unweighted.

Also absent: the correspondence-prior channel (`corr`), multi-prototype correspondence (`mproto`),
squeeze-excitation (`se`), low-rank bilinear interactions (`bilinear`), ECA channel attention
(`eca`), and the DT-regression instance head (`dist_head`, measured negative: DSB2018 0.352 against
0.451).

### 4.4 The auxiliary outline head — REMOVED 2026-09-01

A second `Conv2d(99 → 1, 1×1)` sharing the 99-channel penultimate used to predict **dilated object
outlines**, trained with binary cross-entropy alongside the foreground loss. It is off in the
reported configuration (`_noaux`) and its 100 parameters are no longer in the budget above.

What it actually computed is worth recording, because its name misdescribed it. `_boundary_target`
is `find_boundaries(mode="inner")` over every region — every object's inner rim against background,
**not** only the contact faces between touching instances. Verified 2026-08-28 on a synthetic label
map: an isolated square still receives a full outline. It was also never gated on the presence of
per-instance labels; the target falls back to `sklabel()` on binary ground truth, so two connected
components already switched it on and it was live on ten of the eleven datasets.

Its measured worth, screened at three seeds against the datasets it existed for (registry entry
2026-08-31): MoNuSeg **+0.005**, DSB2018 −0.003, Bacteria −0.001 — a mean of **+0.0003** across the
three datasets with hundreds of touching objects each — against **−0.008 on DRIVE**, a control. So
it did nothing on dense instances, the case it was built for, and everything it was worth was on
vessels, where its "outline" is essentially the whole structure and it acted as an auxiliary
thin-structure target rather than an instance separator. It was removed on that evidence.

---

## 5. Support-derived self-configuration — IMPLEMENTED, MEASURED, AND OFF

**None of this section is live in the reported configuration.** The two rules below are switched off
by `_noloss` and `_nocolor`, and this section documents them because the code still contains them and
the point of this file is that it must be unambiguous which levers are live.

They were removed on 2026-09-01 after being measured on the full panel. Together they are worth
**+0.009** across seven datasets — the colour rule **+0.005** of it, the adaptive loss **+0.003**,
and essentially all of the latter is SpheroidJ. That is under one percentage point for two rules that
cost a page of specification and, more importantly, cost the paper its simplest possible claim: one
fixed configuration, no per-dataset adaptation of any kind. The simplification is not free — it is
what moved the panel mean from 0.789 to 0.782 and lost the tie with nnU-Net on overlap — but a
measurement stated in one line is worth more than a mechanism stated in a page.

The rules as implemented, for anyone re-enabling them: **every rule below is closed-form** — evaluated
once from the K support masks before training starts, no optimisation, no gradient, no user knob.

### 5.1 Axis 1 — the input signal (the larger of the two effects)

**Channel selection.** The bank reads a *single* channel. The candidates are, for a colour dataset:
grayscale, R, G, B, and the haematoxylin and eosin channels from stain deconvolution; for a
monochrome dataset, grayscale only. The chosen channel maximises **Fisher separability between
foreground and a local background ring** across the support set:

- The background is a **dilated ring around each foreground structure**, not the global background.
  The bank is a stack of *local* operators (Frangi ridges, Sauvola adaptive thresholds), so the
  channel that matters is the one with the best local foreground-versus-surround contrast. With a
  global background, DRIVE's black field-of-view border made the bright red channel win spuriously
  over the textbook green vessel channel.
- The decision is made **per dataset, not per image**, in two passes: first a detection pass decides
  whether a majority of support images are genuinely colour; then, if so, all candidates are computed
  with `force=True` on *every* image so the average is over the same image set.
- The **median** across support images is used, not the mean, so one degenerate image cannot dominate.
- Grayscale is displaced only if a colour channel beats it by a margin of **1.05×** (`color_margin`),
  i.e. a 5 % Fisher margin. Monochrome data is therefore untouched.

**Contrast normalisation.** A CLAHE strength is derived from the support foreground-to-background
contrast, referenced to a faintness of **0.3** (`clahe_faint_ref`). Faint vessels are enhanced on
their highest-contrast channel; distinct, high-contrast objects are left alone.

### 5.2 Axis 2 — the loss weights

Five closed-form shape descriptors are computed on each support mask (`_mask_descriptors`):

| descriptor | definition |
|---|---|
| **thinness** τ | 1 − solidity (area / convex-hull area), averaged **per connected component**, area-weighted. ≈0 for compact blobs, ≈1 for branching structures. Per-component is essential: a global hull over scattered round nuclei is huge, would read as "thin", and would wrongly switch on the centreline loss. |
| **foreground fraction** φ | foreground pixels / total pixels |
| **contour complexity** κ | area-weighted mean of `perimeter² / (4π·area) − 1` (0 for a disc, ↑ for jagged or elongated contours) |
| **instance density** | number of connected components with area ≥ 4 px |
| **mean radius** ρ̄ | mean of the Euclidean distance transform over foreground pixels — falls with how narrow the structure is (r/2 for a strip, R/3 for a disc, so not the half-width) |

These drive the per-image weight of each loss term through **fixed monotone ramps**. With
`r(x; a, b) = clip((x − a)/(b − a), 0, 1)`:

| term | weight |
|---|---|
| Dice | `1.0` — always on, the region anchor |
| focal | `0.5 · r(φ_ref − φ; 0, φ_ref)` |
| Tversky | `0.5 · r(φ_ref − φ; 0, φ_ref)` |
| **clDice** | `0.5 · r(τ; 0.10, 0.65) · [1 − r(ρ̄; 4.0, 6.0)]` |
| boundary distance | `0.3 · r(κ; 0.4, 3.0) · (1 − τ)` |
| instance separation | `0.6 · r(density; 1, 12)` — only when the annotation holds multiple instances |

with `φ_ref = 0.15`. A term whose weight falls below **0.02** (`adaptive_loss_eps`) is skipped
entirely rather than computed and multiplied by a near-zero number.

The **two-factor gate on clDice** requires *thin* **and** *narrow*, so the term is progressively
damped as the support objects thicken rather than switched off at a threshold.

> [!warning] **Verified 2026-08-25 — the gate does NOT switch clDice off on MoNuSeg.**
> `_tubularity` downscales masks above 512 px with `resize(m.astype(np.float32), …, order=0)`.
> skimage exempts *integer* input at `order=0` from anti-aliasing; the `.astype(np.float32)` on
> `head_fusion_backend.py:590` defeats that exemption, so the instance map is Gaussian-blurred
> before nearest sampling, ids blend and solidity collapses. Reproduced with the shipped function
> on 1444 synthetic nuclei: **τ = 0.975 → w_cl = 0.500 (the ceiling)**; the identical code with
> `anti_aliasing=False` gives **τ = 0.022 → w_cl = 0.000**. The effect needs a downscale factor
> above ≈1.55, so DRIVE (584 px) and CTC-U373/BBBC010 (696 px) are unaffected, and binary-GT
> datasets are immune because `sklabel` runs after the resize.
> With the defect fixed it is the **thinness** factor that kills the term on nuclei (τ ≈ 0.02–0.04
> against `tau_lo = 0.10`), not the radius factor, which only damps to ≈0.30–0.58 at MoNuSeg's
> ρ̄ ≈ 5. The reported numbers are unaffected — the shipped behaviour is what was benchmarked —
> but any re-run after fixing the resize will change MoNuSeg's training loss.

Note that focal and Tversky are **not** gated by (1 − τ). Suppressing them on thin vessels was
measured and hurt (HRF 0.598 → 0.596), so they remain imbalance-driven only.

### 5.3 Constants that were fixed rather than derived

Even when these rules were live, three settings were the same for all eleven datasets and were *not*
read from the masks:

1. the 5 % Fisher margin for switching away from grayscale,
2. the CLAHE reference faintness of 0.3,
3. the loss-ramp break-points listed above.

---

## 6. Training paradigm

| item | value |
|---|---|
| what is trained | the head only (358,068 parameters). Backbone and bank are frozen. |
| optimiser | **AdamW** (Adam is used only when weight decay is 0; here it is not) |
| learning rate | `1e-3`, **constant** — no schedule, no warm-up, no decay |
| weight decay | `1e-4` (the `wd4` token) |
| dropout | `Dropout2d(0.5)` after each stem activation (the `do5` token). Channel dropout, not element dropout: adjacent positions in a convolutional channel are strongly correlated, so dropping individual elements mostly adds noise the neighbours reconstruct. |
| maximum epochs | **500** (the `ep500` token) |
| stopping | **plateau on the training loss** (the `es` token), see §6.2 |
| batching | `batch_size = 4` with **gradient accumulation** |
| optimiser steps | **one per epoch** — mathematically full-batch |
| augmentation | **random flips only** (`aug = "flip"`); dihedral (`dih`) and elastic augmentation exist but are off, and `n_aug = 0` means the support set is not expanded with pre-encoded copies |
| mixed precision | on, **bfloat16** autocast |
| native-resolution cap | 1536 px (`max_side`) — larger fields are downscaled, at inference as well as during fitting |
| initialisation | random (PyTorch defaults), except GuidedUp which is **zero-initialised to exact bilinear**. `ridge_init` (a closed-form warm start for the classifier) exists in the code and is **off** in the reported configuration. |

### 6.1 Why gradient accumulation rather than a full batch

The gradient is averaged over all K items and one optimiser step is taken per epoch, so the
optimisation is *identical* to full-batch training. The backward pass runs per mini-batch of 4 so
only four items' activation graphs are alive at once. This bounds memory (the out-of-memory failures
at K = 16 on large fields were the full-batch activation graph, not the inputs) at the same step
count and roughly the same speed.

### 6.2 The stopping rule, and what it does and does not do

**All K support masks are trained on. There is no validation split.** The stopping rule therefore
detects **convergence, not overfitting**, and the implementation does not pretend otherwise:
overfitting is handled by weight decay and by keeping the head small.

The rule compares **two adjacent windows of 25 epochs** (`patience = 25`) of the training loss:

```
prev = mean(loss[-50:-25]) ;  cur = mean(loss[-25:])
stop when  prev − cur  <  0.002 × |prev|            (min_rel_improve = 0.002)
```

Two *windows* rather than two points, because the flip augmentation makes any single step's loss
noisy enough to trip a point-wise test early. Stopping cannot fire before epoch 50.

### 6.3 Cost

Fitting is the dominant cost. On one idle RTX A5000 — both cards verified idle before and after, unlike the
2026-07-31 measurement this replaces — per support set of K = 8: a median of **29 s**
at the smallest field (DSB2018, 256 px) rising to **550 s** at the largest (HRF, 3504 px), with a
range of 22–657 s across individual support draws. Against the self-configuring method the fit is
*slower* on the two small fields (19.7 → 29.2 s on DSB2018, 111.9 → 134.2 s on DRIVE) and faster on
the two large ones (400.3 → 354.9 s on MoNuSeg, 631.9 → 549.7 s on HRF): without the adaptive loss
the plateau rule runs more epochs, which costs more where an epoch is cheap. Inference is faster
everywhere, which is what dropping CLAHE, the channel selection and the outline head predicts. Training to a plateau costs roughly three to five times a
fixed 60-epoch fit (measured 3.18× and 5.38× on the smaller `flat_h128` variant, registry C48;
the reported config has no 60-epoch arm), and that is what buys the accuracy. Inference from cached features is 0.16–5.4 s per image.

---

## 7. Inference

1. Encode the query image at both scales (cached where available).
2. Compute the 35 classical priors at native resolution on the selected channel.
3. Forward through the head to a native-resolution foreground logit map.
4. Threshold at **logit 0.0**, i.e. probability 0.5. The threshold is a **fixed constant**; the
   support-calibrated threshold lever (`taucal`) exists and is **off** in the reported configuration.
5. Morphological post-processing (`morph_post`) is likewise **off**.

For a multi-class head the foreground score would be `logsumexp(foreground logits) − background
logit`; the reported configuration is binary.

---

## 8. Evaluation protocol

| item | value |
|---|---|
| datasets | 11, spanning spheroid and nucleus blobs, decaying spheroids, phase-contrast cells, overlapping *C. elegans*, dense bacteria, retinal vessels, EM membranes, fluorescent filaments |
| seeds | 10 |
| support pool | 20 images loaded **once** per dataset (15 for CTC-U373, whose annotated sequence holds only that many while keeping the test split disjoint); the K masks are subsampled from that fixed pool per seed |
| test set | the **full** test split, 14 to 148 images, loaded once and never shifted by K |
| metric | foreground IoU for blob / nucleus / worm / bacteria fields; **centreline Dice** for vessels, membranes and filaments |
| statistics | seeds collapsed to one score per test **image** before testing, so the unit of analysis is the image and not the seed-image pair (this avoids a tenfold inflation of the p-values); paired Wilcoxon signed-rank, Holm-corrected within each claim's family |

The critical protocol detail is that the support pool and the test split are loaded **once** with a
fixed seed and the K masks are subsampled from the pool. Loading `load_dataset(spec, K, test, seed=seed)`
per seed instead shifts the test slice on download-kind datasets and undersells the method.

---

## 9. Instance decoding (present, not reported)

`instance_mode = "affinity"` is active in the reported configuration but produces the *instance*
readout, which the paper does not report. It is a SAM-free affinity-watershed decoder: foreground
from the head → distance-transform markers → Frangi-ridge watershed → merge veto using DINO feature
affinity. Two scalars are calibrated from the support masks (the median instance radius, and the
90th-percentile cosine similarity between adjacent ground-truth instances' mean DINO features).

It was measured against the trained specialists and deliberately excluded from the paper: it trails
all three on DSB2018 and two of three on CTC-U373, and adding it would put a second losing axis into
the paper.

---

## 10. Reproduction

```bash
python scripts/sota_final.py run \
  --method head_fusion_best_nobank_noloss_nocolor_flat_es_ep500_wd4_do5_mix_noaux \
  --seeds 10 --pool 20 --test 10000 --support 8 --res 672 \
  --cache <feature-cache-dir> --score_dir results/final10/lean_k8

python scripts/sota_final.py stats \
  --ours head_fusion_best_nobank_noloss_nocolor_flat_es_ep500_wd4_do5_mix_noaux
```

The configuration string is parsed token by token in `scripts/al_testbed.py::make_backend`, which
**rejects an unknown token** rather than ignoring it, so a typo cannot silently produce a different
method. Tokens that resolve to their own default are also rejected, because such an arm would be
byte-identical to the baseline and would then be logged as though it were an independent A/B.

Meaning of each token in the reported string:

| token | effect |
|---|---|
| `head_fusion_best` | the folded-in base: adaptive loss, colour selection, CLAHE, affinity instance decoder, scale fusion, guided upsampler, outline head |
| `noloss` | switches the adaptive-loss constructor **off** — the objective becomes Dice + cross-entropy + 0.5·clDice, fixed for every dataset |
| `nocolor` | switches the colour/stain channel selection and CLAHE **off** — the bank reads grayscale |
| `noaux` | removes the auxiliary outline head |
| `nobank` | disables the tubularity-gated bank *unfreeze* (dropped: net-negative on quality and a severe latency liability — HRF hung for over an hour) |
| `flat` | 1×1 stem convolutions instead of 3×3 |
| `es` | plateau stopping instead of a fixed epoch count |
| `ep500` | maximum 500 epochs |
| `wd4` | weight decay 10⁻⁴ |
| `do5` | dropout 0.5 |
| `mix` | the nonlinear 1×1 fusion layer |
