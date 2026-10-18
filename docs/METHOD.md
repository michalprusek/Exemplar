# Method specification

This file specifies Exemplar exactly as this package implements it: the configuration reported in the ISBI 2027
paper. Every constant is listed, and each step names the function that implements it as
`module.function` (modules under `exemplar/`). Section 9 lists the numerically delicate points for anyone who
re-implements the method, and section 10 collects every constant in one table.

Notation follows the paper: f_p is the input row of pixel p, A the weighted Gram matrix, b the weighted target sum,
n the total weight, ρ the foreground fraction, t the local thickness, F_c and F_m the feature grids of the coarse
and the magnified (fine) view.

| Item | Value |
|---|---|
| fitted part | one linear readout of 4132 weights: 2048 coarse-view channels, 2048 fine-view channels, 35 priors, 1 constant |
| fitting | closed form: weighted ridge regression over **every pixel** of the support images; no gradient step, no epochs, no learning rate, no stopping rule |
| choices made from the support | the magnification z of the fine view (from the masks alone); the ridge weight λ of each of two pixel weightings (Newton's method on the exact leave-one-image-out squared error); the blend weight a of the two readouts (exact leave-one-image-out metric). The cut is fixed at 1/2 |
| frozen | the DINOv3 ViT-L/16 backbone and the 35 classical priors; nothing is meta-trained and nothing carries over between datasets |
| update | adding a mask gives the same readout, bit for bit, as a fit from scratch on all masks |
| public API | `model.Exemplar` (`fit`, `add`, `predict`, `score`) |

---

## 1. Task

The model receives K images with complete foreground masks (the *support set*) from one dataset and segments
other images of the same dataset. A separate readout is solved for every support set. The output is a binary
foreground mask at the image's native resolution.

Every annotated pixel of a support image counts as foreground and every other pixel as background, so each support
image needs a complete mask. A mask pixel is foreground when its value is above zero (`readout.label_tensor`).

---

## 2. Frozen backbone

| Item | Value |
|---|---|
| model | `facebook/dinov3-vitl16-pretrain-lvd1689m` (DINOv3 ViT-L/16, a gated Hugging Face checkpoint), loaded in float32 and kept in eval mode (`encoder.Dinov3Encoder`) |
| parameters | 303,129,600, all frozen |
| patch | 16 px |
| encoder input | 672 × 672 px, a 42 × 42 patch grid |
| blocks read | the last hidden state (after the final LayerNorm) and the output of block 12 of 24 (`hidden_states[12]`, before any final norm) |
| tokens dropped | 5 prefix tokens (CLS and 4 registers) |
| features | each block's 1024-vector is L2-normalised separately, the two are concatenated, and the 2048-vector is L2-normalised again; within the 2048 channels, the last hidden state comes first |
| precision | only the transformer runs under bfloat16 autocast; its outputs are cast back to float32 before any normalisation (`encoder.install_bf16_forward`) |

A 16-pixel patch grid cannot give both context and detail on a large image, so every image is read twice.

### 2.1 Coarse view (`encoder.Dinov3Encoder.coarse`)

The whole image is encoded once, giving semantic context over the entire field at a stride of
`native side × 16 / 672` pixels.

1. `encoder._to_rgb01`: a grey image is stacked to three channels, an alpha channel is dropped, and the image is
   min-max normalised over all its pixels and channels: `(x - min) / (max - min + 1e-6)`.
2. `encoder.Dinov3Encoder._preprocess`: bilinear resize to 672 × 672 (`align_corners=False`). The resize is **not**
   aspect-preserving. Then ImageNet normalisation (mean 0.485, 0.456, 0.406; std 0.229, 0.224, 0.225).
3. One forward pass; the two blocks are read and normalised as in the table above. Result: a [42, 42, 2048] grid.

The coarse view is the only input cached on disk (`encoder.CoarseView`, `encoder.EmbeddingCache`). The key is
the SHA-1 of the image bytes, its shape, its dtype and the tag `dinov3-vitl16-pretrain-lvd1689m-res672-Lf-1_12`.
Under bfloat16 the cache directory gets the suffix `_bf16` (`encoder.coarse_cache_dir`), so bfloat16 and float32
features never share a directory. Entries are written to a temporary file and published by an atomic rename.

### 2.2 Fine view (`fineview.fine_view`)

The image is magnified by an integer factor z (section 2.3) and encoded in overlapping 672-px windows. Each patch
then covers 16/z native pixels, whatever the field size.

1. The image is converted to float32 without any global normalisation.
2. **z > 1 (registered).** The image is magnified bilinearly with half-pixel centres (`align_corners=False`) to
   sides rounded **up** to multiples of 16: `ceil(H z / 16) · 16` by `ceil(W z / 16) · 16`. Windows step by
   496 px, a multiple of 16, so every window lies exactly on the patch grid.
3. **z = 1 (native tiling).** Windows step by 504 px (a quarter overlap, `fineview.tile_grid`).
4. In both cases the window origins are 0, step, 2·step, ..., and one more window is added so that the last one
   ends at the image edge (`fineview._positions`). A side shorter than 672 px gets one window.
5. Each window is reflect-padded to 672 px where the image is smaller (numpy's `reflect` mode for any pad width,
   `fineview._reflect_idx`); a grey window is repeated to three channels and an alpha channel dropped; the window is
   min-max normalised over all its pixels and channels (`+1e-6` in the denominator) and ImageNet-normalised.
6. Six windows per forward pass. The two blocks are read and normalised as in the table above.
7. **Stitching.** The grid has `max(1, floor(h / 16))` by `max(1, floor(w / 16))` cells, h and w being the sides of
   the (magnified) image. Each window's 42 × 42 grid is placed at cell `(floor(y / 16), floor(x / 16))`, cropped at
   the grid's edge, and overlapping cells are averaged. The averages are not re-normalised.
8. **Cap.** If the longer grid side exceeds `ceil(2560 z / 16)` cells (`fineview.fine_max_grid`; 2560 native
   pixels at any z, 160 cells at z = 1), the grid is resampled bilinearly by the factor `cap / longer side`:
   to the explicit size `(max(1, round(h_g · s)), max(1, round(w_g · s)))` when registered, by `scale_factor`
   otherwise. On the paper's panel only HRF reaches the cap.

### 2.3 Magnification from the support masks (`fineview.decide_zoom`)

A patch cannot describe a structure narrower than itself, and the annotated structures range from vessels a few
pixels wide to whole spheroids. The fine view is therefore read at the **smallest integer z at which a patch-wise
drawing of the annotation is good enough**. z is decided at every fit, from the K support masks alone (never from
other images), before any fine grid is read, and it is used for every image the readout reads.

- **Pixel weights** (`fineview._thickness_weights`): `u_p = 1 / max(t_p, 1)`, where t_p is the local thickness of
  the annotated structure at the pixel's nearest skeleton pixel (twice the Euclidean distance transform of the
  mask there). Background pixels take the value of their nearest skeleton pixel too. A structure of thickness t and
  length L covers about t·L pixels, so its total weight is about L: a structure counts by its length, not its area.
  A mask with fewer than 4 foreground pixels, or without a skeleton, is skipped.
- **Weighted patch Dice** (`fineview._patch_dice_weighted`): the mask and its weights are magnified by z (nearest
  neighbour) and tiled into 16-px cells. With g_c the weighted foreground and a_c the total weight of cell c, and G
  the weighted foreground of the mask,

  ```
  D*_u(z) = max over sets S of cells of  2 Σ_{c∈S} g_c / (G + Σ_{c∈S} a_c)
  ```

  the best Dice that one label per patch can reach, every pixel weighted by u. The optimum is a prefix of the cells
  sorted by g_c / a_c in descending order (exact). D* is averaged over four phases of the grid: offsets
  (0, 0), (8, 8), (0, 8) and (8, 0) px.
- **Rule.** `z = the smallest z in 1, ..., z_top with median over the support of D*_u(z) ≥ 0.8`, where
  `z_top = max(1, min(4, floor(8192 / longest support side)))`: z is at most 4, and the magnified longest side at
  most 8192 px, because the cost of the fine view grows with z². If no z reaches 0.8, z = z_top. If no support mask
  has a measurable structure, z = 1.

The per-mask work runs in a thread pool of `OMP_NUM_THREADS` workers (default 8, `fineview._executor`); the result
does not depend on the number of threads. For an exact update, each mask keeps its weights and its D* values per z
(the `measures` argument), so adding a mask measures only the new one.

---

## 3. Frozen classical priors (`priors.priors`, `priors.PriorBank.feature_maps`)

Patch features describe the image only at the spacing of their grid. A fixed bank of 35 classical filter responses
supplies detail at native resolution. It runs on the whole native image in one pass: no downscaling, no tiling.
Nothing in it is fitted and its scales are the same for every dataset.

**Input.** The grey image is the mean over the image's channels, min-max normalised per image (`priors._gray01`),
then mapped to [0, 1] between its 1st and 99th percentiles and clipped (`priors.normalize_image`, computed with
`torch.quantile`).

**Channels**, in the order of the readout's weights. Each family is standardised per image by GroupNorm with one
group, no affine parameters and eps 1e-5.

| Family | Channels | Scales | Detail |
|---|---|---|---|
| Frangi vesselness (`priors.FrangiBank`) | 10 | σ = 1, 2, 4, 8, 16 px | on the image × 255; Hessian scaled by σ²; β = 0.5, γ = 15; per σ: dark-on-bright (λ₂ > 0) then bright-on-dark |
| Sauvola threshold (`priors.SauvolaBank`) | 6 | windows 15, 51, 151 px | `T = μ (1 + k (σ / R − 1))`, k = 0.2, R = 0.5; output `sigmoid(16 (T − I))`; per window r then 1 − r |
| structure tensor (`priors.StructureTensorBank`) | 4 | σ = 2, 8 px | per σ: λ₁ then λ₂ (λ₁ ≥ λ₂) |
| intensity (`priors.IntensityBank`) | 2 | | the normalised image and its complement |
| gradient magnitude (`priors.GradMagBank`) | 3 | σ = 0.001, 0.004, 0.016 × min(H, W) | scale-normalised (× σ) |
| Laplacian of Gaussian (`priors.LoGBank`) | 10 | σ = 0.001, 0.002, 0.004, 0.008, 0.016 × min(H, W) | scale-normalised (× σ²); per σ: signed then absolute |

A scale below 1 is a fraction of min(H, W) (`priors._resolve_sigma`), so the gradient and LoG scales grow with
the field (LoG 0.26 to 4.1 px on a 256-px image, 2.3 to 37 px on HRF's 2336-px short side). The others are pixels.

Building blocks: Gaussian kernels of size `max(3, int(8σ + 0.5) + 1)`, made odd (`priors._gaussian_kernel_1d`),
applied separably after reflect padding (replicate padding where the image is narrower than the pad,
`priors.reflect_pad`); derivatives are central differences `[-0.5, 0, 0.5]` with reflect padding; Sauvola's local
mean and variance are box averages over the window. Frangi's β and γ and Sauvola's k are stored as frozen float32
parameters (β and γ as logarithms), exactly as in the original bank, so `exp(log(0.5))` rounds as it did there.

`priors.PRIOR_CHANNELS` lists the 35 channel names and `priors.FAMILY_SLICES` the slice of each family.

---

## 4. The readout, solved in closed form (`readout.ridge_fit`, `readout.solve`)

### 4.1 Rows and objective

Every pixel p of every support image gives one row

```
f_p = [U_c F_c ; U_m F_m ; B_p ; 1]        d = 2048 + 2048 + 35 + 1 = 4132
```

the coarse and the fine grid interpolated bilinearly to the pixel, the 35 priors at the pixel, and a constant
(`readout.gather`, `readout.rows`). Pixel i of a side of n pixels reads grid coordinate
`(i + 0.5) · n_grid / n − 0.5`, clamped to `[0, n_grid − 1]`. This is `grid_sample(bilinear, align_corners=False,
padding_mode="border")`, the same interpolation the prediction uses (section 7).

With y_p ∈ {0, 1} the mask and q_p ≥ 0 a pixel weight,

```
w = argmin_w (1/n) Σ_p q_p (y_p − w'f_p)² + λ w'P w  =  (A/n + λ P)⁻¹ (b/n)
A = Σ_p q_p f_p f_p'      b = Σ_p q_p y_p f_p      n = Σ_p q_p      P = diag(1, ..., 1, 0)
```

in coordinates where every column except the constant is standardised over the support pixels (section 4.4). The
constant is not penalised. Two weightings come from the same sums:

- **plain**: q_p = 1, so the score estimates the probability of foreground;
- **balanced**: foreground and background carry equal total weight, `q = 1/(2ρ)` on the foreground and
  `1/(2(1 − ρ))` on the background, ρ being the foreground fraction of the images in the fit, clamped to
  `[1e-6, 1 − 1e-6]`.

### 4.2 Column shift (`readout.shift_sample`, `readout.column_shift`)

The DINOv3 channels have a mean far larger than their spread, and products of raw columns in reduced precision lose
definiteness. Before any product, every column except the constant is therefore shifted and scaled by fixed
constants: the mean and the **unbiased** standard deviation (n − 1) of 4096 pixels of the first support image,
drawn without replacement (all pixels if the image has fewer); a scale below 1e-6 is replaced by 1. Any fixed
affine map is exact here, and the solution is mapped back (`readout.unshift`).

The 4096 pixels are the readout's only random draw. `Exemplar.fit` takes the generator as `rng`; the benchmark
passes the generator that drew the support indices, so the support draw comes first and the pixel sample second.

### 4.3 Per-image sums, exact on the patch grids (`readout.image_sums`)

A and b are sums over images, so each support image keeps its own raw sums: its Gram matrix G over all its pixels,
its foreground Gram Gf, its foreground sum bb, and its pixel and foreground counts. Any weighting and any
standardisation follow from these (section 4.4).

- **All pixels, on the grids** (`readout.grid_gram`, `readout.interp_matrix`). A grid column read at every pixel is
  X = U F with `U = U_y ⊗ U_x`, because bilinear interpolation factorises over rows and columns. So
  `Σ_p (UF)(UF)' = F' (U_y'U_y ⊗ U_x'U_x) F`, and the cross blocks between the two grids and with the priors
  likewise. The cost of the feature blocks is set by the grid cells, not the pixels. Only the 35 prior columns are
  read pixel by pixel; their own block is accumulated in chunks of 2²⁰ pixels. All products in float32 with TF32
  **off**. The result equals the pixel-by-pixel Gram to float32 rounding (`tests/test_grid_gram.py`).
- **Foreground pixels, directly** (`readout.gmm`). Gf is formed from the foreground rows in chunks of 65,536 rows.
  The rows are rounded to bfloat16 on the way in and multiplied with float32 output and TF32 **on**; only the upper
  block triangle is computed, in 8 column blocks with edges `round(i · d / 8)`, and mirrored. bb is summed from the
  same rows **unrounded**, in float32.
- **K = 1.** With a single support image, each of its four quadrants (rows split at `floor(H/2)`, columns at
  `floor(W/2)`) keeps its own sums, so that the quadrants can be left out (section 5.1).

### 4.4 Standardisation and solves (`readout.totals`, `readout.tstd`, `readout.tback`, `readout.chol_solve`)

The mean μ and the population standard deviation σ of each column come from the summed raw Grams:
`μ = G[const, :] / n`, `σ = sqrt(max(diag(G)/n − μ², 0))`, with σ < 1e-6 replaced by 1. The weighted Gram of a
weighting is `q_bg (G − Gf) + q_fg Gf`, its target sum `q_fg bb`, its weight `q_bg (n − n_fg) + q_fg n_fg`.

The standardisation x ↦ (x − μ)/σ is applied to a Gram matrix and a target sum in O(d²) and O(d)
(`readout.tstd`), never as a d³ product, and mapped back from the solution (`readout.tback`, then
`readout.unshift` to the raw columns).

The full system `(A/n + λP) w = b/n` is solved by a Cholesky factor in **float32**, with no refinement; if the
float32 factor is not definite, one float64 factor is tried (`readout.chol_solve`, counted as
`single_fallbacks`), and if that also fails the fit stops with an error.

---

## 5. Choices made on held-out support images

### 5.1 Exact leave-one-image-out (`readout.fold_pieces`)

The fold without image k is a fit from scratch on the other images. Its standardisation (μ and σ of the 4131
non-constant columns), its foreground fraction ρ and therefore its balanced weights all come from the other images'
raw sums (`G − G_k`, `Gf − Gf_k`, counts). Its system is built from those sums in those coordinates and weights.
Image k's own sums are standardised and weighted with **that fold's** μ, σ and weights, and its weighted squared
error follows from them:

```
w_{-k} = ((A − A_k)/(n − n_k) + λP)⁻¹ (b − b_k)/(n − n_k)
E_k    = (c_k − 2 w_{-k}'b_k + w_{-k}'A_k w_{-k}) / n_k        c_k = q_fg · n_fg,k   (Σ q y² with y ∈ {0, 1})
```

No pixel is read twice. With K = 1 the four quadrants of the single image are the folds.

### 5.2 λ by Newton's method (`readout.heldout_cv`, `readout.newton_lam_folds`, `readout.newton_outer`)

For each weighting, `CV(λ)` is the mean of E_k over the folds. With `M_k = (A − A_k)/(n − n_k) + λP` for a fold,
one float32 Cholesky factor gives

```
w = M_k⁻¹ r      dw/dλ = −M_k⁻¹ P w      d²w/dλ² = −2 M_k⁻¹ P (dw/dλ)
```

so CV' and CV'' are exact. Newton works on t = log λ: `CV_t' = λ CV'`, `CV_t'' = λ² CV'' + λ CV'`.

- Start at λ = 1. If a fold's factor is not definite there, step up by one decade until it is.
- Step `−CV_t' / CV_t''`, or one decade downhill where `CV_t'' ≤ 0`; at most one decade (ln 10 in t); λ kept in
  `[1e-4, 100]`.
- Halve the step up to six times while a fold's factor is not definite or CV rises.
- Stop when the step falls below 0.01 in t (about 1 % of λ) or after 30 steps.
- A λ at which any fold's float32 factor is not definite cannot be chosen (counted as `single_rejected_evals`).
- **The chosen λ** is the evaluated point with the lowest CV over all definite evaluations, including those at
  which CV rose and the step was halved; ties go to the **larger** λ.

### 5.3 The blend of the two readouts, cut at 1/2 (`readout.choose_blend`)

The two weightings give two readouts, w_p (plain) and w_b (balanced), each with its own λ. Exemplar uses one
readout,

```
w_a = a · w_p + (1 − a) · w_b,   a ∈ [0, 1],   mask = s > 1/2
```

so a = 1 is the plain readout at 1/2 and a = 0 the balanced one. A blend moves both the direction of the readout
and how much of the image passes the cut, so this one number replaces the choice of weighting and a search for a
threshold.

**Held-out maps.** For each held-out image (or quadrant), the fold's solution at the chosen λ is mapped back with
that fold's μ and σ, and its score map is computed on the image's kept inputs (`readout.logit_map`). With K = 1
the map of the whole image is cropped to the quadrant, and quadrants without foreground are skipped. Because both
weightings' folds are standardised from the same images and the score map is affine in the weights, the held-out
map of the blended fold readout is exactly `a · L_p + (1 − a) · L_b` of the two held-out maps. Choosing a needs no
further solve.

**Choice of a.**

1. a on 0, 0.2, 0.4, 0.6, 0.8, 1 (rounded to three decimals). Each held-out blend map is cut at 1/2 and scored
   against its mask with the chosen metric (`readout.metric_masks`): foreground IoU, or centreline Dice with a
   skeleton identical to `skimage.morphology.skeletonize` computed on the device (`readout.zs_skel`). The edge
   cases are those of `benchmark/metrics.py`.
2. Then a ± 0.1 and a ± 0.05 around the best, inside [0, 1].
3. The best a is the one with the highest mean over the held-out images among all evaluated values; ties go to the
   **larger** a. The blended weights are mixed in float64 and cast to float32.
4. **One-standard-error rule.** Let d be the per-image difference between the best a and a = 1. The best a replaces
   a = 1 only if `mean(d) > std(d, ddof=1) / sqrt(number of held-out images)` (strict; the standard error is 0
   with a single held-out image). Otherwise the readout is the plain readout, cut at 1/2. A nearly flat held-out
   curve thus cannot lead the choice into noise.

The metric is the user's choice for the structures at hand (`Exemplar(metric="iou" | "cldice")`). The paper's
experiments use the metric by which each dataset is scored.

`ridge_fit` returns four readouts: `plain_mse` and `bal_mse` (each weighting at its λ, cut at 1/2), `blend_ex`
(the best a without the one-standard-error rule) and `blend1_ex`, **the reported readout**.

---

## 6. Adding a mask (`model.Exemplar.add`)

The fit keeps, per support image, its raw sums, its inputs (for the held-out maps) and its magnification
measurements. `add(image, mask)`:

1. re-runs the magnification rule on the K masks, measuring only the new one;
2. if z is unchanged, reads only the new image and adds its sums (with the same column shift); the
   standardisation, both weightings' folds, λ (Newton started at λ = 1, where a cold fit starts) and a are all
   recomputed from the updated sums (`readout.ridge_fit` with `state`);
3. if z changed, the kept fine grids are stale, so the support is refitted from scratch with the same column-shift
   sample. The same happens when the previous fit had one mask (its sums are per quadrant) or after
   `release_support()`.

With the same sums and the same starting λ every later step is the same computation, so **the result is
bit-identical to a cold fit of the same K masks** with the same column-shift sample (`tests/test_refit.py`, also
across a change of magnification).

---

## 7. Inference (`model.Exemplar.predict`, `model.Exemplar.score`, `readout.logit_map`)

A new image is read at the support's magnification z. Each grid is reduced to one channel by its weights, that
channel is interpolated bilinearly to the image (`align_corners=False`), and the priors' weighted sum and the
constant are added:

```
s = U(F_c w_c) + U(F_m w_m) + B w_B + w_0        mask = s > 1/2
```

This equals interpolating every channel and then weighting, because both steps are linear; the 4132 maps are never
formed. The standardisation is folded into the weights.

---

## 8. What the API exposes

- `Exemplar.readout`: the reported readout, a dict with `variant` (`"plain"` or `"baNNN"` for a blend with
  a = NNN/100), `a`, `lam` (λ of the plain weighting), `lam_bal` (λ of the balanced one, blends only), `tau`
  (always 0.5), `heldout` (the mean held-out metric), `w` (float32, on the raw columns
  `[coarse | fine | priors | constant]`), `per_img`, `gain` and `se` (the one-standard-error test).
- `Exemplar.weights`: `readout["w"]`.
- `Exemplar.info`: K, z, a, variant, both λ, the cut, the held-out metric, gain, standard error, and the counters
  `single_fallbacks` and `single_rejected_evals`, each present only when it is non-zero.
- `Exemplar.result`: all four readouts of section 5.3.

---

## 9. Numerics for re-implementers

The package reproduces the research code bit for bit on the same GPU and library versions. These points decide
that equality; change any of them and the numbers move, usually in the last digits, sometimes in a discrete choice.

- **Weight layout.** `[coarse 2048 | fine 2048 | priors 35 | constant]`; within each 2048 block,
  `[last hidden state 1024 | block 12 1024]`.
- **Two precisions of the fine grid.** The Gram sums use the float32 fine grid; the held-out maps use the fine grid
  kept in bfloat16 and cast back to float32.
- **Two kinds of matrix products.** The whole-image Gram (`grid_gram`) runs in float32 with TF32 off; the
  foreground Gram (`gmm`) rounds its rows to bfloat16 and runs with TF32 on. Both functions restore the TF32
  switch afterwards. The cuDNN TF32 switch is left at PyTorch's default; it affects the priors' convolutions on
  GPUs with TF32.
- **Two standard deviations.** The fixed column shift uses the unbiased standard deviation of 4096 pixels; the
  standardisation uses the population σ of all support pixels.
- **Random draws.** One generator draws the support indices first and the 4096-pixel sample second.
- **Resizes.** The coarse view is resized to 672 × 672 without keeping the aspect ratio. The fine view is magnified
  bilinearly with half-pixel centres to sides rounded up to multiples of 16. The cap resamples to an explicit
  rounded size when registered and by `scale_factor` otherwise.
- **Encoder batches.** Six fine-view windows per forward pass, in raster order of their origins.
- **Ties.** λ ties go to the larger λ, a ties to the larger a; the one-standard-error test is strict, with
  `ddof=1`.
- **Skeletons.** The device skeleton (`readout.zs_skel`) reproduces the lookup table of scikit-image 0.26's 2-D
  `skeletonize`; `tests/test_numerics.py` checks it against the installed scikit-image. The test metric uses
  scikit-image directly.
- **Masks.** Foreground is `mask > 0`; a mask is resized (nearest neighbour) only if its shape differs from the
  image's.
- **Solves.** Float32 Cholesky without refinement. A fold that is not definite in float32 makes that λ
  unchoosable; only the full solve has a float64 fallback. In the paper's score records (all K and all
  ablations, 10 seeds each) the fallback never fired. Newton evaluations were rejected 5 times at K = 8 (all on
  Decay), 47 times at K = 16 (all on Decay) and once at K = 4 (SpheroidJ).
- **Large images.** `torch.quantile` in `priors.normalize_image` refuses inputs above 2²⁴ elements, so the priors
  fail on images with more than 16,777,216 pixels (for example 4096 × 4096). Priors above 5 × 10⁷ elements are
  kept on the host for the held-out maps (`readout.HOST_BANK_ELEMENTS`); this changes memory, not numbers.

---

## 10. Constants

None of these is read from the data; they are the same for every dataset.

| Constant | Value | Where |
|---|---|---|
| backbone | `facebook/dinov3-vitl16-pretrain-lvd1689m` | `encoder.MODEL_ID` |
| encoder input, patch, prefix tokens | 672 px, 16 px, 5 | `encoder.RESOLUTION`, `PATCH`, `N_PREFIX` |
| blocks | last hidden state and block 12 | `encoder.LAYERS = (-1, 12)` |
| fine-view window steps | 496 px (z > 1), 504 px (z = 1) | `fineview.fine_view` |
| windows per forward | 6 | `fineview.TILE_BATCH` |
| fine-grid cap | `ceil(2560 z / 16)` cells per side | `fineview.fine_max_grid` |
| magnification: patch Dice level, maximum z, side budget | 0.8, 4, 8192 px | `fineview.ZOOM_TAU`, `ZOOM_MAX`, `SIDE_BUDGET` |
| magnification: minimum foreground, grid phases | 4 px, 4 | `fineview._thickness_weights`, `_patch_dice_weighted` |
| Frangi σ, β, γ | 1, 2, 4, 8, 16 px; 0.5; 15 | `priors.FRANGI_SIGMAS`, `FrangiBank` |
| Sauvola windows, k, R, temperature | 15, 51, 151 px; 0.2; 0.5; 16 | `priors.SAUVOLA_WINDOWS`, `SauvolaBank` |
| structure-tensor σ | 2, 8 px | `priors.STRUCT_SIGMAS` |
| gradient-magnitude σ | 0.001, 0.004, 0.016 × min(H, W) | `priors.GRADMAG_SIGMAS` |
| LoG σ | 0.001, 0.002, 0.004, 0.008, 0.016 × min(H, W) | `priors.LOG_SIGMAS` |
| percentile clip, GroupNorm eps | 1 % and 99 %, 1e-5 | `priors.normalize_image`, `PriorBank` |
| readout dimension | 4132 | |
| column-shift sample | 4096 pixels of the first support image | `readout.SHIFT_PX` |
| foreground rows per product | 65,536 | `readout.CHUNK_PX` |
| foreground-Gram column blocks | 8 | `readout.gmm` |
| prior-block chunk | 2²⁰ pixels | `readout.grid_gram` |
| λ: start, bounds | 1; [1e-4, 100] | `readout.LAM_START`, `LAM_MIN`, `LAM_MAX` |
| Newton: step cap, halvings, stop, steps | one decade; 6; 0.01 in log λ; 30 | `readout.newton_outer` |
| ρ clamp, scale floor | [1e-6, 1 − 1e-6]; 1e-6 | `readout.solve`, `readout.fold_pieces` |
| blend: grid, refinement | 0, 0.2, ..., 1; ± 0.1, ± 0.05 | `readout.choose_blend` |
| cut | 1/2 | `readout.choose_blend` |
| host-memory threshold for held-out priors | 5 × 10⁷ elements | `readout.HOST_BANK_ELEMENTS` |
