# Baselines

This file documents every comparison method of Table 1, Figure 2 and Table 2 of the paper: which code and version
ran, at which input size, with which settings, how its output was turned into a binary foreground mask, and how it
was timed. It also lists **every choice that favours a baseline**, so a reader can judge the comparison.

The baselines were run by the research code (AutoSeg, not public), not by this package. Their score records use the
same record format as Exemplar's (`benchmark/run.py`) and are shipped under `results/`; the record directories named
below are those of the research score tree. `results/` ships the directories that the paper's tables and figures read
(`results/MANIFEST.md`); a directory named here but not shipped is marked so.

## Common protocol

Every method, Exemplar included, uses the same split and the same support masks (see [DATASETS.md](DATASETS.md)):

- the pool and the full test split of each dataset, loaded once with seed 0;
- for each of ten seeds, the support indices `default_rng(seed).choice(len(pool), K, replace=False)`;
- the dataset's metric (foreground IoU, or centreline Dice on DRIVE, HRF, ISBI2012-EM and FISBe) on a binary
  foreground mask at the image's native size;
- a record enters a table only if its test-split fingerprint equals Exemplar's.

The score of a dataset is its mean over test images and seeds; a mean over datasets weights each dataset equally.
Methods are compared with paired Wilcoxon signed-rank tests on per-image scores averaged over the seeds, so that
the unit of analysis is the test image, not the seed-image pair. Holm correction is applied within each group of
comparisons: the 40 comparisons with the four forward-pass methods (4 methods × 10 datasets), and each of nnU-Net,
fine-tuned Cellpose-SAM and Convpaint against Exemplar at K = 8 (10 comparisons each).

All groups are compared at the same annotation budget, not the same computation.

## Choices that favour a baseline

| Baseline | Choice | Effect |
|---|---|---|
| Tyche | scored by the best of its 16 candidates against the **test** mask, by the dataset's metric, as in its paper | an oracle: the test mask picks the candidate |
| INSID3 | run with the conditional-random-field refinement behind its published results, not its code default (bilinear) | with the default, its published anchor (78.8) comes out 1.4 to 2.4 points lower |
| Convpaint | its feature extractor (DINOv2 + VGG16) chosen on **test** images of four datasets | the best of four variants on that screen |
| Cellpose-SAM, StarDist, micro-SAM | training length chosen on a ten-image slice of the **test** split | budgets doubled where that slice scored higher |
| Cellpose-SAM, StarDist, micro-SAM (Fig. 2) | at K = 1, 4 and 16, the better of two training lengths on the figure's **test** mean, per tool and K | two budgets per point |
| Cellpose-SAM, StarDist, micro-SAM | scored on their own foreground map, not on the union of their instance masks | the union drops structures their object models cannot represent (fine-tuned Cellpose-SAM: 0.823 as a foreground map, K = 8 panel mean, far more than as instances) |
| micro-SAM | the predicting checkpoint is the best of the run by a validation metric whose loader reuses the support images | a checkpoint choice the other networks do not get |
| StarDist | its timing leaves out the threshold search after training | shorter fit time |
| Forest on the priors (Table 2) | a random forest over **our** 35 priors (plus the raw colour channels on colour data), the features tuned for these morphologies | stronger features than a generic pixel classifier starts from |

Choices that may work against a baseline are stated where they arise: SegGPT, UniverSeg, Tyche and Convpaint have
no reproduced published anchor; UniverSeg and Tyche see every image at 128 px; StarDist's training length came from
a rule that may have made it too long.

---

## Forward pass, nothing fitted

These methods apply a pretrained model to the support set and fit nothing to the new dataset. Each runs at its own
input size, as released.

### SegGPT

- **Code:** `BAAI/seggpt-vit-large` through Hugging Face `transformers` (`SegGptForImageSegmentation`,
  `SegGptImageProcessor`), its default inference configuration.
- **Input:** a fixed 448 × 448 canvas (the processor resizes); images min-max normalised per image to 8-bit RGB;
  masks binarised.
- **K shots:** SegGPT's own few-shot mode. The K support pairs are batched against K copies of the query with
  `feature_ensemble=True`; support masks without foreground are dropped from the prompt.
- **Output:** each painted mask is post-processed back to the native size (label > 0); the K masks are averaged and
  cut at 0.5.
- **Anchor:** no published number was reproduced; the integration is checked in code only.
- **Records:** `seggpt_k{1,4,8,16}`.

### UniverSeg

- **Code:** the `universeg` package, `universeg(pretrained=True)`.
- **Input:** 128 × 128 grey (channel mean, min-max per image), its training resolution, applied to the whole image;
  support masks resized by nearest neighbour.
- **Output:** the sigmoid map resized bilinearly to the native size and cut at 0.5.
- **Anchor:** none reproduced.
- **Records:** `universeg_k{1,4,8,16}`.

### Tyche

- **Code:** <https://github.com/mariannerakic/tyche>, `tychets(version="v1", pretrained=True)` (the CVPR 2024
  checkpoint).
- **Input:** 128 × 128 grey, as UniverSeg; the K support pairs are its context set.
- **Output:** 16 candidate maps per test image, each resized to the native size and cut at 0.5. **Reported:** the
  candidate that scores best against the test mask by the dataset's metric (Tyche's own metric, an oracle). The
  mean of the 16 candidates, cut at 0.5, is kept as a separate record (`tyche_k*`).
- **Anchor:** none reproduced.
- **Records:** `tyche_bestof_k{1,4,8,16}` (reported), `tyche_k{1,4,8,16}` (mean of candidates; not shipped, no table
  reads them).

### INSID3

- **Code:** the authors' implementation, <https://github.com/visinf/INSID3>, unmodified, called as their own
  `inference_segmentation.py` calls it (`set_reference`, `segment`). Its CRF refinement is
  <https://github.com/netw0rkf10w/CRF>.
- **Backbone:** DINOv3 ViT-L/16. The Hugging Face checkpoint was rewritten into Meta's original `torch.hub` format
  so that their loader runs untouched; `load_state_dict(strict=True)` passes and the features agree with the
  original model to 2.7 × 10⁻⁶.
- **Input:** 1024 px, their setting.
- **Refinement:** CRF on (`--crf-mask-refinement`), the configuration of their published results.
- **K shots:** their documented multi-shot mode, which averages the reference prototypes over the shots.
- **Anchor reproduced:** chest X-ray, one shot, 600 episodes, their script and data: 78.7 mIoU with CRF (seed 0)
  against 78.8 published; 77.3, 76.4 and 77.4 with the bilinear default (seeds 0, 1, 2).
- **Records:** `insid3_k{1,4,8,16}` (method name `insid3_official`).

---

## Fitted without gradient descent

### ilastik

- **Code:** the ilastik software, version 1.4.2 (the official release), its Pixel Classification workflow run
  headless.
- **Features:** all of its filter features, 49 per channel (147 on colour images), computed on the raw intensities
  at native resolution.
- **Classifier:** its default random forest, 100 trees in 8 sub-forests.
- **Labels:** pixels sampled from the support masks, 10,000 foreground and 10,000 background pixels per support image
  (all foreground pixels when an image has fewer).
- **Output:** ilastik's Simple Segmentation: foreground where its probability of the foreground class is above 0.5.
- **Records:** `ilastik142_k8` (method name `ilastik`; Table 1).

### Convpaint

- **Code:** `napari-convpaint` 1.1.0, CatBoost 1.2.10, through `ConvpaintModel.train` / `predict_probas`.
- **Feature extractor:** `combo_dino_vgg` (DINOv2 ViT-S/14 with registers plus VGG16), at scale 1, with the tool's
  own defaults for that extractor. **Chosen on test images:** a screen of four variants (VGG16 layer 0, the tool's
  default; `vgg-m`; `vgg-l`; `combo_dino_vgg`) on DSB2018, MoNuSeg, DRIVE and ISBI2012-EM, seeds 0 to 2, full test
  splits; `combo_dino_vgg` had the best mean (0.706 against 0.693, 0.667 and 0.634).
- **Classifier:** CatBoost, 100 iterations, depth 5.
- **Labels:** each support mask enters as a class-balanced random sample of 20,000 labelled pixels per class (the
  tool trains on scribbles and holds every labelled pixel's features in memory). `tile_annotations=False`: the
  features are extracted once over the whole image, as at prediction; on DSB2018 and DRIVE this gave the same
  score as the default, 40 times faster.
- **Input:** colour as `channel_mode="rgb"`.
- **Output:** foreground = P(foreground class) > 0.5.
- **Anchor:** not reproduced (the authors' benchmark needs the Cellpose dataset, whose download requires accepting
  its terms).
- **Records:** `convpaint_k8`.

### Table 2: forest on the priors

Table 2's "Forest on the priors" (called Prior-bank RF in the timing records) is a random forest over our own 35
priors, in the style of the ilastik workflow. It is **not the ilastik software**, which is the ilastik column of
Table 1 (above). It is compared with the linear readout on the priors ("Priors only").

- **Features:** the prior bank of [METHOD.md](METHOD.md) section 3, on the grey image downscaled so that its longer
  side is at most 1536 px. On colour datasets (when most support images are colour) the three raw colour planes are
  added, 38 features in all.
- **Training:** scikit-learn `RandomForestClassifier`, 100 trees, `min_samples_leaf=4`, on 20,000 pixels per support
  image, half foreground and half background (fewer if a class has fewer pixels).
- **Output:** the foreground probability, upsampled bilinearly to the native size and cut at 0.5.
- **Records:** `ilastik_k8` (method name `ilastik_rf`; Table 2's "Forest on the priors"), `ilastik_k1` (not shipped,
  no table reads it).

### Table 2: forest on the views

Table 2's "Forest on the views" is the classifier of the forest on the priors (100 trees, `min_samples_leaf=4`, 20,000
pixels per support image, longer side at most 1536 px, cut at 0.5) on Exemplar's coarse and fine views (4096
features, no priors), with the magnification decided by Exemplar's rule. It is compared with the linear readout on
the same inputs ("Coarse + fine view only"). Records: `views_rf3_k8`.

---

## Networks trained on the support set

### nnU-Net

- **Code:** nnU-Net v2, version 2.8.1 (the version recorded by its timing runs; 2.8.1 at the time the baseline was
  set up), trained from scratch on the K support masks of each seed.
- **Configuration:** `2d`, the residual-encoder planner (`ResEncUNetPlanner`, plans `nnUNetResEncUNetPlans`),
  `-f all` (no cross-validation: a five-way split of eight images is not validation), binary foreground target,
  `nnUNetTrainer_100epochs`. Everything else is nnU-Net's own self-configuration. `nnUNet_compile=f`, because the
  cluster's GLIBC is older than Triton requires; eager mode changes no result.
- **Training length: 100 epochs** (nnU-Net's epoch is 250 iterations, so 25,000 iterations). Checked on one support
  draw (seed 0) against its default of 1000 epochs: SpheroidJ +0.016 (0.800 → 0.816), BBBC010 −0.014
  (0.644 → 0.630), Bacteria −0.006, MoNuSeg −0.005. Raised on one dataset and lowered on three. On DRIVE, 250 epochs
  changed the score by −0.0015 (0.8121 → 0.8106). The 100-epoch budget does not under-train nnU-Net.
- **Output:** nnU-Net's own prediction of the binary foreground.
- **Records:** `nnunet_k1`, `nnunet_k4`, `nnunet_k8`, `nnunet_k16` (all at 100 epochs).

### The fine-tuned specialists

StarDist, Cellpose-SAM and micro-SAM use no support masks as released. Each is fine-tuned on the same K masks per
seed, starting from the checkpoint its off-the-shelf version uses, with each library's own optimiser, loss,
augmentation and learning-rate schedule. A fresh model is trained for every seed; the bench refuses to score a model
whose weights did not move. Datasets with binary masks (SpheroidJ, Decay, DRIVE, HRF, ISBI2012-EM) are turned into
instance labels by connected components for training.

| | Cellpose-SAM | StarDist | micro-SAM |
|---|---|---|---|
| version | cellpose 4.0.9 (torch 2.14.1) | stardist 0.9.1, csbdeep 0.8.2 (TensorFlow 2.15.1) | micro_sam 1.8.5, torch_em 0.10.0 (torch 2.12.1) |
| start checkpoint | `cpsam` | `2D_versatile_fluo`; `2D_versatile_he` on MoNuSeg | `vit_b_lm`; `vit_l_histopathology` on MoNuSeg |
| learning rate | 1e-5 | 3e-4 | 1e-5 |
| training length at K = 8 (epochs over the support, batch 1) | 800 | 1600 | 50 |
| foreground read-out | `cellprob > 0` (Cellpose's default `cellprob_threshold`) | object probability on StarDist's grid (cell i on pixel i·g, interpolated), cut at a threshold chosen per seed on the eight **support** images with the dataset's metric, from 15 values 0.005 to 0.9 | raw decoder foreground > 0.5 (micro-SAM's default `foreground_threshold`), without the Gaussian smoothing of its instance reconstruction |
| records at K = 8 | `cellpose_ft11sem_k8` | `stardist_ft11sem_k8` | `microsam_ft11sem_k8` |

**Why the foreground map.** The metrics are semantic, and the union of instance masks discards what a tool's object
model cannot represent. Fine-tuned Cellpose-SAM marks DRIVE's vessels correctly in its probability map (centreline
Dice 0.79) while its mask reconstruction keeps almost nothing of them (0.006). One read-out per tool was used for the
whole column, the one with the higher panel mean.

**Training length at K = 8, chosen on test images.** The starting budgets (Cellpose 200, StarDist 400, micro-SAM
50 epochs) came from an earlier sweep on two spheroid datasets. A rule fixed before the run then checked them on
DRIVE, Bacteria and ISBI2012-EM, with 2 seeds, on the first ten test images: a budget was doubled for every
dataset if the doubled budget scored higher by more than 0.02 and by more than the larger seed standard deviation,
and four times the budget was checked once by the same rule. Cellpose rose to 800 (on ISBI2012-EM twice), StarDist
to 1600 (on DRIVE, then on Bacteria), micro-SAM stayed at 50.

Two caveats on StarDist. Its step to 1600 epochs fired on Bacteria, whose first ten test images later turned out to
be the easy end of that dataset. In the same check (scored on instance masks), its 1600-epoch score on the other two
check datasets and on HRF was lower than at 800 epochs. A full semantic panel of StarDist at 800 epochs was not run, so its column may be trained too long.

**micro-SAM settings.** `checkpoint_path=None` with the model type alone, so that the pretrained instance decoder is
kept; `with_segmentation_decoder=True`; `n_iterations = 50 × K`; `early_stopping=None`; 8 objects per batch, 4
sub-iterations; training patches of up to 512 px (at most the smallest support side, on the 16-px grid). With only K
images the validation loader reuses the training images, and the predicting checkpoint `best.pt` is the best of the
run by that validation metric. Inference is untiled, as in the off-the-shelf version, so SAM resizes the whole image
to 1024 px; on SpheroidJ, Decay, HRF, Bacteria and FISBe, whose images exceed that, the model is tested at another
object scale than the 512-px training patches at native scale.

**Figure 2: K = 1, 4 and 16.** Each tool was trained with two budgets per K: the same number of epochs over the
support as at K = 8 (Cellpose 800, StarDist 1600, micro-SAM 50; fixed before these runs) and as many gradient steps as
at K = 8. Gradient steps are epochs × K for all three tools. Per tool and K, the budget with the higher mean over the
figure's ten datasets is drawn; this is a selection on test images in the baselines' favour. The picks (computed from
the records):

| | K = 1 | K = 4 | K = 8 | K = 16 |
|---|---|---|---|---|
| Cellpose-SAM (epochs) | 800 or **6400** | 800 or **1600** | 800 | **800** or 400 |
| StarDist | **1600** or 12800 | 1600 or **3200** | 1600 | **1600** or 800 |
| micro-SAM | 50 or **400** | 50 or **100** | 50 | **50** or 25 |

Records: `{cellpose,stardist,microsam}_ft11sem_k{1,4,16}_e<epochs>`.

---

## Controls not in the tables

**The specialists off the shelf**, with no support masks, scored on the same foreground maps as above:
Cellpose-SAM `cpsam` (`cellprob > 0`), StarDist `2D_versatile_fluo` / `2D_versatile_he` (object probability above
the model's own `prob_thresh`), micro-SAM `vit_b_lm` / `vit_l_histopathology` (raw decoder foreground > 0.5). Each
is deterministic, so one scoring is copied across the ten seeds. Their panel means at K = 8 are far below those of
the fine-tuned versions, both as foreground maps and as the union of instance masks. Without fine-tuning none of
them marks vessels, membranes or filaments well. Records (not shipped, no table reads them): `cellpose_sam_sem`, `stardist_sem`, `microsam_sem` (foreground map) and `cellpose_sam`,
`stardist`, `microsam` (instances).

**Other methods tried and not reported:** Matcher (run in its one-shot setting only, its published FSS-1000 anchor
reproduced; lowest column) and PerSAM.

---

## Timing (last row of Table 1)

Every fit and inference time in the paper was measured under the same conditions:

- **One dedicated NVIDIA RTX A5000 (24 GB).** A holder process kept every other job of ours off the card; each job
  started only after ten idle checks in a row; a sampler recorded the card about three times a second, and a job
  counted only if no other process showed activity on the card during it. The CPU was not dedicated (the load
  average on the 32-core host reached 40 during the campaign), which matters for CPU-bound parts such as
  nnU-Net's augmentation and CatBoost.
- **ilastik runs on the CPU only.** It was timed on the CPU of the same host, with the card idle, on 8 threads, and
  with the host load kept at or below 16.
- **One support draw** (seed 0, K = 8) per method and dataset.
- **Fit** = the method's own fitting call at its Table 1 length. **Inference** = the first five test images, from a
  numpy image to a binary mask at native size, GPU synchronised, after one untimed warm-up on a support image.
- **Every time includes encoding the images.** Exemplar's benchmark starts its clock after the support's coarse
  views are encoded, so its fit time adds eight times the median time of one coarse view; its refit (the eighth mask
  added to a fit of seven) adds one.
- **Table 1 shows the median over the ten datasets.** The image sizes differ by more than a factor of 100, so a
  mean would be set by the two or three largest datasets.

Per method:

| Method | Fit | Image | What is timed |
|---|---|---|---|
| SegGPT | – | 2.2 s | forward pass |
| UniverSeg | – | 43 ms | forward pass |
| INSID3 | – | 1.9 s | forward pass with CRF at 1024 px; the eight references are encoded again for every query, as their `segment()` does |
| Tyche | – | 290 ms | the 16 candidates and their resize; not the oracle choice |
| **Exemplar** | **8.5 s** | **620 ms** | cold fit plus eight coarse views; per-image prediction |
| ilastik | 40 s | 1.3 s | training (filter features of the support images and the forest) and batch prediction, on the CPU |
| Convpaint | 43 s | 4.2 s | CatBoost on the CPU and on the GPU, the faster fit per dataset, after one untimed warm-up training call; prediction on 8 CPU threads |
| nnU-Net | 53 min | 490 ms | full 100-epoch training (preprocessing and training) on every dataset; inference with the fully trained model |
| Cellpose-SAM | 25 min | 490 ms | the training call (800 epochs); model construction outside the clock |
| StarDist | 8 min | 150 ms | the training call (1600 epochs); its threshold search after training not added; warm-up on a zero array of each test shape |
| micro-SAM | 6 min | 270 ms | the training call (50 epochs), including building SAM |
| Forest on the priors (Table 2 only) | 18 s | 910 ms | features and forest (100 trees, 8 jobs) |

Exemplar's fit and refit were timed with the blend rule alone (the options `--blend_rule grid --blend_nogrid` of the
research code), which is what this package computes. The scored runs used `--blend_rule both`, which also records two
other choices in the same fits without changing the reported readout. Per-dataset times of Exemplar are in
[REPRODUCE.md](REPRODUCE.md).

**nnU-Net.** Its fit is the measured time of the full 100-epoch training (preprocessing and training) on every
dataset, under the same protocol: 47 to 174 minutes, median 53 min. Inference uses the fully trained model.

The forest on the priors appears only in Table 2; its time is recorded (`Prior-bank_RF__*`) but printed in no table.

The medians above come from the per-method timing records in `results/c119/timing_all_records/`;
`paper/make_paper.sh` writes them, per dataset, to `paper/out/timing_all_a5000.json` beside `numbers_fitinf.tex`.
