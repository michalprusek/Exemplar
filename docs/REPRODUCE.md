# Reproducing the paper

There are two ways to reproduce the paper:

- **Regenerate every table, figure and number from the shipped score records** (section 3). This needs no GPU,
  no data and no model weights.
- **Re-run Exemplar** on the ten datasets and compare the new records with the shipped ones (sections 1 and 2).
  This needs a CUDA GPU, the datasets laid out as in [DATASETS.md](DATASETS.md) and access to the gated DINOv3
  checkpoint (see the [README](../README.md)).

The baselines were run by the research code and are not re-runnable from this package; their settings are in
[BASELINES.md](BASELINES.md).

## 1. Re-run Exemplar (Table 1, Figure 2, Figure 3)

`benchmark/run.py` implements the paper's protocol (see [DATASETS.md](DATASETS.md), "The split protocol"): per
dataset, the pool and the test split are loaded once; per seed, the K support masks are drawn, the magnification is
decided and the readout fitted; every test image is then scored with the dataset's metric. It writes one record per
dataset, `<score_dir>/ridge__<dataset>.json`, in the format the paper's generators read.

The arms of Table 1 (K = 8) and Figure 2 (K = 1, 4, 8, 16), ten seeds each, all ten datasets:

```bash
export ASG_DATA_ROOT=... SPHEROID_ROOT=... PANEL_DL_ROOT=...    # see DATASETS.md
export OMP_NUM_THREADS=8                                       # threads of the magnification rule and the scoring
for K in 1 4 8 16; do
  python -m benchmark.run --k $K --seeds 10 --gpu 0 \
      --cache /path/to/cache --score_dir out/ridge4_k$K
done
```

- `--datasets drive,hrf` restricts a run to some datasets; one process per dataset is a convenient unit of work.
- A dataset whose record already exists in `--score_dir` is skipped, so an interrupted run can be restarted.
- `--cache` holds the coarse view of every image, keyed by the image bytes; the run uses `<cache>_bf16`. Entries
  are written atomically, so processes may share a cache. Only the coarse view is cached; the fine view and the
  priors are recomputed at every fit.
- `--gpu` sets `CUDA_VISIBLE_DEVICES` before torch is imported.

The paper's records were produced by the research code on NVIDIA L40S GPUs (46 GB). Their directory names in the
research score tree are `ridge4_k1`, `ridge4_k4`, `ridge4_k8` and `ridge4_k16`.

**Figure 3 (predictions on eight datasets)** shows seed 0's prediction on one test image per dataset. To dump those
panels, run seed 0 alone with `--dump_qual`, **into its own score directory**: a dataset whose record already exists
is skipped, and then nothing is dumped.

```bash
python -m benchmark.run --k 8 --seeds 1 --gpu 0 --cache /path/to/cache \
    --datasets rozpad,spheroidj,dsb2018,monuseg,bbbc010,drive,isbi2012em,fisbe \
    --score_dir out/qual_records --dump_qual_dir out/qual \
    --dump_qual rozpad_0,spheroidj_0,dsb2018_2,monuseg_1,bbbc010_2,drive_2,isbi2012em_2,fisbe_1
```

Each key is `<dataset>_<test index>`. The file `out/qual/<key>.npz` holds the image, the annotation (`gt`), the
prediction (`pred_fg`) and the dataset, metric, seed and K. Seed 0 of a one-seed run is the same fit as seed 0 of a
ten-seed run. For ISBI2012-EM and FISBe the dumped image is the test image whose score is closest to the seed-0
median of its dataset.

## 2. Table 2 (ablations at K = 8)

| Row of Table 2 | How to get it |
|---|---|
| Coarse + fine view only | `--blocks coarse,fine` |
| Priors only | `--blocks bank` |
| Coarse view + priors | `--blocks coarse,bank` |
| Fine view + priors | `--blocks fine,bank` |
| Last layer only | `--last_layer_only` |
| Fine view not magnified | `--zoom_fixed 1` |
| Plain readout, threshold 0.5 | no new run: `alt["plain_mse"]` of the K = 8 records |
| Balanced readout, threshold 0.5 | no new run: `alt["bal_mse"]` of the K = 8 records |
| Forest on the views, Forest on the priors | separate baselines, not in this package (see [BASELINES.md](BASELINES.md)) |
| Exemplar | the K = 8 records of section 1 |

Each input row is one run of the K = 8 command of section 1 with one option added, for example

```bash
python -m benchmark.run --k 8 --seeds 10 --gpu 0 --cache /path/to/cache \
    --blocks coarse,fine --score_dir out/ridge4_nocls_k8
```

The records carry the option in their protocol (`blocks`, `zoom_fixed`, `feat_method`), and the ablation table's
generator checks it. The research score tree names these arms `ridge4_nocls_k8`, `ridge4_bankonly_k8`,
`ridge4_coarseonly_k8`, `ridge4_fineonly_k8`, `ridge4_nolf12_k8` and `ridge4_nozoom_k8`.

## 3. Regenerate the paper's tables, figures and numbers

```bash
pip install -e ".[paper]"          # matplotlib, for Figure 2
bash paper/make_paper.sh            # writes into paper/out/ (or: bash paper/make_paper.sh <out_dir>)
```

From the shipped score records under `results/` (listed with what reads them in `results/MANIFEST.md`), it regenerates:

- `tab_fewshot.tex` (Table 1, including its last row of fit and inference times),
- `tab_ablation.tex` (Table 2),
- the `numbers_*.tex` macro files that hold every number quoted in the text (statistics, timing, ablation),
- `fig_kscale.tex` and `figures/kscale.pdf` (Figure 2),
- Figure 1's thumbnails and Figure 3's panels, only if you supply the qualitative dumps: they hold test images of
  the datasets, which this repository does not redistribute. Make them with section 1's dump command into
  `results/c119/qual_v4/` (or set `QUAL_DIR`); without them `make_paper.sh` skips these two steps.

The outputs equal the paper's files (version of 18 October 2026) byte for byte (`paper/expected_outputs.sha256`,
checked by `tests/test_paper.py`; the PDF differs only in its creation date, the PNGs need the same matplotlib and
Pillow).

Every number in the paper is a macro written by these generators, so a re-run updates the paper mechanically. The
generators refuse to pair records whose test-split fingerprints differ.

To check a re-run of section 1 against the paper, compare its records with the shipped ones dataset by dataset:
same `split_fp`, same `zooms_decided`, and the per-image scores (`per_image`, seed-major) equal up to the GPU's
floating-point behaviour (section 4).

## 4. Tests and the golden comparison

### The CPU tests

```bash
pip install -e ".[test]"
pytest
```

The tests need neither a GPU nor the DINOv3 weights: they feed the readout small synthetic grids and maps
(`tests/synth.py`), which exercise the same code that runs on real inputs. They check:

| Test file | What it checks |
|---|---|
| `test_grid_gram.py` | the Gram matrix computed on the patch grids equals the Gram of the explicitly interpolated pixel rows, for a whole image and for quadrants |
| `test_loo.py` | the leave-one-image-out errors computed from the stored sums equal brute-force refits without that image, with each fold's standardisation and class weights from the other images; the per-image sums equal the explicit sums of the rows |
| `test_refit.py` | adding a mask gives the same readout, bit for bit, as a cold fit of all masks, also through `Exemplar.add` and across a change of magnification |
| `test_blend.py` | the blend choice on constructed held-out maps: the grid, the refinement, ties to the larger a, and both branches of the one-standard-error rule |
| `test_priors.py` | the 35 prior channels come out in the documented order, each family with its own normalisation |
| `test_numerics.py` | the device skeleton equals `skimage.morphology.skeletonize`; the blocked foreground Gram equals X'X of the bfloat16-rounded rows |
| `test_ablations.py` | the ablation options of Table 2: a readout on some column blocks refits exactly and gives the removed blocks weight 0, an excluded fine view never reaches the readout, a fixed magnification holds through `add`, and the runner records the options and scores the two readout rows as the ablation table's generator expects |
| `test_paper.py` | `paper/make_paper.sh` on the shipped records reproduces the paper's files (`paper/expected_outputs.sha256`); needs matplotlib (in the `test` extra) and Pillow, and is skipped without matplotlib; the figures are compared only with the reference versions (matplotlib 3.11.2, Pillow 12.3) |
| `test_prepare_data.py` | the preparation script knows exactly the ten datasets, its split fingerprints equal those of the paper's records, the custom split lists, the numbering of colour-coded labels, the round-robin naming, and the loaders' refusals of incomplete data |
| `test_release.py` | the shipped generators' comments and docstrings and the shipped score records hold no internal notes, host names, environment fields or absolute paths |
| `test_crosscheck_autoseg.py` | optional, for the authors, see below |

Without the research code, the cross-checks are skipped: on macOS (arm64, torch 2.14.1, CPU, matplotlib 3.11.2) the
suite gives 36 passed and 53 skipped.

**Cross-check against the research code (for the authors).** The research code (AutoSeg) is not public, so this check
can only be run by the authors; its result on a GPU, the golden comparison below, is in `docs/verification/`. With
`EXEMPLAR_AUTOSEG_TREE` pointing at a checkout of the research repository, `test_crosscheck_autoseg.py` runs the
**original** functions of the research benchmark (`ridge_fit`, `fast_fine`, `zoom_unified_parallel`, ...) unchanged,
beside the extracted ones, on synthetic inputs, and requires equality bit for bit: two noise levels (λ at its bound
and inside), both branches of the one-standard-error rule, K = 4 and K = 1, IoU and centreline Dice, the refit, the
priors, the coarse view (with a stand-in model), the loaders, the split fingerprint, the metrics and the dataset
registry.

```bash
EXEMPLAR_AUTOSEG_TREE=/path/to/AutoSeg pytest
```

On the same macOS CPU environment all 89 tests pass.

### The golden comparison on a GPU (`tools/golden_compare.py`)

Like the cross-check, this comparison needs the research code, which is not public: it is for the authors, and its
result is recorded in `docs/verification/` ("What has been verified" below).

The golden comparison runs the original research benchmark and this package on the same GPU, one after the other,
each in its own process and with its own coarse-view cache, and compares them seed by seed in pipeline order, so
that the first mismatch locates a divergence:

support indices → magnification z → inputs of every support image (coarse view, fine view, priors) → column shift
→ per-image sums → λ of each weighting → blend weight a, variant and cut → weights → held-out values → test inputs
→ records (per-image scores of all four readouts, split fingerprint, magnifications, λ).

```bash
python tools/golden_compare.py --autoseg_tree /path/to/AutoSeg --datasets dsb2018,drive \
    --seeds 2 --gpu 0 --cache /path/to/golden_cache --out /path/to/golden_out
```

Run it on a CUDA machine, in one Python environment with the same torch, transformers, numpy, scipy and
scikit-image as the reference run, and with the dataset variables of [DATASETS.md](DATASETS.md). The research
benchmark runs from a copy in which two lines capture what it never writes (every image's inputs, and the per-image
sums); no arithmetic line is changed.

**PASS** requires identical choices (support, z, both λ, a, variant, cut), per-image test scores within 1e-6 for the
reported readout and the three others, and `max|w_new − w_orig| / max|w_orig|` within 1e-6. Differences in input or
sum fingerprints are reported as warnings with their size; they explain a failure but cannot make a PASS wrong. The
report goes to `<out>/report.json`; the exit code is 0 for PASS, 1 for FAIL and 2 for an error.

**What has been verified.**

- On the CPU, a dry run of the tool with synthetic inputs gives PASS for identical implementations and FAIL for a
  weight perturbation of 1e-4.
- In the research code's timing campaign on an RTX A5000, the exact refit (the eighth mask added to a fit of seven)
  was identical to the cold fit on all ten datasets (seed 0).
- **On a GPU (17 October 2026), PASS, bit for bit.** One NVIDIA A100 80GB PCIe; Python 3.12.14, torch 2.5.1+cu121
  (CUDA 12.1, cuDNN 9.0), transformers 5.13.0, numpy 2.4.6, scipy 1.18.0, scikit-image 0.26.0, both runs in the same
  environment. Reference: the research code tree that produced the paper's records, with the research benchmark at
  its later revision (which only adds options that are off by default). DSB2018, ISBI2012-EM (centreline Dice) and
  DRIVE (magnified fine view, z = 4), seeds 0 and 1, K = 8: all 192 input stacks identical; z, both λ and a identical
  on every seed (a = 1, 1, 0.7, 0.65, 0.35, 0.4); weights identical (relative difference 0); all 168 per-image test
  scores identical (largest difference 0). Report and log: `docs/verification/golden_A100_2026-10-17.json`, `.log`.
  The same seeds against the paper's records, which were computed on L40S GPUs, agree to 4 decimals in the dataset
  means (per-image differences up to 0.0015, the expected difference between GPU models).

Bit-identity holds only on the same GPU model and the same library versions: TF32 products, cuDNN kernels and
bfloat16 autocast differ between GPU generations and torch releases, and the skeleton follows scikit-image 0.26.

## 5. Run times and memory

**Time.** Exemplar's fit, exact refit and prediction on one dedicated RTX A5000 (24 GB), measured by the research
code under the protocol of [BASELINES.md](BASELINES.md) ("Timing"): seed 0's support draw, K = 8, the first five
test images. Every time includes encoding the images. This package runs the same computation, but it has not been
timed separately.

| Dataset | z (seed 0) | First test image | Fit from 8 masks | Adding the 8th mask | Per image |
|---|---|---|---|---|---|
| SpheroidJ | 1 | 966 × 1296 | 6.6 s | 2.6 s | 0.48 s |
| Decay | 4 | 2048 × 2048 | 101 s | 18.4 s | 10.0 s |
| DSB2018 | 3 | 256 × 256 | 2.6 s | 0.91 s | 0.19 s |
| MoNuSeg | 2 | 1000 × 1000 | 11.4 s | 2.1 s | 0.74 s |
| BBBC010 | 3 | 520 × 696 | 5.6 s | 1.8 s | 0.50 s |
| Bacteria | 2 | 600 × 600 | 4.6 s | 1.6 s | 0.35 s |
| DRIVE | 4 | 584 × 565 | 10.3 s | 2.5 s | 0.89 s |
| HRF | 2 | 2336 × 3504 | 72.7 s | 14.9 s | 5.97 s |
| ISBI2012-EM | 3 | 512 × 512 | 5.7 s | 1.7 s | 0.37 s |
| FISBe | 4 | 680 × 680 | 13.6 s | 2.9 s | 1.27 s |
| **median** | | | **8.5 s** | | **0.62 s** |

The magnification did not change when the eighth mask was added on any dataset, so every refit above is an
incremental update. The fine view dominates on large or strongly magnified images: its cost grows with z², and on
Decay (2048 px at z = 4) it took 75 s of the 101-s fit.

A full re-run of section 1 fits 10 seeds per dataset and K and scores every test image once per decided
magnification.

**Memory.** All ten K = 8 fits, refits and predictions above ran on the 24 GB RTX A5000. The paper's records
were produced on 46 GB L40S GPUs. Before starting a dataset, the research launcher waited for this much free GPU
memory: 12 GB at K = 1 or 4, 14 GB at K = 8, 22 GB at K = 16; for HRF 18 GB at K = 1 or 4, 22 GB at K = 8 and
28 GB at K = 16. These are thresholds the launcher used, not measured peaks.

GPU memory grows with K, because the fit keeps every support image's inputs for the held-out maps and the exact
update. `Exemplar.release_support()` frees them; a later `add` then refits from scratch, with the same result.

One hard limit: the priors' percentile normalisation calls `torch.quantile`, which refuses inputs above 2²⁴
elements, so images above 16,777,216 pixels (for example 4096 × 4096) fail. HRF, the largest panel image at
8.2 million pixels, is within it.
