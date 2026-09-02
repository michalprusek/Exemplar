# Exemplar

**Classical priors and frozen features are complementary: few-shot biomedical image segmentation
from a handful of masks.**

Reference implementation for the ISBI 2027 submission *"Exemplar: Classical Priors Complement Frozen Features for Few-Shot Microscopy Segmentation at Native Resolution"* (Prusek, Novozamsky, Sroubek).

A biologist annotates about eight example objects. Exemplar segments the rest of the dataset. It
keeps a frozen DINOv3 backbone, adds a fixed bank of 35 classical native-resolution filter
responses, and trains only a light head that **derives its loss weighting and its input colour
channel from the support masks alone** — one pipeline, no per-dataset tuning, across eleven
biomedical imaging datasets spanning blobs, cells, worms, bacteria, vessels, membranes and thin
filaments.

The paper's claim is about the two feature families rather than about either one: **in the
few-mask, native-resolution regime, classical priors and frozen self-supervised features are
complementary.** Adding the prior bank to the frozen features raises the seven-dataset ablation mean
from 0.700 to 0.801 — an order of magnitude more than the 0.009 the support-derived rules
contribute. Neither ingredient is ours: the bank comes from our own earlier HyperBank paper, and
pairing hand-designed with frozen deep features is established practice. The measurement, and the
regime in which it holds, is what this repository backs.

The method produces a **semantic foreground map** (a per-pixel foreground probability), not
separated instances. Separating touching objects is out of scope for this paper; a clean foreground
is the quantity behind area- and coverage-based readouts and the seed for standard instance
post-processing.

## Results at eight support masks

Foreground IoU, or centreline Dice where marked, mean ± sd over ten seeds. Full table with all
baselines: `scripts/make_semantic_tables.py` (see *Reproducing the paper* below).

| Dataset | Morphology | Exemplar | Best few-shot baseline |
|---|---|---|---|
| SpheroidJ | spheroids | **0.917** ± 0.027 | 0.895 (SegGPT) |
| Decay | decaying spheroids | **0.795** ± 0.008 | 0.588 (INSID3) |
| DSB2018 | nuclei | 0.848 ± 0.008 | 0.804 (INSID3) |
| MoNuSeg | H&E nuclei | 0.637 ± 0.018 | 0.443 (INSID3) |
| CTC-U373 | phase-contrast cells | 0.800 ± 0.008 | 0.758 (SegGPT) |
| BBBC010 | *C. elegans* | **0.610** ± 0.005 | 0.419 (SegGPT) |
| Bacteria | dense rods | **0.917** ± 0.006 | 0.730 (INSID3) |
| DRIVE † | retinal vessels | **0.762** ± 0.006 | 0.479 (Tyche) |
| HRF † | retinal vessels | **0.728** ± 0.011 | 0.235 (SegGPT) |
| ISBI2012-EM † | EM membranes | **0.920** ± 0.002 | 0.853 (Tyche) |
| FISBE † | thin filaments | **0.742** ± 0.014 | 0.598 (INSID3) |

† centreline Dice. Bold = best across *all* methods including trained specialists.

Trained specialists (Cellpose, StarDist, micro-SAM) use no support masks but were trained on
thousands of objects. They lead on the three standard-cell datasets they were built for; on vessels,
membranes and filaments they drop to ≈0.30 or below, where Exemplar holds 0.74–0.92. That is breadth
of applicability, not a claim of being more accurate than a specialist in its own domain.

**nnU-Net trained from scratch on the same eight masks leads on nine of these eleven datasets**,
by 0.015 on the panel mean (0.8035 against 0.7889), and takes 14–114× longer to fit. The ordering
separates along the metric, not the dataset: the two are level on the seven overlap-scored datasets
(0.7893 against 0.7884) and nnU-Net leads on the four scored by centreline Dice (0.8298 against
0.7881). At a single mask the ranking reverses overall — Exemplar 0.7105 against 0.6821 — because it
leads by 0.059 on overlap while trailing by 0.025 on centreline. The two curves cross between one
and four masks.

Only Exemplar and that nnU-Net stay above 0.6 on every dataset (worst case 0.610 and 0.645). The
next-best fitted method, a random forest over the same prior bank, drops to 0.369; every method that
consumes the masks in a forward pass falls below 0.23 somewhere (0.092–0.224), and every
off-the-shelf specialist below 0.02 (0.006–0.012).

## Install

```bash
git clone https://github.com/michalprusek/Exemplar.git
cd Exemplar
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

DINOv3 is a **gated** HuggingFace model: request access at
[facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
and `huggingface-cli login` before the first run. A CUDA GPU is required to *run* the method
(≈16 GB is comfortable; the largest images, HRF at 3504 px, want more). **Reproducing the paper's
tables and figures needs no GPU** — the score records are in this repository.

Baseline reproduction needs additional, mutually conflicting packages; see
`requirements-baselines.txt` and install each baseline family in its own environment, as the paper's
fairness protocol requires.

## Use it on your own data

Annotate a few images, point the script at the rest:

```bash
python scripts/predict.py --support supp/ --images raw/ --out masks/
```

`supp/` holds example pairs — an image and its mask sharing a stem, the mask carrying a `_mask`
suffix:

```
supp/cells_01.png   supp/cells_01_mask.png
supp/cells_02.png   supp/cells_02_mask.png
```

Masks may be binary or per-instance label maps; either is read as foreground versus background.
Around eight examples is the operating point the paper reports — one already works, and past sixteen
the curve is flat on most datasets. Every image in `raw/` is written to `masks/` as a PNG, and
`--prob` additionally writes the float probability map as a TIFF for your own thresholding.

Nothing needs configuring. The run prints what the method decided for itself:

```
fitting on 3 example(s) ...
    [color_adaptive] dataset is monochrome → source=gray
    [affinity] binary/semantic support (median max-label=1) → instance decoder inactive here
    [corr_prior] prototypes built (fg 1241 / bg 4051 patches); fg·bg cos=0.975
  [1/2] 03.png -> 03_mask.png (23.0% foreground)
```

Features are cached under `<out>/.feature_cache`, so a second pass over the same images is nearly
free. Do not point two concurrent runs at one cache directory.

## Run the benchmark

The reported configuration is `head_fusion_best_nobank_noloss_nocolor_flat_es_ep500_wd4_do5_mix_noaux`,
and its score records are the `results/final10/lean_k*` directories. Four levers this repository
contains are deliberately **off** in it, each having been implemented, measured and rejected: the
competitive gate and the FiLM modulation (`_cgate`, `_film`), the support-derived loss constructor
(`_noloss` switches it off), the colour/stain channel selection (`_nocolor`), and the auxiliary
outline head (`_noaux`). A method name containing `_cgate` or `_film`, or lacking `_noloss`,
`_nocolor` or `_noaux`, is **not** the method the paper reports.

```bash
python scripts/sota_final.py run \
  --method head_fusion_best_nobank_noloss_nocolor_flat_es_ep500_wd4_do5_mix_noaux \
  --datasets monuseg --support 8 --pool 20 --test 10000 --seeds 10 \
  --res 672 --metric_override fg_iou \
  --cache /path/to/feature_cache --score_dir results/my_run
```

Note `--metric_override fg_iou` for the seven overlap-scored datasets; the four centreline-scored
ones (DRIVE, HRF, ISBI2012-EM, FISBE) take their own metric, and ISBI2012-EM must be run at
`--pool 16` because its pool size shifts its test slice.

The method name is composable — `head_fusion_best[_cgate][_film][_nobank][_noloss][_nocolor][_noaux]
[_flat|_lean][_h<N>][_es][_ep<N>][_wd<N>][_do<N>][_mix][_nocls][_bankonly][_coarseonly][_fineonly]`,
and an unrecognised token is rejected rather than ignored — which is how the ablation arms are built.
The five arms behind Table 2 and the component deltas are `lean_nocls_k8` (bank zeroed),
`lean_bankonly_k8` (features zeroed), `lean_coarseonly_k8`, `lean_fineonly_k8` and
`lean_nomix_k8`; each differs from the reported name by exactly one token, which
`scripts/make_ablation_table.py` checks before it will print a delta.

> **Do not let two concurrent runs share a writable `--cache`.** Pre-build once, or give each
> process its own cache directory. A shared writable cache produced irreproducible numbers during
> development and cost a full re-run.

## The method, in full

The paper's Implementation section states that the remaining implementation-level constants are listed in the released code. They are, and [`METHOD.md`](METHOD.md) collects them in one place: the four support-mask shape descriptors, every loss term with its weight and ramp break-points, the channel-selection margin and contrast reference, the 35 prior-bank channels family by family, the head's parameter budget module by module, and the full training paradigm including the plateau rule. It also names the levers this repository contains that the reported configuration does **not** use, so there is no ambiguity about which code path produced the numbers.

## Reproducing the paper

Every number in the paper is generated from the score records in `results/final10/`, never typed by
hand. The generators are fail-loud: they refuse to write a table with a missing cell, with unequal
seed counts, or with more than one metric-matching record per cell, and the figure generator refuses
to overwrite its output if the data root produced no points.

```bash
OUT=/tmp/out && mkdir -p $OUT

# Table 1 — main results, 11 datasets, ours vs few-shot baselines and trained specialists
ASG_SEM_TREE=results/final10 ASG_SEM_OUT=$OUT python scripts/make_semantic_tables.py

# Table 2 — component ablation (prior bank, then the self-configuration rules)
ASG_SEM_TREE=results/final10 ASG_SEM_OUT=$OUT python scripts/make_ablation_table.py

# Paired Wilcoxon + Holm: the 55 forward-pass comparisons and the 11 against nnU-Net
ASG_SEM_TREE=results/final10 ASG_RESULTS_ROOT=results ASG_SEM_OUT=$OUT python scripts/make_stats.py

# The frozen-backbone probe and the ilastik-RF control
ASG_SEM_TREE=results ASG_RESULTS_ROOT=results ASG_SEM_OUT=$OUT python scripts/make_probe_numbers.py

# Fit and inference timings, all from results/timing_a5000.json (one idle RTX A5000)
ASG_TIMING=results/timing_a5000.json ASG_SEM_OUT=$OUT python scripts/make_timing_numbers.py

# Figure 2 — K-scaling curve, K = 1, 4, 8, 16   (note: ROOT is the PARENT of final10)
ASG_RESULTS_ROOT=results ASG_KSCALE_OUT=$OUT/kscale.pdf python scripts/make_final_kscale.py
```

Each writes a `.tex`/`.pdf` that is **byte-identical to the file the manuscript compiles** (six `.tex` files: `make_ablation_table.py` writes two) —
verified for all five `.tex` artefacts from a fresh clone of this repository. If your
output differs, that is a real discrepancy worth reporting, not a formatting artefact.

## Repository layout

| Path | What |
|---|---|
| `active_segmenter/segment/head_fusion_backend.py` | The method: scale fusion, classical prior bank, and every closed-form self-configuration rule (adaptive loss, colour selection, scale selection) |
| `active_segmenter/segment/head_fusion.py` | The trainable head: shared stem, edge-guided upsampling, 1×1 fusion and classifier. The competitive gate and FiLM modulation live here too but are **not** part of the reported configuration — they were measured and rejected (Table 2) |
| `active_segmenter/segment/upsamplers.py` | The edge-guided upsampler (`GuidedUp`) and the guided-filter bank upsampling |
| `active_segmenter/encoder/dinov3.py` | Frozen DINOv3 encoder, feature super-resolution, caching |
| `active_segmenter/eval/` | Dataset registry, metrics, scoring, score-record format |
| `active_segmenter/acquire/` | Active-learning acquisition functions (not used in the paper; groundwork for the tool) |
| `scripts/predict.py` | **Segment your own images from a few masks** (the use path) |
| `scripts/nnunet_bench.py` | nnU-Net trained on the same K support masks, as an annotation-efficiency rival |
| `scripts/sota_final.py` | Benchmark harness: multi-draw fixed-pool protocol, paired statistics |
| `scripts/run_campaign.py`, `run_ablation.py` | The campaign and ablation launchers |
| `scripts/make_*.py` | Paper table and figure generators |
| `scripts/prep_public_datasets.py` | Dataset download and preparation |
| [`METHOD.md`](METHOD.md) | **Full specification of the reported configuration**: every component, every constant, the training paradigm, and which levers in this repository are NOT part of it |
| `results/final10/` | Score records for every cell of every reported table |
| `results/rescontrol/` | Resolution control: the same method at the baselines' 448-pixel input |
| `tests/` | 287 tests, including ones that pin the ablation arms as genuinely distinct configurations |

## Datasets

None are redistributed here. See [`DATASETS.md`](DATASETS.md) for sources and licenses. Point
`PANEL_DL_ROOT` at your download directory; `scripts/prep_public_datasets.py` fetches and prepares
DRIVE, ISBI2012-EM, MoNuSeg, and CTC-U373 automatically.

## Tests

```bash
python -m pytest tests -q     # 282 passed, 3 skipped
```

The three skips need `pydensecrf`, an optional dependency used only to strengthen the INSID3
baseline. The code deliberately **refuses to substitute a guided filter** when it is missing, since
a silent substitution would undersell that baseline.

## Protocol notes

The benchmark uses a multi-draw fixed-pool design: the support pool and test split are loaded once
per dataset, then K support masks are subsampled per seed, so the test set never shifts with K.
Scores are collapsed to one value per test image before testing, so the unit of analysis is the
image rather than the seed-image pair; comparisons use paired Wilcoxon signed-rank tests with Holm
correction. On the smallest datasets, prefer effect sizes reproduced across seeds over single
p-values.

## Citation

```bibtex
@inproceedings{prusek2027exemplar,
  title     = {Exemplar: Classical Priors Complement Frozen Features for Few-Shot
              Microscopy Segmentation at Native Resolution},
  author    = {Pr{\r u}{\v s}ek, Michal and Novoz{\'a}msk{\'y}, Adam and {\v S}roubek, Filip},
  booktitle = {IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2027}
}
```

The classical prior bank that Exemplar freezes and fuses was introduced in our earlier work
(HyperBank, arXiv:2607.10684); this paper's contribution is the in-context pipeline that configures
itself from the support masks, not the bank.

## License

MIT for the code — see [`LICENSE`](LICENSE). Datasets keep their own licenses; the Bacteria
(Omnipose/BPCIS) data is CC-BY-NC-3.0 and may not be used commercially.
