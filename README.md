# Exemplar

**Self-configuring in-context segmentation of microscopy images from a handful of labels.**

Reference implementation for the ISBI 2027 submission *"Exemplar: Self-Configuring In-Context
Segmentation of Microscopy Images from a Handful of Labels"* (Prusek, Novozamsky, Sroubek).

A biologist annotates about eight example objects. Exemplar segments the rest of the dataset. It
keeps a frozen DINOv3 backbone, trains only a light head, and **configures its own loss, its input
colour channel, and its feature modulation from the support masks alone** — one pipeline, no
per-dataset tuning, across eleven microscopy datasets spanning blobs, cells, worms, bacteria,
vessels, membranes, and thin filaments.

The method produces a **semantic foreground map** (a per-pixel foreground probability), not
separated instances. Separating touching objects is out of scope for this paper; a clean foreground
is the quantity behind area- and coverage-based readouts and the seed for standard instance
post-processing.

## Results at eight support masks

Foreground IoU, or centreline Dice where marked, mean ± sd over ten seeds. Full table with all
baselines: `scripts/make_semantic_tables.py` (see *Reproducing the paper* below).

| Dataset | Morphology | Exemplar | Best few-shot baseline |
|---|---|---|---|
| SpheroidJ | spheroids | **0.902** ± 0.027 | 0.895 (SegGPT) |
| Decay | decaying spheroids | **0.784** ± 0.007 | 0.588 (INSID3) |
| DSB2018 | nuclei | 0.846 ± 0.009 | 0.804 (INSID3) |
| MoNuSeg | H&E nuclei | 0.628 ± 0.013 | 0.443 (INSID3) |
| CTC-U373 | phase-contrast cells | 0.739 ± 0.010 | 0.758 (SegGPT) |
| BBBC010 | *C. elegans* | **0.571** ± 0.009 | 0.419 (SegGPT) |
| Bacteria | dense rods | **0.900** ± 0.006 | 0.730 (INSID3) |
| DRIVE † | retinal vessels | **0.690** ± 0.009 | 0.400 (SegGPT) |
| HRF † | retinal vessels | **0.680** ± 0.008 | 0.235 (SegGPT) |
| ISBI2012-EM † | EM membranes | **0.876** ± 0.008 | 0.765 (UniverSeg) |
| FISBE † | thin filaments | **0.669** ± 0.015 | 0.598 (INSID3) |

† centreline Dice. Bold = best across *all* methods including trained specialists.

Trained specialists (Cellpose, StarDist, micro-SAM) use no support masks but were trained on
thousands of objects. They lead on the three standard-cell datasets they were built for; away from
those they drop to ≈0.30 or below on vessels, membranes, and filaments, where Exemplar holds
0.67–0.88.

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

## Run the method

The reported configuration is `head_fusion_best_cgate_film_nobank`.

```bash
python scripts/sota_final.py run \
  --method head_fusion_best_cgate_film_nobank \
  --datasets monuseg --support 8 --pool 20 --test 10000 --seeds 10 \
  --res 672 --metric_override fg_iou \
  --cache /path/to/feature_cache --score_dir results/my_run
```

The method name is composable — `head_fusion_best[_cgate][_film][_nobank]` — which is how the
ablation arms in Table 2 are built (`scripts/run_ablation.py`).

> **Do not let two concurrent runs share a writable `--cache`.** Pre-build once, or give each
> process its own cache directory. A shared writable cache produced irreproducible numbers during
> development and cost a full re-run.

## Reproducing the paper

Every number in the paper is generated from the score records in `results/final10/`, never typed by
hand. The generators are fail-loud: they refuse to write a table with a missing cell, with unequal
seed counts, or with more than one metric-matching record per cell, and the figure generator refuses
to overwrite its output if the data root produced no points.

```bash
# Table 1 — main results, 11 datasets, ours vs few-shot baselines and trained specialists
ASG_SEM_TREE=results/final10 ASG_SEM_OUT=/tmp/out python scripts/make_semantic_tables.py

# Table 2 — component ablation (architecture ladder + self-configuration block)
ASG_SEM_TREE=results/final10 ASG_SEM_OUT=/tmp/out python scripts/make_ablation_table.py

# Figure 2 — K-scaling curve, K = 1, 4, 8, 16   (note: ROOT is the PARENT of final10)
ASG_RESULTS_ROOT=results ASG_KSCALE_OUT=/tmp/out/kscale.pdf python scripts/make_final_kscale.py
```

Each writes a `.tex`/`.pdf` that is **byte-identical to the file the manuscript compiles**. If your
output differs, that is a real discrepancy worth reporting, not a formatting artefact.

## Repository layout

| Path | What |
|---|---|
| `active_segmenter/segment/head_fusion_backend.py` | The method: scale fusion, classical prior bank, and every closed-form self-configuration rule (adaptive loss, colour selection, scale selection) |
| `active_segmenter/segment/head_fusion.py` | The trainable head: competitive gate, FiLM modulation, 1×1 classifier |
| `active_segmenter/segment/upsamplers.py` | The edge-guided upsampler (`GuidedUp`) and the guided-filter bank upsampling |
| `active_segmenter/encoder/dinov3.py` | Frozen DINOv3 encoder, feature super-resolution, caching |
| `active_segmenter/eval/` | Dataset registry, metrics, scoring, score-record format |
| `active_segmenter/acquire/` | Active-learning acquisition functions (not used in the paper; groundwork for the tool) |
| `scripts/sota_final.py` | Benchmark harness: multi-draw fixed-pool protocol, paired statistics |
| `scripts/run_campaign.py`, `run_ablation.py` | The campaign and ablation launchers |
| `scripts/make_*.py` | Paper table and figure generators |
| `scripts/prep_public_datasets.py` | Dataset download and preparation |
| `results/final10/` | Score records for every cell of every reported table (398 files) |
| `tests/` | 285 tests, including ones that pin the ablation arms as genuinely distinct configurations |

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
  title     = {Exemplar: Self-Configuring In-Context Segmentation of Microscopy
               Images from a Handful of Labels},
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
