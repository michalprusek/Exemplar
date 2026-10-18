# Exemplar

**Few-shot biomedical segmentation that refits exactly in seconds.**

Exemplar segments a new biomedical image dataset from a few annotated images. It reads each image with a frozen
DINOv3 backbone, in a coarse view of the whole image and in a magnified fine view, and computes 35 classical filter
responses at native resolution. A linear readout of these inputs is the only part that is fitted. It is solved in
closed form over every pixel of the annotated images, and its regularisation and its pixel weighting are chosen by
leaving out one annotated image at a time. When the annotator adds another mask, the readout is updated exactly:
the result equals a fit from scratch on all masks.

> Exemplar solves a segmentation readout in closed form and updates it exactly with each new mask in seconds, as
> accurately as networks trained for minutes.

On ten biomedical datasets, from cells to vessels and filaments, Exemplar with eight masks has a mean score of
0.822, against 0.804 for nnU-Net trained from scratch on the same masks. A fine-tuned Cellpose-SAM, the strongest
baseline, is as accurate on average (0.823), but each of its fits takes a median of 25 minutes, against 8.5 seconds
for Exemplar.

This repository holds the method as reported in the ISBI 2027 paper (v4 of the research code), the benchmark
protocol of the paper, and the documentation needed to reproduce it.

- [docs/METHOD.md](docs/METHOD.md): the method, step by step, with every constant
- [docs/DATASETS.md](docs/DATASETS.md): the ten datasets, their sources, licences and splits, and how to get them
- [docs/BASELINES.md](docs/BASELINES.md): every comparison method and how it was run and timed
- [docs/REPRODUCE.md](docs/REPRODUCE.md): how to re-run the experiments and regenerate the paper

## Installation

Requirements: Python ≥ 3.10, PyTorch ≥ 2.5 and transformers ≥ 4.57 (for DINOv3; see `pyproject.toml`). A CUDA GPU is
needed for practical use. The code should also run on the CPU, slowly; the CPU tests run it on synthetic inputs
only. Other devices are untested.

```bash
git clone https://github.com/michalprusek/Exemplar.git
cd Exemplar
pip install -e .                 # the method
pip install -e ".[benchmark]"    # + requests, to download the benchmark datasets
pip install -e ".[test]"         # + pytest
```

Install the PyTorch build for your CUDA version first if pip would otherwise pick the wrong one
(<https://pytorch.org/get-started/locally/>).

The verified environment (docs/REPRODUCE.md, section 4): Python 3.12, torch 2.5.1 with CUDA 12.1, transformers 5.13,
numpy 2.4, scipy 1.18 and scikit-image 0.26. Bit-identical reproduction needs the same versions and the same GPU
model (see [REPRODUCE.md](docs/REPRODUCE.md)). Other versions or GPUs can change the last digits of the scores
and, rarely, a discrete choice (the ridge weight, the blend weight or the magnification).

### The DINOv3 checkpoint is gated

Exemplar uses `facebook/dinov3-vitl16-pretrain-lvd1689m` from Hugging Face, which requires accepting Meta's terms:

1. Open <https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m>, log in and request access.
2. Once access is granted, authenticate on the machine that runs Exemplar: `hf auth login` (`huggingface-cli
   login` in older versions of `huggingface_hub`), or set the environment variable `HF_TOKEN` to an access token.
3. The weights (303 million parameters) are downloaded on first use into the Hugging Face cache. Set `HF_HOME` to
   put the cache on a disk with space.

Without access, loading the model fails with an access error from Hugging Face.

## Quickstart

```python
import numpy as np
from exemplar import Exemplar

# images: a list of numpy arrays, H x W (grey) or H x W x 3 (RGB), any dtype
# masks:  a list of numpy arrays of the same H x W, foreground where the value is above zero
model = Exemplar(device="cuda", cache_dir="cache", metric="iou")   # "cldice" for vessels, membranes, filaments

model.fit(images[:7], masks[:7], rng=np.random.default_rng(0))     # closed-form fit on 7 annotated images
model.add(images[7], masks[7])     # exact update: the same readout as a fit from scratch on all 8 masks

fg = model.predict(new_image)      # bool array H x W, the foreground mask
s = model.score(new_image)         # float32 score map; predict is s > 0.5
print(model.info)                  # K, magnification z, blend weight a, both ridge weights, held-out score, ...
                                   # and the counters single_fallbacks, single_rejected_evals when non-zero
```

What the inputs must look like:

- **Images** are numpy arrays, grey (H × W) or colour (H × W × 3). Every image is min-max normalised by itself, so
  8-bit, 16-bit and float images all work. Support images may differ in size. An alpha channel is dropped by the
  backbone's two views but averaged into the grey image of the priors, as in the paper's code, so pass grey or RGB.
- **Masks** are complete: every pixel above zero is foreground and every other pixel is background, so an object
  left unannotated counts as background. Instance label maps work as they are (all labels above zero are
  foreground). A mask should have its image's height and width; otherwise it is resized by nearest neighbour.
  Every support mask should contain some foreground.
- **K** can be 1. With a single mask, its four quadrants serve as the held-out images.
- **`metric`** is the held-out metric that chooses the blend of the two readouts: `"iou"` (foreground
  intersection-over-union) for objects, `"cldice"` (centreline Dice) for thin structures.
- **`rng`** (a numpy `Generator` or a seed) is drawn once, for the 4096 pixels of the first support image that fix
  a numerical column shift. The same generator gives the same readout.
- **`cache_dir`** caches the coarse view of every image on disk (in `<cache_dir>_bf16` under the default
  bfloat16 setting); `None` disables it.

More of the API:

- `model.readout` is the fitted readout: the blend weight `a`, the ridge weight `lam` of the plain readout (and
  `lam_bal` of the balanced one when the readout is a blend, a < 1), the cut (always 0.5) and the weight vector `w`
  on `[coarse 2048 | fine 2048 | priors 35 | constant]`. `model.info` gives both ridge weights in every case.
- `model.release_support()` frees the support images' inputs held on the GPU for the exact update; a later `add`
  then refits from scratch, with the same result.
- `FrozenInputs` computes the frozen inputs of an image. To share one backbone between several models, create
  `inputs = FrozenInputs("cuda", cache_dir="cache")` once and pass `Exemplar(inputs=inputs)`.
- `bf16=False` runs the backbone in float32. That is a different numerical configuration from the paper's.

## Hardware notes

- **GPU memory.** With eight masks, every dataset of the paper fits, updates and predicts on a 24 GB RTX A5000; the
  paper's score records were produced on 46 GB L40S GPUs. Memory grows with the number of masks and the
  magnification, because the fit keeps every support image's inputs for the exact update. Peak memory per dataset
  has not been measured.
- **Time.** On the RTX A5000, a fit from eight masks takes 2.6 to 100 seconds depending on image size and
  magnification (median 8.5 s), adding the eighth mask 0.9 to 18 seconds, and segmenting an image 0.2 to 10
  seconds (median 0.62 s), encoding included. Per-dataset times are in [REPRODUCE.md](docs/REPRODUCE.md).
- **Largest image.** The priors' percentile normalisation uses `torch.quantile`, which refuses inputs above 2²⁴
  elements, so images above 16,777,216 pixels (for example 4096 × 4096) fail. Larger images would have to be
  downscaled or tiled, which the paper did not evaluate.
- **CPU threads.** `OMP_NUM_THREADS` sets the size of the thread pool for the magnification rule and the scoring
  (default 8). It does not change any result.

## Repository layout

```
exemplar/     the method
  encoder.py    the frozen DINOv3 backbone, the coarse view and its disk cache
  fineview.py   the magnified fine view and the rule that sets its magnification from the masks
  priors.py     the 35 classical priors at native resolution
  readout.py    the closed-form readout: Gram sums on the patch grids, leave-one-image-out, Newton's method for
                the ridge weight, the blend of two readouts, the exact update, the score map
  model.py      the public API: Exemplar.fit / add / predict / score
benchmark/    the paper's protocol
  registry.py   the ten datasets, their pools and metrics, and the split
  datasets.py   the data loaders
  metrics.py    foreground IoU and centreline Dice
  run.py        the benchmark runner: ten seeds, K support masks per seed, the full test split scored
  prepare_data.py  downloads the ten datasets, lays them out and checks the splits against the paper's
tools/        maintainer tools that need the research code (not public)
  golden_compare.py  the original research benchmark against this package on one GPU
  sync_paper.py      copies the paper's generators and score records from the research code and checks them
tests/        CPU tests on synthetic inputs (no GPU, no weights) and the regeneration of the paper's files
docs/         METHOD.md, DATASETS.md, BASELINES.md, REPRODUCE.md; verification/: the golden comparison on a GPU
results/      the score records behind every number of the paper (see results/MANIFEST.md)
paper/        the table and figure generators and paper/make_paper.sh
  c119/         the timing table (Table 1's last row) from the timing records
```

## Running the tests

```bash
pip install -e ".[test]"
pytest
```

The tests run on the CPU in seconds and need neither a GPU nor the DINOv3 weights. They check the Gram sums on the
patch grids, the exact leave-one-image-out errors, the exact update, the blend choice, the priors, the device
skeleton and the ablation options of Table 2. `tests/test_paper.py` regenerates the paper's tables and numbers from
the shipped score records with `paper/make_paper.sh` and compares them with the paper's files; it needs matplotlib
(in the `test` extra) and is skipped without it.

An optional cross-check runs the original research code beside this package and requires equality bit for bit. It
is for the authors: the research code (AutoSeg) is not public, and without it these tests are skipped. The result of
the golden comparison on a GPU is in [docs/verification/](docs/verification/).

```bash
EXEMPLAR_AUTOSEG_TREE=/path/to/AutoSeg pytest     # authors only: needs the research code
```

See [REPRODUCE.md](docs/REPRODUCE.md) for what each test covers and for the golden comparison on a GPU.

## Benchmark data

The ten datasets of the paper are public but not redistributed here. One script downloads each from its
source and lays it out as the benchmark reads it, and a second call checks that the splits are the paper's:

```bash
pip install -e ".[benchmark]"            # requests and pyarrow (MoNuSeg is distributed as parquet files)
export ASG_DATA_ROOT=/path/with/space    # default: ./data (about 1.2 GB of downloads, 1.6 GB of data)
python -m benchmark.prepare_data all
python -m benchmark.prepare_data --check # compares each split's fingerprint with the paper's records
```

[docs/DATASETS.md](docs/DATASETS.md) gives every source, its licence and how each split was made.

## Reproducing the paper

[docs/REPRODUCE.md](docs/REPRODUCE.md) describes how to

1. re-run Exemplar on the ten datasets at K = 1, 4, 8 and 16 with ten seeds (Table 1, Figure 2) and dump the
   predictions of Figure 3 (`python -m benchmark.run ...`);
2. re-run the ablations of Table 2;
3. regenerate every table, figure and number of the paper from the shipped score records with
   `paper/make_paper.sh`, without a GPU;
4. run the tests and the golden comparison against the research code.

The mean scores of Exemplar in the paper's records (mean over test images and ten seeds, then over datasets):

| K | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| all ten datasets | 0.774 | 0.808 | 0.822 | 0.828 |

## Citation

If you use Exemplar, please cite the paper:

```bibtex
@inproceedings{prusek2027exemplar,
  title     = {Exemplar: Few-Shot Biomedical Segmentation That Refits Exactly in Seconds},
  author    = {Pr{\r u}{\v s}ek, Michal and Novoz{\'a}msk{\'y}, Adam and {\v S}roubek, Filip},
  booktitle = {IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2027},
  note      = {Submitted}
}

@article{prusek2026exemplar_arxiv,
  title   = {Exemplar: Classical Priors Complement Frozen Features for Few-Shot Microscopy Segmentation at
             Native Resolution},
  author  = {Pr{\r u}{\v s}ek, Michal and Novoz{\'a}msk{\'y}, Adam and {\v S}roubek, Filip},
  journal = {arXiv preprint arXiv:2609.03080},
  year    = {2026}
}
```

The arXiv preprint (version 1, September 2026) describes an earlier version of the method, in which a small head
was trained by gradient descent; this repository implements the closed-form readout of the ISBI 2027 submission.

The classical prior bank comes from our earlier work, HyperBank (arXiv:2607.10684).

## License

The code is released under the MIT License (see [LICENSE](LICENSE)). The licence covers the code only. Every
benchmark dataset keeps its own licence: Bacteria and MoNuSeg are for non-commercial use only (see
[docs/DATASETS.md](docs/DATASETS.md)). No dataset is redistributed here. CTC-U373, which earlier versions of the
paper evaluated, is not included, because its conditions require the organisers' permission for public use outside
the Cell Tracking Challenge. The DINOv3 weights are distributed by Meta under their own terms, which you accept when
you request access.

## Acknowledgements

This work was supported by the Czech Science Foundation, grant GA25-15933S.
