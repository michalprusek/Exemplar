# Datasets

The paper evaluates Exemplar on ten biomedical imaging datasets. **No dataset is redistributed in this
repository.** Each one is public and keeps its own licence. `benchmark/prepare_data.py` downloads every dataset
from its public source and lays it out as the benchmark reads it; `--check` then confirms that you have the
paper's splits.

The panel covers two kinds of structure on purpose. One pipeline, with no per-dataset tuning, should hold up on
compact objects (spheroids, nuclei, worms, bacteria) and on thin structures (vessels, membranes, filaments)
alike. The two kinds are scored by different metrics, because overlap is the wrong measure for structures a few
pixels wide.

## The ten datasets

The **key** is the name the code uses (`benchmark/registry.py`). Decay is called `rozpad` in the code.

| Key | Name | What it shows | Metric | Pool | Test images |
|---|---|---|---|---|---|
| `spheroidj` | SpheroidJ | spheroids, one large object per image | foreground IoU | 20 | 24 |
| `rozpad` | Decay | decaying spheroids that break into many small crumbs | foreground IoU | 20 | 24 |
| `dsb2018` | DSB2018 | fluorescence nuclei | foreground IoU | 20 | 50 |
| `monuseg` | MoNuSeg | nuclei in H&E-stained histology | foreground IoU | 20 | 14 |
| `bbbc010` | BBBC010 | live and dead *C. elegans*, overlapping elongated bodies | foreground IoU | 20 | 60 |
| `bacteria` | Bacteria | densely packed bacteria of 17 species (Omnipose `bact_phase`) | foreground IoU | 20 | 148 |
| `drive` | DRIVE | retinal vessels in colour fundus images | centreline Dice | 20 | 20 |
| `hrf` | HRF | retinal vessels in high-resolution fundus images | centreline Dice | 20 | 25 |
| `isbi2012em` | ISBI2012-EM | neuronal membranes in electron microscopy | centreline Dice | **16** | 14 |
| `fisbe` | FISBe | long thin neurons in fluorescence light microscopy | centreline Dice | **16** | 14 |

*Pool* is the set of annotated images from which the support masks are drawn. *Test images* are the full test
split (14 to 148 images per dataset); every method is scored on all of them.

The metric of a dataset is fixed in the code: `benchmark/registry.py` lists `CLDICE = {drive, hrf, isbi2012em,
fisbe}`, and every other dataset is scored by foreground IoU. Both metrics are in `benchmark/metrics.py`:

- **Foreground IoU** (`foreground_iou`): intersection over union of the predicted mask and the annotated
  foreground (label > 0). Two empty masks score 1.
- **Centreline Dice** (`cldice`, Shit et al., CVPR 2021): the harmonic mean of the share of the predicted
  skeleton that lies inside the annotation and the share of the annotation's skeleton that lies inside the
  prediction. Skeletons come from `skimage.morphology.skeletonize`. Both masks empty scores 1; one empty scores 0.
  Centreline Dice rewards an unbroken centreline and is less sensitive than overlap to errors in the width of a
  structure a few pixels wide.

Every dataset is scored as a binary foreground. Where the annotation holds one label per object (DSB2018,
MoNuSeg, BBBC010, Bacteria, FISBe), all labels above zero are foreground.

## Getting the data

```bash
pip install -e ".[benchmark]"            # requests, and pyarrow for MoNuSeg (parquet files)
export ASG_DATA_ROOT=/path/with/space    # default: ./data
python -m benchmark.prepare_data all     # or name datasets: python -m benchmark.prepare_data drive hrf
python -m benchmark.prepare_data --check
```

The script needs no login: every source is a direct download. It fetches about 1.2 GB into
`$ASG_DATA_ROOT/_downloads`, writes about 1.6 GB of prepared data, and keeps the downloads so that a second run
does not fetch them again; delete `_downloads` once the check has passed. A dataset whose folder exists already is
left alone; `--force` rebuilds it. Each dataset is written aside and moved into place when complete, so an
interrupted run never leaves half a dataset. Where the host allows it, the source is pinned to the revision the
paper's data came from (a commit of a GitHub repository or of a Hugging Face dataset).

`--check` loads every split exactly as the benchmark does and prints, per dataset, the number of pool and test
images and two fingerprints (below). It exits with status 1 if any dataset is missing or differs. On data prepared
by the script it prints `ok` for all ten datasets:

```
dataset     pool test  test fingerprint  pool fingerprint  result
spheroidj     20   24  0221b7005adbfcae  40714537854d76ff  ok
rozpad        20   24  8017232bf8a5d616  caff16786c6721a6  ok
...
```

## Sources, citations and licences

The licence column quotes what the source itself states, read on 17 October 2026 on the page given. **Check each
source's own terms before you use the data**, especially for any commercial purpose.

| Dataset | Where the paper's copy comes from | Cite | Licence, as the source states it |
|---|---|---|---|
| SpheroidJ | Deep-Tumour-Spheroid repository, <https://github.com/WaterKnight1998/Deep-Tumour-Spheroid> (`datasets/Images/test`, `datasets/Labels/test`), the deep-learning part of SpheroidJ (<https://github.com/joheras/SpheroidJ>) | Lacalle et al., *Comput. Methods Programs Biomed.* 200:105837, 2021 | No separate licence for the data is stated. Both repositories carry the GNU GPL v3.0 in their root `LICENSE`. |
| Decay | HyperBank repository, <https://github.com/michalprusek/hyperbank> (`datasets/decay_dataset.zip.part-aa` and `part-ab`) | Průšek et al., HyperBank, ICIP Workshops 2026, arXiv:2607.10684 | CC BY 4.0 (the `LICENSE` file inside the archive and the repository's README) |
| DSB2018 | StarDist's release of a subset of the 2018 Data Science Bowl (`stage1_train`), <https://github.com/stardist/stardist/releases/tag/0.1.0> (`dsb2018.zip`) | Caicedo et al., *Nat. Methods* 16:1247, 2019; Schmidt et al., MICCAI 2018 (StarDist) | The StarDist release states no licence. The full set, BBBC038 (<https://bbbc.broadinstitute.org/BBBC038>): "Copyright: CC0. To the extent possible under law, the various contributors of the imagesets have waived all copyright and related or neighboring rights to BBBC038v1." |
| MoNuSeg | Hugging Face mirror `RationAI/MoNuSeg` (train 37 / test 14) | Kumar et al., *IEEE TMI* 36:1550, 2017; Kumar et al., *IEEE TMI* 39:1380, 2020 | <https://monuseg.grand-challenge.org/Data/>: "The challenge data is released under the creative commons license", linked to CC BY-NC-SA 4.0; the mirror's card gives `cc-by-nc-sa-4.0`. **Non-commercial.** |
| BBBC010 | <https://bbbc.broadinstitute.org/BBBC010> | Ljosa et al., *Nat. Methods* 9:637, 2012 | "Copyright: CC0." (the BBBC010 page) |
| Bacteria | Omnipose datasets on OSF, <https://osf.io/xmury> (`datasets/bact_phase`) | Cutler et al., *Nat. Methods* 19:1438, 2022 | The project points to its `license.txt` (<https://osf.io/download/hvku3/>): Creative Commons Attribution-NonCommercial 3.0 Unported. **Non-commercial.** |
| DRIVE | <https://drive.grand-challenge.org/>; the copy used is the Hugging Face mirror `Zomba/DRIVE-digital-retinal-images-for-vessel-extraction` (40 images, first manual annotation) | Staal et al., *IEEE TMI* 23:501, 2004 | The challenge page states no licence or terms for the data. The mirror's card gives `mit`; the mirror is not the data's author. |
| HRF | <https://www5.cs.fau.de/research/data/fundus-images/> (`all.zip`) | Budai et al., *Int. J. Biomed. Imaging* 2013:154860, 2013 | "We release it under Creative Commons 4.0 Attribution License" and "The database can be used freely for research purposes." |
| ISBI2012-EM | <https://imagej.net/events/isbi-2012-segmentation-challenge>; the copy used is the GitHub mirror `alexklibisz/isbi-2012` (30 training slices, 512 × 512) | Arganda-Carreras et al., *Front. Neuroanat.* 9:142, 2015; Cardona et al., *PLoS Biol.* 8:e1000502, 2010 (the data) | No licence is stated on the challenge page, which gives the data after registration on the challenge server; the mirror has no licence file. |
| FISBe | <https://kainmueller-lab.github.io/fisbe/>; Zenodo, <https://doi.org/10.5281/zenodo.10875063> (`fisbe_v1.0_mips.zip`) | Mais et al., CVPR 2024 | "The FlyLight Instance Segmentation Dataset (FISBe) is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license." The Zenodo record gives CC BY 4.0. |

Some consequences:

- **Commercial use:** leave out Bacteria and MoNuSeg (non-commercial licences).
- **SpheroidJ:** the HyperBank release describes its own SpheroidJ subset as CC BY 4.0, inherited from the
  upstream release. The upstream repositories state no such licence for the images; they carry the GPL v3.0.
- **DRIVE and ISBI2012-EM** state no licence on the pages above; the paper used community mirrors of both.

## CTC-U373, evaluated in earlier versions

Earlier versions of the paper, including arXiv version 1, also evaluated CTC-U373, phase-contrast cells of the Cell
Tracking Challenge (PhC-C2DH-U373). The paper no longer reports it, because of its conditions of use
(<https://celltrackingchallenge.net/datasets/>): "Any public non-CTC-related, scientific use of our datasets
requires explicit permission from the challenge organizers", and "Cloning of our datasets or their parts, including
reference annotations, is strictly forbidden." This repository therefore holds no CTC-U373 data and no score records
for it, and the benchmark code does not know the dataset.

## The split protocol

Every method in the paper uses the same protocol, implemented in `benchmark/registry.py` (`load_split`) and
`benchmark/run.py` (`run_dataset`):

1. **Pool and test set are loaded once per dataset, with seed 0:**
   `load_dataset(spec, pool_size, 10000, seed=0)`. The pool holds 20 images (16 for ISBI2012-EM and FISBe). The
   test limit of 10000 means "the whole test split".
2. **Ten seeds draw the support from the fixed pool.** For seed `s`, `rng = numpy.random.default_rng(s)` and the
   support is `rng.choice(len(pool), K, replace=False)`. The same generator then draws the 4096 pixels that fix
   the readout's column shift (see [METHOD.md](METHOD.md)), so the order of the draws is part of the result.
3. **The test images never move.** They are the same for every K, every seed and every method.

Loading the split per seed instead (`load_dataset(spec, K, test, seed=seed)`) would shift the test slice on the
datasets that are split by a permutation (below), so the protocol loads it once. `load_split` refuses data that
give fewer pool images than the protocol needs or no test image.

Some consequences of the pool sizes:

- **Every pool holds at least 16 images**, so Table 1 and Figure 2 average the same ten datasets at every K.
- **On ISBI2012-EM and FISBe at K = 16**, the pool of 16 is drawn whole, so all ten seeds use the same support set.
- **At K = 1**, the ten seeds draw only 7 distinct images from a pool of 20 or 16 (computed from
  `default_rng(s).choice(n, 1)` for s = 0, ..., 9).

No validation split is held out. The constants of the method were set while developing it on these datasets.

## How each split is made

The loaders are in `benchmark/registry.py` and `benchmark/datasets.py`; the preparation is in
`benchmark/prepare_data.py`.

| Loader kind | Datasets | Pool | Test |
|---|---|---|---|
| `fewshot` | SpheroidJ, Decay | the first 20 PNG files, by filename, of `support/images` | every file of `test/images` |
| `dsb` | DSB2018 | the first 20 TIFF files, by filename, of `train/images` | every file of `test/images` |
| `instance` | MoNuSeg, BBBC010, Bacteria, FISBe | the first N TIFF files, by filename, of `train/images` | every file of `test/images` |
| `download` (flat directory) | DRIVE, HRF, ISBI2012-EM | the first N images of one permutation, `default_rng(0).permutation` | the remaining images |

Because the pool is "the first N by filename", the filenames matter, and the preparation script writes the names
the paper's runs read. Per dataset:

- **SpheroidJ**: 72 of the 366 images in `datasets/Images/test` of the Deep-Tumour-Spheroid repository, 18 from a
  Leica 5x microscope (named `BL5S_<n>`) and 54 from a Nikon 2x microscope (`BN2S_<n>`), split into 48 support
  and 24 test images (12 + 36 and 6 + 18 by microscope). They were drawn once with seed 42 and recorded by name;
  the script that drew them was not kept, so `prepare_data.py` lists the 72 images by name. Images are copied
  unchanged; each mask is its label TIFF with every label above zero as foreground. This is not the SpheroidJ
  subset in the HyperBank repository, which is another 72-image draw (from the original SpheroidJ files) and has
  another split fingerprint.
- **Decay**: the 48 support and 24 test images of the HyperBank release, split with seed 42 as its
  `manifest.json` records. The archive is used as it is.
- **DSB2018**: the training (447) and test (50) folders of the StarDist release.
- **MoNuSeg**: the training (37) and test (14) files of the mirror, as the mirror splits them. The challenge page
  describes 30 training images; the mirror's training file holds 37, and the paper used those. Each image is
  written as a TIFF named by its TCGA identifier; its label map numbers the nuclei in the order the file lists
  them.
- **BBBC010**: one image per well, 100 wells, channel `w1` (the channel the annotation is aligned with). Sorted by
  well name, the first 40 wells (A01–A24, B01–B16) are the training split and the other 60 (B17–B24, C01–C24,
  D01–D24, E01–E04) the test split. Both splits hold ampicillin-treated wells (plate columns 1–12) and untreated
  wells (columns 13–24); the pool, A01–A20, holds 12 treated and 8 untreated. The label map of a well numbers its
  worms in the order of their mask files (`A01_01`, `A01_02`, ...); where two worms overlap (1.15 % of the
  foreground pixels), the later one keeps the pixel. The research code kept no preparation script for BBBC010;
  this rule was read from the file names of the data the paper used and is confirmed by the split fingerprint.
- **Bacteria**: the `bact_phase` training (249) and test (148) images, with the `_masks.tif` label maps as they
  are. The training files are renamed round-robin by species, so that the first 20 span all 17 species rather
  than three: the species folders are sorted by name (upper case before lower case), the files within a species
  by name, and pass i takes the i-th file of every species that has one; the files are named
  `<index>__<species>__<file>`. Test files are named `<species>__<file>`. The test split covers 16 species
  (Francisella appears only in the training split).
- **DRIVE**: no official split is used. Both official halves (40 images) go into one flat directory and are
  permuted: 20 pool, 20 test. The official training images keep their numbers 21–40 and the official test images
  01–20; the 20 test images are 11 of the official test images and 9 of the official training images.
- **HRF** defines no official split. All 45 images go into one flat directory and are permuted: 20 pool, 25 test.
- **ISBI2012-EM** withholds the labels of its official test volume, so only the 30 annotated training slices are
  used, permuted into 16 pool and 14 test images. The foreground is the membrane (label 0 in the original labels).
- **FISBe**: only the 30 completely labelled images are used. The partly labelled images are left out, because
  their missing annotations would count correctly found neurons as false positives. The images and labels come
  from the maximum-intensity projections published with the dataset (`fisbe_v1.0_mips.zip`): each projection
  shows the image on the left and the neurons, one colour each, on the right. The image is the left half; the
  label map numbers the colours of the right half in sorted order (black is background). The pool is the 5
  images of the official validation split and 11 of the 18 official training images; the test set is the 7
  official test images and the other 7 official training images (R38F04 G3, G5 and H1, R53A10 A4, R75E01 D1,
  VT047848 J2, VT061467 E5). How those 7 were chosen was not recorded, so `prepare_data.py` lists all 30 images
  by name.

### Checking your split

Each score record stores a fingerprint of the test split it was scored on (`split_fp`, a SHA-256 over every test
image and mask, `benchmark/run.py: split_fingerprint`). The paper's table generators refuse to pair two records
whose fingerprints differ. The pool has no fingerprint in the records; the pool fingerprint below is the same
digest over the pool, computed on data prepared by `prepare_data.py` whose file names equal those of the data the
paper's runs read. `python -m benchmark.prepare_data --check` compares both.

| Dataset | test `split_fp` (paper's records) | pool fingerprint |
|---|---|---|
| spheroidj | `0221b7005adbfcae` | `40714537854d76ff` |
| rozpad | `8017232bf8a5d616` | `caff16786c6721a6` |
| dsb2018 | `6eed932d47871100` | `33db9075dfec9d83` |
| monuseg | `8752244ba838397d` | `64c39b4a9e450e5a` |
| bbbc010 | `8c4b1bdf3ddb991a` | `7cced9e1ca4bfa6e` |
| bacteria | `4200850b1b81f401` | `cb7c8257f3ccfb3e` |
| drive | `5513231d22432d80` | `ae8d7835202abace` |
| hrf | `cbb4f13a09ada9f9` | `d7f62491035ca135` |
| isbi2012em | `274c5d1057f9faf2` | `76c4a6b4f37e5b86` |
| fisbe | `b33e8ce39b94efda` | `05c3371a45190fcc` |

## Directory layout and environment variables

`benchmark/registry.py` reads four environment variables. A relative path is taken from the directory you run in.

| Variable | Default | Holds |
|---|---|---|
| `ASG_DATA_ROOT` | `data` | the data tree; the three variables below default to folders inside it |
| `SPHEROID_ROOT` | `$ASG_DATA_ROOT/spheroids` | SpheroidJ and Decay |
| `DSB_ROOT` | `$ASG_DATA_ROOT/dsb2018` | DSB2018 (downloaded here when missing) |
| `PANEL_DL_ROOT` | `$ASG_DATA_ROOT/panel_datasets` | the other seven datasets |

The research code also knew a `SPHEROSEG_ROOT` variable. It points at datasets that are not in the paper's panel,
and this package does not read it.

The layout `prepare_data.py` writes:

```
$SPHEROID_ROOT/dataset_fewshot_spheroidj/{support,test}/{images,masks}/*.png
$SPHEROID_ROOT/dataset_fewshot_rozpad/{support,test}/{images,masks}/*.png
$DSB_ROOT/dsb2018/{train,test}/{images,masks}/*.tif
$PANEL_DL_ROOT/monuseg/{train,test}/{images,masks}/*.tif
$PANEL_DL_ROOT/bbbc010/{train,test}/{images,masks}/*.tif
$PANEL_DL_ROOT/bacteria/{train,test}/{images,masks}/*.tif
$PANEL_DL_ROOT/fisbe/{train,test}/{images,masks}/*.tif
$PANEL_DL_ROOT/drive/{images,masks}/
$PANEL_DL_ROOT/hrf/{images,manual1}/
$PANEL_DL_ROOT/isbi2012em/{images,masks}/
```

Rules the loaders apply:

- `fewshot`: an image and its mask share a filename. Masks are binarised (> 0).
- `instance` and `dsb`: an image and its mask share a filename (`.tif`). Masks are label maps (uint16 or int32),
  one label per object. Exactly one `{split}/images` folder may exist under the dataset's root; the loaders refuse
  more than one.
- `download` (flat): the mask has the image's stem with any of the extensions `.png .jpg .jpeg .tif .tiff .bmp`.
  Masks are binarised (> 0). An image without a mask is an error.

DSB2018 and HRF are also downloaded by the loaders themselves on first use when their folder is missing (into
`DSB_ROOT` and `$PANEL_DL_ROOT/hrf`), with the same result as the preparation script.

The converters of DRIVE, ISBI2012-EM and MoNuSeg follow the preparation script of the research code
(AutoSeg's `scripts/prep_public_datasets.py`), with two differences that leave the data unchanged: the mirrors are
pinned to a revision, and MoNuSeg files are named by their identifier alone, as in the data the paper used.
