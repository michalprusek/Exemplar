# Datasets

**No dataset is redistributed in this repository.** Each is public and keeps its own license; the
table below is where to get them and what you are allowed to do with them. Point `PANEL_DL_ROOT` at
the directory you download into (default: `$ASG_DATA_ROOT/panel_datasets`), and set `ASG_DATA_ROOT`
to wherever your data lives.

The eleven datasets in the paper span the morphologies deliberately: if one pipeline is to serve a
biologist without per-dataset tuning, it has to hold up on round blobs, crowded stained nuclei,
overlapping worms, dense rods, thin vessels, membranes, and filaments alike.

| Registry key | Name | Morphology | Metric | Source | License |
|---|---|---|---|---|---|
| `spheroidj` | SpheroidJ | spheroids | fg IoU | [SpheroidJ](https://github.com/BIIG-UC3M/SpheroidJ) | see source |
| `rozpad` | Decay | decaying spheroids | fg IoU | released with our HyperBank work | see source |
| `dsb2018` | Data Science Bowl 2018 | nuclei | fg IoU | [Kaggle DSB2018](https://www.kaggle.com/c/data-science-bowl-2018) | CC0 / competition terms |
| `monuseg` | MoNuSeg | H&E nuclei | fg IoU | [HF RationAI/MoNuSeg](https://huggingface.co/datasets/RationAI/MoNuSeg) | CC-BY-NC-SA |
| `ctc_u373` | CTC PhC-C2DH-U373 | phase-contrast cells | fg IoU | [Cell Tracking Challenge](https://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip) | CC-BY 4.0 |
| `bbbc010` | BBBC010 | *C. elegans*, overlapping | fg IoU | [Broad BBBC010](https://bbbc.broadinstitute.org/BBBC010) | CC-BY 3.0 |
| `bacteria` | Omnipose / BPCIS | dense bacterial rods | fg IoU | [OSF osf.io/xmury](https://osf.io/xmury) | **CC-BY-NC 3.0 — non-commercial** |
| `drive` | DRIVE | retinal vessels | centreline Dice | [DRIVE](https://drive.grand-challenge.org/) | research use, registration |
| `hrf` | HRF | retinal vessels | centreline Dice | [HRF](https://www5.cs.fau.de/research/data/fundus-images/) | CC-BY 4.0 |
| `isbi2012em` | ISBI 2012 EM | neuronal membranes | centreline Dice | [ISBI 2012 challenge](https://imagej.net/events/isbi-2012-segmentation-challenge) | see source |
| `fisbe` | FISBE | thin fluorescent filaments | centreline Dice | [FISBE](https://kainmueller-lab.github.io/fisbe/) | CC-BY 4.0 |

## Automated preparation

`scripts/prep_public_datasets.py` downloads and prepares four of them end to end — DRIVE,
ISBI2012-EM, MoNuSeg, and CTC-U373 — converting each to the uncompressed uint16 label-map layout the
registry expects and asserting the result is non-blank. The rest must be downloaded manually from
the links above and placed under `PANEL_DL_ROOT`; `active_segmenter/eval/registry.py` documents the
directory layout each one expects.

```bash
PANEL_DL_ROOT=/path/to/data python scripts/prep_public_datasets.py
```

The preparation step asserts rather than warns: a dataset that arrives blank, mis-paired, or with
the wrong label dtype fails immediately. Silent data corruption is the failure mode that quietly
poisons every downstream number, so it is checked at the only point where it is cheap to catch.

## A note on licenses

The Bacteria (Omnipose/BPCIS) data is **CC-BY-NC-3.0**: non-commercial use only. MoNuSeg is
similarly non-commercial. If you intend to build on this work commercially, drop those two datasets
from the panel — the code takes `ASG_DATASETS` to scope a run to a subset.
