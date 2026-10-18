"""The ten datasets of the paper's panel and the split the benchmark uses.

Every dataset resolves to (pool, test) = (list[(image, label_map)], list[...]) through `load_dataset`. The benchmark
loads the pool (20 images; 16 for ISBI2012-EM and FISBe) and the full test split ONCE with seed 0 and
subsamples the K support masks from the fixed pool per seed, so the test images never move with K or the seed.

Roots follow the AutoSeg layout and are set by environment variables:
    ASG_DATA_ROOT   the data tree (default ./data, relative to the directory you run from)
    SPHEROID_ROOT   SpheroidJ and Decay (default $ASG_DATA_ROOT/spheroids)
    DSB_ROOT        DSB2018 (default $ASG_DATA_ROOT/dsb2018; downloaded on a miss)
    PANEL_DL_ROOT   the other seven datasets (default $ASG_DATA_ROOT/panel_datasets)
`python -m benchmark.prepare_data all` downloads and lays out every dataset (docs/DATASETS.md).

Source: active_segmenter/eval/registry.py (AutoSeg) for the panel entries, `_load_instance_split` and `load_dataset`;
POOL and CLDICE from scripts/closed_form_bench.py:54-55. The research code also lists CTC-U373, which earlier versions
of the paper evaluated; the paper no longer reports it, and this package leaves it out (docs/DATASETS.md).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from benchmark import datasets as D

_DATA = os.environ.get("ASG_DATA_ROOT", "data")
_SPHEROID = os.environ.get("SPHEROID_ROOT", f"{_DATA}/spheroids")
_DSB = os.environ.get("DSB_ROOT", f"{_DATA}/dsb2018")
_DL = os.environ.get("PANEL_DL_ROOT", f"{_DATA}/panel_datasets")

ALL_DATASETS = ["spheroidj", "rozpad", "dsb2018", "monuseg", "bbbc010",
                "bacteria", "drive", "hrf", "isbi2012em", "fisbe"]
POOL = {"isbi2012em": 16, "fisbe": 16}                         # the support pool; 20 for every other dataset
CLDICE = {"drive", "hrf", "isbi2012em", "fisbe"}               # scored by centreline Dice; the rest by foreground IoU


@dataclass
class DatasetSpec:
    name: str
    kind: str            # fewshot | dsb | download | instance
    root: str
    url: str = ""        # download kind only
    images_sub: str = "" # download kind: images dir inside the root
    masks_sub: str = ""  # download kind: masks dir inside the root
    note: str = ""


PANEL: dict[str, DatasetSpec] = {
    "spheroidj": DatasetSpec("spheroidj", "fewshot", f"{_SPHEROID}/dataset_fewshot_spheroidj", note="spheroid variant J"),
    "rozpad":    DatasetSpec("rozpad", "fewshot", f"{_SPHEROID}/dataset_fewshot_rozpad", note="decaying spheroids"),
    "dsb2018":   DatasetSpec("dsb2018", "dsb", _DSB, note="fluorescence nuclei"),
    "monuseg":   DatasetSpec("monuseg", "instance", f"{_DL}/monuseg", note="MoNuSeg H&E nuclei (train37/test14)"),
    "bbbc010":   DatasetSpec("bbbc010", "instance", f"{_DL}/bbbc010", note="BBBC010 C. elegans, channel w1 (train40/test60)"),
    "bacteria":  DatasetSpec("bacteria", "instance", f"{_DL}/bacteria", note="Omnipose bact_phase (249 pool / 148 test)"),
    "drive":     DatasetSpec("drive", "download", f"{_DL}/drive", images_sub="images", masks_sub="masks",
                             note="DRIVE retinal vessels (40 images, 1st manual)"),
    "hrf":       DatasetSpec("hrf", "download", f"{_DL}/hrf",
                             url="https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip",
                             images_sub="images", masks_sub="manual1", note="HRF retinal vessels"),
    "isbi2012em": DatasetSpec("isbi2012em", "download", f"{_DL}/isbi2012em", images_sub="images", masks_sub="masks",
                              note="ISBI 2012 EM membranes (30 slices)"),
    "fisbe":     DatasetSpec("fisbe", "instance", f"{_DL}/fisbe", note="FISBe neurons, the 30 completely labelled images"),
}


def _load_instance_split(root: str, split: str, limit: int | None):
    """A prepared `{root}/**/{split}/images/*.tif` + matching `masks/*.tif` split (per-instance uint16 label maps);
    never downloads. Returns list[(image, label_map int32)]."""
    import glob as _glob

    import numpy as np
    import tifffile

    hits = sorted(_glob.glob(os.path.join(root, "**", split, "images"), recursive=True))
    if not hits:
        raise FileNotFoundError(f"instance dataset: {split}/images not under {root} (prepare it with: "
                                f"python -m benchmark.prepare_data {os.path.basename(root)}; see docs/DATASETS.md)")
    if len(hits) > 1:
        raise RuntimeError(f"instance dataset: {len(hits)} '{split}/images' dirs under {root}: {hits}")
    base = hits[0]
    mbase = os.path.join(os.path.dirname(base), "masks")
    paths = sorted(_glob.glob(os.path.join(base, "*.tif")))
    if not paths:
        raise FileNotFoundError(f"instance dataset: no *.tif in {base}")
    if limit is not None:
        paths = paths[:limit]
    out = []
    for ip in paths:
        im = tifffile.imread(ip)
        mp = os.path.join(mbase, os.path.basename(ip))
        if not os.path.exists(mp):
            raise FileNotFoundError(f"instance dataset: mask missing for {ip} (expected {mp})")
        out.append((im, tifffile.imread(mp).astype(np.int32)))
    return out


def load_dataset(spec: DatasetSpec, support: int, test: int, seed: int = 0):
    """(pool, test) for any kind of dataset."""
    if spec.kind == "fewshot":
        return (D.load_fewshot(spec.root, "support", support),
                D.load_fewshot(spec.root, "test", test))
    if spec.kind == "dsb":
        return (D.load_dsb2018(spec.root, "train", support),
                D.load_dsb2018(spec.root, "test", test))
    if spec.kind == "instance":
        return (_load_instance_split(spec.root, "train", support),
                _load_instance_split(spec.root, "test", test))
    if spec.kind == "download":
        img_dir = os.path.join(spec.root, spec.images_sub)
        if not os.path.isdir(img_dir):
            if not spec.url:
                raise FileNotFoundError(f"{spec.name}: {img_dir} missing (prepare it with: "
                                        f"python -m benchmark.prepare_data {spec.name}; see docs/DATASETS.md)")
            D.download_and_extract(spec.url, spec.root)
        return D.load_flat_fewshot(img_dir, os.path.join(spec.root, spec.masks_sub), support, test, seed=seed)
    raise ValueError(f"unknown dataset kind: {spec.kind}")


def load_split(name: str):
    """The split the benchmark uses: (pool, test), loaded once with seed 0. (closed_form_bench.py:2358)
    Raises when the prepared data hold fewer pool images than the protocol needs or no test image, so an
    incomplete tree fails here instead of running on another split."""
    n_pool = POOL.get(name, 20)
    pool, test = load_dataset(PANEL[name], n_pool, 10000, seed=0)
    if len(pool) < n_pool or not test:
        raise RuntimeError(f"{name}: the prepared data give {len(pool)} pool images (need {n_pool}) and {len(test)} "
                           f"test images under {PANEL[name].root}; re-run python -m benchmark.prepare_data {name} "
                           f"and check it with python -m benchmark.prepare_data --check {name}")
    return pool, test
