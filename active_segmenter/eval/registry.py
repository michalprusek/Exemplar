"""Dataset registry — the broad benchmark panel.

The DSB-only geoloop result was misleading precisely because it was one dataset; this panel
makes "always test on a wide range" the default. Every entry resolves to a uniform
``(support_pairs, test_pairs)`` few-shot split via :func:`load_dataset`, so the panel runner
treats them identically. Datasets that will not load (missing on this host, download failed)
are skipped by the runner, never crash the panel.

Roots default to the tulen layout and are overridable by env var. Kinds:
- ``fewshot``   — ``{root}/{support,test}/{images,masks}`` (the prusek_spheroid variants)
- ``traintest`` — ``{root}/{train,test}/{images,masks}`` big splits, random-subsampled (SpheroSeg)
- ``dsb``       — DSB2018 tif per-instance masks (downloads on miss)
- ``download``  — fetch+extract a zip, then split a flat images/masks dir (Kvasir-SEG, …)
- ``instance``  — prebuilt ``{root}/{train,test}/{images,masks}/*.tif`` per-instance uint16
                  label maps (MoNuSeg / Cell-Tracking-Challenge). Like ``dsb`` but reads a
                  locally-prepared dir (``scripts/prep_public_datasets.py``) and never downloads.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from active_segmenter.eval import datasets as D

# One knob for the whole data tree, so a second GPU host that lacks /disk1 (kajman: /scratch only)
# can be pointed at its copy with a single export instead of one env var per dataset. The default is
# tulen's literal layout, so an unset environment reproduces tulen's paths byte-for-byte. The
# per-dataset overrides below still win when set, for a tree that is only partially relocated.
_DATA = os.environ.get("ASG_DATA_ROOT", "/disk1/prusek")

# NOT derived from _DATA: the spheroid sets live on the NFS home, a different filesystem from the
# /disk1 scratch tree, so relocating _DATA must not silently move them. They get their own override.
_SPHEROID = os.environ.get("SPHEROID_ROOT", "/home/prusek/prusek_spheroid")
_SPHEROSEG = os.environ.get("SPHEROSEG_ROOT", f"{_DATA}/SpheroSeg/data")
_DSB = os.environ.get("DSB_ROOT", f"{_DATA}/dsb2018")
_DL = os.environ.get("PANEL_DL_ROOT", f"{_DATA}/panel_datasets")


@dataclass
class DatasetSpec:
    name: str
    kind: str            # fewshot | traintest | dsb | download
    root: str
    url: str = ""        # download kind only
    images_sub: str = "" # download kind: relative images dir inside the extracted root
    masks_sub: str = ""  # download kind: relative masks dir
    note: str = ""       # morphology/domain, for the report
    # The metric APPROPRIATE for this dataset's structure — every benchmark evaluates with
    # the metric designed for it (user directive), not one-size-fits-all fg-IoU:
    #   "instance_ap" — per-instance GT (nuclei) -> the DSB/Kaggle AP@[.5:.95]
    #   "cldice"      — tubular/thin structures (vessels, filaments) -> centerline Dice
    #   "iou"         — semantic blobs (spheroids, polyps) -> fg-IoU (+boundary-F reported too)
    metric: str = "iou"


PANEL: dict[str, DatasetSpec] = {
    "dsb2018":   DatasetSpec("dsb2018", "dsb", _DSB, metric="instance_ap",
                             note="fluorescence nuclei, per-instance GT"),
    "spheroid":  DatasetSpec("spheroid", "fewshot", f"{_SPHEROID}/dataset_fewshot", metric="iou",
                             note="brightfield spheroids"),
    "rozpad":    DatasetSpec("rozpad", "fewshot", f"{_SPHEROID}/dataset_fewshot_rozpad", metric="iou",
                             note="decay spheroids, small crumbs"),
    "spheroidj": DatasetSpec("spheroidj", "fewshot", f"{_SPHEROID}/dataset_fewshot_spheroidj", metric="iou",
                             note="spheroid variant J"),
    "spherohq":  DatasetSpec("spherohq", "traintest", f"{_SPHEROSEG}/SpheroHQ", metric="iou",
                             note="high-quality spheroids"),
    "spheromix": DatasetSpec("spheromix", "traintest", f"{_SPHEROSEG}/SpheroMix", metric="iou",
                             note="mixed spheroids"),
    "kvasir":    DatasetSpec("kvasir", "download", f"{_DL}/kvasir-seg", url=D.KVASIR_URL,
                             images_sub="Kvasir-SEG/images", masks_sub="Kvasir-SEG/masks",
                             metric="iou", note="GI-endoscopy polyps (non-microscopy)"),
    "hrf":       DatasetSpec("hrf", "download", f"{_DL}/hrf",
                             url="https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip",
                             images_sub="images", masks_sub="manual1", metric="cldice",
                             note="retinal fundus VESSELS — THIN structures (filament proxy)"),
    # ---- Standard public biomedical datasets (WACV panel; prep via scripts/prep_public_datasets.py) ----
    "drive":     DatasetSpec("drive", "download", f"{_DL}/drive",
                             images_sub="images", masks_sub="masks", metric="cldice",
                             note="DRIVE retinal fundus VESSELS — thin tubular; HF Zomba mirror (40 imgs, 1st-manual)"),
    "isbi2012em": DatasetSpec("isbi2012em", "download", f"{_DL}/isbi2012em",
                             images_sub="images", masks_sub="masks", metric="cldice",
                             note="ISBI2012 EM neuronal MEMBRANES — thin; GitHub mirror (30 slices, fg=membrane=label0)"),
    "monuseg":   DatasetSpec("monuseg", "instance", f"{_DL}/monuseg", metric="instance_ap",
                             note="MoNuSeg H&E histopathology NUCLEI, per-instance; HF RationAI mirror (train37/test14)"),
    "ctc_u373":  DatasetSpec("ctc_u373", "instance", f"{_DL}/ctc_u373", metric="instance_ap",
                             note="Cell-Tracking-Challenge PhC-C2DH-U373 cells, per-instance; seq01 train(15)/seq02 test(19)"),
    "bacteria":  DatasetSpec("bacteria", "instance", f"{_DL}/bacteria", metric="instance_ap",
                             note="Omnipose BPCIS bact_phase (Cutler et al., Nature Methods 2022; OSF xmury, CC BY-NC). "
                                  "17 bacterial species (test covers 16; francisella is pool-only) — rods, cocci, curved, branched, filamentous — a morphology "
                                  "class absent from the rest of the panel. 249 pool / 148 test, 47k instances "
                                  "(mean 119/image, very dense). Pool files are renamed round-robin by species so the alphabetical paths[:20] the loader takes spans all 17 species, not the 3 it would otherwise."),
    "bbbc010":   DatasetSpec("bbbc010", "instance", f"{_DL}/bbbc010", metric="instance_ap",
                             note="BBBC010 C. elegans live/dead (Broad, CC0). Scored on the channel the GT aligns with, w1 (byte-identical in 100/100 wells; "
                                  "98.9% of annotated worms are invisible in w2). Curved elongated OVERLAPPING bodies — a "
                                  "morphology that is neither blob nor filament nor membrane. train40(pool)/test60."),
    "fisbe":     DatasetSpec("fisbe", "instance", f"{_DL}/fisbe", metric="cldice",
                             note="FISBe (Mais et al., CVPR 2024) Drosophila neurons — the first public benchmark for "
                                  "long THIN FILAMENTOUS instances; multicolor LM MIPs, CC BY 4.0. ONLY the 30 "
                                  "'completely' labelled images (the 'partly' subset has deliberately incomplete GT, "
                                  "which would score unannotated neurons as false positives). train16(pool)/test14."),
}


def _load_instance_split(root: str, split: str, limit: int | None):
    """Read a prebuilt ``{root}/**/{split}/images/*.tif`` + matching ``masks/*.tif`` split whose
    masks are per-instance uint16 label maps (MoNuSeg, Cell-Tracking-Challenge). Same tif layout
    as DSB2018 but NEVER downloads — the dir is produced by ``scripts/prep_public_datasets.py``.
    Returns ``list[(image, label_map[int32])]``."""
    import glob as _glob

    import numpy as np
    import tifffile

    hits = sorted(_glob.glob(os.path.join(root, "**", split, "images"), recursive=True))
    if not hits:
        raise FileNotFoundError(f"instance dataset: {split}/images not under {root} "
                                f"(run scripts/prep_public_datasets.py)")
    if len(hits) > 1:
        raise RuntimeError(f"instance dataset: {len(hits)} '{split}/images' dirs under {root} "
                           f"— ambiguous (leftover staging?): {hits}")
    base = hits[0]
    mbase = os.path.join(os.path.dirname(base), "masks")   # sibling of images/, not str.replace (root-safe)
    paths = sorted(_glob.glob(os.path.join(base, "*.tif")))
    if not paths:                                          # empty dir must fail loud, not return [] (→ "0 imgs")
        raise FileNotFoundError(f"instance dataset: no *.tif in {base} (prep crashed / wrong extension?)")
    if limit is not None:
        paths = paths[:limit]
    out = []
    for ip in paths:
        im = tifffile.imread(ip)
        mp = os.path.join(mbase, os.path.basename(ip))
        if not os.path.exists(mp):                         # a missing mask must error, not misalign pairs
            raise FileNotFoundError(f"instance dataset: mask missing for {ip} (expected {mp})")
        out.append((im, tifffile.imread(mp).astype(np.int32)))
    return out


def load_dataset(spec: DatasetSpec, support: int, test: int, seed: int = 0):
    """Return ``(support_pairs, test_pairs)`` = ``list[(image, label_map)]`` for any kind.
    Raises on genuine failure (missing data / download error); the runner catches to skip."""
    if spec.kind == "fewshot":
        return (D.load_fewshot(spec.root, "support", support),
                D.load_fewshot(spec.root, "test", test))
    if spec.kind == "traintest":
        return (D.load_split_dir(spec.root, "train", support, seed=seed),
                D.load_split_dir(spec.root, "test", test, seed=seed))
    if spec.kind == "dsb":
        return (D.load_dsb2018(spec.root, "train", support),
                D.load_dsb2018(spec.root, "test", test))
    if spec.kind == "instance":
        return (_load_instance_split(spec.root, "train", support),
                _load_instance_split(spec.root, "test", test))
    if spec.kind == "download":
        img_dir = os.path.join(spec.root, spec.images_sub)
        if not os.path.isdir(img_dir):
            D.download_and_extract(spec.url, spec.root)
        return D.load_flat_fewshot(img_dir, os.path.join(spec.root, spec.masks_sub),
                                   support, test, seed=seed)
    raise ValueError(f"unknown dataset kind: {spec.kind}")
