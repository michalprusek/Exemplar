"""Loaders of the benchmark datasets, each returning list[(image, label_map)].

- `load_fewshot`: {root}/{support,test}/{images,masks}/*.png, binary masks (the spheroid sets), first N by filename.
- `load_flat_fewshot`: a flat images/ + masks/ directory split into a pool and a test set by one seeded permutation
  (pool first, the rest is the test set) - DRIVE, HRF, ISBI2012-EM.
- `load_dsb2018`: DSB2018 (StarDist release) tif per-instance label maps, downloaded on a miss.

Source: active_segmenter/eval/datasets.py (AutoSeg), the loaders the panel uses, unchanged.
"""
from __future__ import annotations

import glob
import io
import os
import zipfile

import numpy as np
import tifffile

DSB_URL = "https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip"
_IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def download_and_extract(url: str, root: str) -> None:
    import requests

    os.makedirs(root, exist_ok=True)
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(root)


def load_fewshot(root: str, split: str, limit: int | None = None):
    """`{root}/{split}/{images,masks}/*.png`; masks binarised to {0, 1}; images sorted by filename; "train" aliases
    "support"."""
    from PIL import Image

    if split == "train":
        split = "support"
    img_dir = os.path.join(root, split, "images")
    msk_dir = os.path.join(root, split, "masks")
    paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"no PNG images under {img_dir}")
    if limit is not None:
        paths = paths[:limit]
    out = []
    for ip in paths:
        im = np.asarray(Image.open(ip))
        mk = np.asarray(Image.open(os.path.join(msk_dir, os.path.basename(ip))))
        if mk.ndim == 3:
            mk = mk[..., 0]
        out.append((im, (mk > 0).astype(np.int32)))
    return out


def _read_mask(path: str) -> np.ndarray:
    from PIL import Image

    m = np.asarray(Image.open(path))
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.int32)


def _mask_for(img_path: str, msk_dir: str) -> str | None:
    """The mask whose basename stem matches the image (the extension may differ)."""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    for ext in _IMG_EXT:
        cand = os.path.join(msk_dir, stem + ext)
        if os.path.exists(cand):
            return cand
    exact = os.path.join(msk_dir, os.path.basename(img_path))
    return exact if os.path.exists(exact) else None


def load_flat_fewshot(img_dir: str, msk_dir: str, support: int, test: int, seed: int = 0):
    """Split a flat images/ + masks/ directory into disjoint pool and test lists by a permutation drawn with `seed`.
    Every image needs a mask (the research loader dropped images without one; none of the paper's datasets has any)."""
    from PIL import Image

    paths = sorted(p for p in glob.glob(os.path.join(img_dir, "*")) if p.lower().endswith(_IMG_EXT))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths)).tolist()
    pick = order[: support + test]
    chosen = [paths[i] for i in pick]

    def _pairs(subset):
        out = []
        for ip in subset:
            mp = _mask_for(ip, msk_dir)
            if mp is None:      # a dropped image would shift the permutation's split without a sign
                raise FileNotFoundError(f"no mask for {ip} in {msk_dir}")
            out.append((np.asarray(Image.open(ip)), _read_mask(mp)))
        return out

    return _pairs(chosen[:support]), _pairs(chosen[support:])


def download_dsb2018(root: str) -> None:
    import requests

    os.makedirs(root, exist_ok=True)
    r = requests.get(DSB_URL, timeout=300)
    r.raise_for_status()
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(root)


def _split_dir(root: str, split: str) -> str:
    hits = sorted(glob.glob(os.path.join(root, "**", split, "images"), recursive=True))
    if not hits:
        raise FileNotFoundError(f"DSB2018 {split}/images not under {root}")
    if len(hits) > 1:
        raise RuntimeError(f"DSB2018: {len(hits)} '{split}/images' dirs under {root}: {hits}")
    return hits[0]


def load_dsb2018(root: str, split: str, limit: int | None = None):
    """DSB2018 (StarDist release): list[(image, per-instance label map int32)], downloaded on a miss."""
    if not glob.glob(os.path.join(root, "**", split, "images"), recursive=True):
        download_dsb2018(root)
    base = _split_dir(root, split)
    mbase = os.path.join(os.path.dirname(base), "masks")
    paths = sorted(glob.glob(base + "/*.tif"))
    if limit is not None:
        paths = paths[:limit]
    out = []
    for ip in paths:
        im = tifffile.imread(ip)
        lbl = tifffile.imread(os.path.join(mbase, os.path.basename(ip)))
        out.append((im, lbl.astype(np.int32)))
    return out
