"""Benchmark datasets.

- DSB2018 nuclei (StarDist release): per-instance label maps, the spike's benchmark
  and our regression anchor. Downloaded on miss.
- Synthetic overlap generator (Task 9): translucent discs with controlled, known
  overlap; GT is a list of per-instance boolean masks so overlap is representable.
"""
from __future__ import annotations

import glob
import io
import os
import zipfile

import numpy as np
import tifffile

DSB_URL = "https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip"


def download_dsb2018(root: str) -> None:
    import requests

    os.makedirs(root, exist_ok=True)
    data = requests.get(DSB_URL, timeout=300).content
    zipfile.ZipFile(io.BytesIO(data)).extractall(root)


def _split_dir(root: str, split: str) -> str:
    hits = glob.glob(os.path.join(root, "**", split, "images"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"DSB2018 {split}/images not under {root}")
    return hits[0]


def make_synthetic_overlap(
    n_images: int,
    n_instances: int = 3,
    overlap_frac: float = 0.3,
    seed: int = 0,
    size: int = 128,
):
    """Translucent overlapping discs with a KNOWN, controlled amount of overlap.

    Returns ``list[(image[H,W], list[mask])]`` where GT is a list of per-instance
    boolean masks (NOT a label map) so overlap is representable — the whole point:
    DSB2018 nuclei do not overlap, so this is where the amodal-overlap claim is tested.
    ``overlap_frac`` in [0, 1): 0 = discs just touch, higher = more overlap.
    """
    rng = np.random.default_rng(seed)
    radius = size // 8
    shift = int(2 * radius * (1 - overlap_frac))  # center-to-center spacing
    yy, xx = np.mgrid[0:size, 0:size]
    out = []
    for _ in range(n_images):
        cy = size // 2 + rng.integers(-radius, radius)
        cx0 = size // 2 - shift * (n_instances - 1) // 2
        masks, image = [], np.zeros((size, size), np.float32)
        for k in range(n_instances):
            cx = cx0 + k * shift + int(rng.integers(-2, 3))
            r = radius + int(rng.integers(-2, 3))
            disc = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
            masks.append(disc.astype(bool))
            image += disc * (120 + 30 * (k % 3))  # translucent: overlaps sum brighter
        image = np.clip(image, 0, 255).astype(np.uint8)
        out.append((image, masks))
    return out


# Visually distinct "domains" with ALTERNATING contrast polarity (odd domains are
# inverted: fg darker than bg). An uncovered opposite-polarity domain reliably
# mis-segments — its blobs resemble the bank's background and score as background,
# so recall drops to ~0. The bank MUST hold an exemplar of each domain, which is
# exactly what makes coverage (TypiClust) beat random: real AL headroom.
_DOMAINS = [
    {"fg": 210, "bg": 40, "tex": 0.0, "r": (10, 14)},   # 0: bright on dark
    {"fg": 40, "bg": 210, "tex": 0.0, "r": (10, 14)},   # 1: dark on bright (inverted)
    {"fg": 170, "bg": 90, "tex": 0.0, "r": (10, 14)},   # 2: mid-bright on mid-dark
    {"fg": 90, "bg": 170, "tex": 0.0, "r": (10, 14)},   # 3: mid-dark on mid-bright (inv)
    {"fg": 235, "bg": 120, "tex": 0.0, "r": (10, 14)},  # 4: very bright on mid
    {"fg": 120, "bg": 235, "tex": 0.0, "r": (10, 14)},  # 5: mid on very bright (inv)
    {"fg": 160, "bg": 60, "tex": 32.0, "r": (10, 14)},  # 6: textured bright
    {"fg": 60, "bg": 160, "tex": 32.0, "r": (10, 14)},  # 7: textured dark (inv)
]


def make_heterogeneous(n_images, n_domains=3, skew=True, seed=0, size=128,
                       max_blobs=4, return_domains=False):
    """Heterogeneous pool of `n_domains` visually distinct domains (for AL headroom).

    If `skew`, domains follow a decaying distribution (rare domains exist) so uniform
    random sampling tends to miss them while coverage (TypiClust) finds them. Returns
    `list[(image[H,W] uint8, label_map[H,W] int per-instance)]` (+ domain ids if asked)."""
    rng = np.random.default_rng(seed)
    doms = list(range(min(n_domains, len(_DOMAINS))))
    if skew:
        w = np.array([0.5 ** d for d in doms], float)
    else:
        w = np.ones(len(doms), float)
    w /= w.sum()
    yy, xx = np.mgrid[0:size, 0:size]
    out, dom_ids = [], []
    for _ in range(n_images):
        d = int(rng.choice(doms, p=w))
        spec = _DOMAINS[d]
        img = np.full((size, size), spec["bg"], np.float32)
        if spec["tex"] > 0:
            img += spec["tex"] * np.sin(xx / 5.0) * np.cos(yy / 5.0)
        lbl = np.zeros((size, size), np.int32)
        for inst in range(1, rng.integers(2, max_blobs + 1) + 1):
            cy, cx = rng.integers(15, size - 15), rng.integers(15, size - 15)
            r = rng.integers(*spec["r"])
            disc = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
            img[disc] = spec["fg"] + (spec["tex"] * np.sin(xx / 4.0))[disc]
            lbl[disc] = inst
        img += rng.normal(0, 4, img.shape)
        out.append((np.clip(img, 0, 255).astype(np.uint8), lbl))
        dom_ids.append(d)
    return (out, dom_ids) if return_domains else out


def load_fewshot(root: str, split: str, limit: int | None = None):
    """Load a ``{root}/{split}/{images,masks}/*.png`` few-shot dataset (support/test
    layout, e.g. the decay/rozpad spheroid set). Masks are binary (0/255) foreground →
    ``{0,1}`` label maps. ``split="train"`` aliases to ``"support"``. Returns
    ``list[(image[H,W], label_map[H,W] int)]``, images sorted by filename."""
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


_IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _read_mask(path: str) -> np.ndarray:
    from PIL import Image

    m = np.asarray(Image.open(path))
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.int32)


def _mask_for(img_path: str, msk_dir: str) -> str | None:
    """Find the mask whose basename stem matches the image (extension may differ)."""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    for ext in _IMG_EXT:
        cand = os.path.join(msk_dir, stem + ext)
        if os.path.exists(cand):
            return cand
    exact = os.path.join(msk_dir, os.path.basename(img_path))
    return exact if os.path.exists(exact) else None


def load_split_dir(root: str, split: str, limit: int | None = None, seed: int = 0):
    """Load ``{root}/{split}/{images,masks}/*`` (any image extension), binarising masks to
    ``{0,1}``. Unlike ``load_fewshot`` (first-N), this randomly subsamples ``limit`` by
    ``seed`` — for big train/test splits (SpheroHQ/SpheroMix). Returns ``list[(image, label)]``."""
    from PIL import Image

    img_dir = os.path.join(root, split, "images")
    msk_dir = os.path.join(root, split, "masks")
    paths = sorted(p for p in glob.glob(os.path.join(img_dir, "*")) if p.lower().endswith(_IMG_EXT))
    if not paths:
        raise FileNotFoundError(f"no images under {img_dir}")
    if limit is not None and len(paths) > limit:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(paths), size=limit, replace=False).tolist())
        paths = [paths[i] for i in idx]
    out = []
    for ip in paths:
        mp = _mask_for(ip, msk_dir)
        if mp is None:
            continue
        out.append((np.asarray(Image.open(ip)), _read_mask(mp)))
    return out


KVASIR_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"


def download_and_extract(url: str, root: str) -> None:
    import requests

    os.makedirs(root, exist_ok=True)
    data = requests.get(url, timeout=600).content
    zipfile.ZipFile(io.BytesIO(data)).extractall(root)


def load_flat_fewshot(img_dir: str, msk_dir: str, support: int, test: int, seed: int = 0):
    """Split a FLAT ``images/`` + ``masks/`` directory (e.g. Kvasir-SEG) into disjoint
    support/test few-shot lists, chosen deterministically by ``seed``."""
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
            if mp is not None:
                out.append((np.asarray(Image.open(ip)), _read_mask(mp)))
        return out

    return _pairs(chosen[:support]), _pairs(chosen[support:])


def load_dsb2018(root: str, split: str, limit: int | None = None):
    """Returns ``list[(image[H,W], label_map[H,W] int per-instance)]``."""
    if not glob.glob(os.path.join(root, "**", split, "images"), recursive=True):
        download_dsb2018(root)
    base = _split_dir(root, split)
    mbase = base.replace("images", "masks")
    paths = sorted(glob.glob(base + "/*.tif"))
    if limit is not None:
        paths = paths[:limit]
    out = []
    for ip in paths:
        im = tifffile.imread(ip)
        lbl = tifffile.imread(os.path.join(mbase, os.path.basename(ip)))
        out.append((im, lbl.astype(np.int32)))
    return out
