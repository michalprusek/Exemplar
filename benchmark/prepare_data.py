#!/usr/bin/env python3
"""Download the paper's ten datasets and lay them out as benchmark/registry.py reads them.

    python -m benchmark.prepare_data all              # or name some: spheroidj rozpad dsb2018 monuseg ...
    python -m benchmark.prepare_data --check          # compare the prepared splits with the paper's
    python -m benchmark.prepare_data --check drive hrf

The datasets go under the roots of benchmark/registry.py (ASG_DATA_ROOT, default ./data; see docs/DATASETS.md). The
downloaded archives are kept in $ASG_DATA_ROOT/_downloads, so a second run does not fetch them again; delete that
folder when the check has passed. A dataset whose folder exists already is left alone; --force rebuilds it.

Each converter writes the file names the paper's runs read, because the split follows from them: the pool is the
first N files by name (the first N of one seeded permutation for DRIVE, HRF and ISBI2012-EM), and the test split is
the rest. --check loads every dataset with benchmark.registry.load_split and compares the fingerprint of its test
split (benchmark.run.split_fingerprint) with the one in the paper's score records, and the fingerprint of its pool
with the pool of the data the paper used.

Needs numpy, Pillow, tifffile and requests; MoNuSeg also needs pyarrow (pip install pyarrow). The converters of DRIVE,
ISBI2012-EM and MoNuSeg follow scripts/prep_public_datasets.py of the research code (AutoSeg).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import os
import shutil
import sys
import zipfile

import numpy as np
import tifffile
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark import datasets as D                       # noqa: E402
from benchmark.registry import ALL_DATASETS, PANEL        # noqa: E402

# The fingerprints of the paper's splits. TEST_FP: the `split_fp` of every score record of the paper
# (results/final10/ridge4_k8/ridge__<dataset>.json; tests/test_prepare_data.py checks they agree). POOL_FP: the same
# digest over the pool, from data prepared by this script whose file names equal those of the data tree the paper's
# runs read.
TEST_FP = {
    "spheroidj": "0221b7005adbfcae", "rozpad": "8017232bf8a5d616", "dsb2018": "6eed932d47871100",
    "monuseg": "8752244ba838397d", "bbbc010": "8c4b1bdf3ddb991a",
    "bacteria": "4200850b1b81f401", "drive": "5513231d22432d80", "hrf": "cbb4f13a09ada9f9",
    "isbi2012em": "274c5d1057f9faf2", "fisbe": "b33e8ce39b94efda",
}
POOL_FP = {
    "spheroidj": "40714537854d76ff", "rozpad": "caff16786c6721a6", "dsb2018": "33db9075dfec9d83",
    "monuseg": "64c39b4a9e450e5a", "bbbc010": "7cced9e1ca4bfa6e",
    "bacteria": "cb7c8257f3ccfb3e", "drive": "ae8d7835202abace", "hrf": "d7f62491035ca135",
    "isbi2012em": "76c4a6b4f37e5b86", "fisbe": "05c3371a45190fcc",
}

# --- sources, pinned to the revision the paper's data came from where the host allows it -------------------------
DTS = ("https://raw.githubusercontent.com/WaterKnight1998/Deep-Tumour-Spheroid/"
       "680de6e31431d9ee130ba24705ef9d70b123698b/datasets")
HYPERBANK = ("https://raw.githubusercontent.com/michalprusek/hyperbank/"
             "12049329348b06c2bb0ed6eefda2b656ea688d2c/datasets")
DECAY_SHA256 = "c710b700adda4c2c56b6aec83c11593748addbd91b99ef1147d788a7692ab13b"   # datasets/README.md there
MONUSEG = ("https://huggingface.co/datasets/RationAI/MoNuSeg/resolve/4812ea499ff8f6fad2df7af0c9aa459773e22459/"
           "data/{split}-00000-of-00001.parquet")
BBBC010 = "https://data.broadinstitute.org/bbbc/BBBC010"
BACT_PHASE = "https://files.osf.io/v1/resources/xmury/providers/osfstorage/62f5813e0beb5f0558b0ba6c/?zip="
DRIVE = ("https://huggingface.co/datasets/Zomba/DRIVE-digital-retinal-images-for-vessel-extraction/resolve/"
         "e43e3f8a4b1146db231648786912d0ac882067b2")
ISBI = "https://raw.githubusercontent.com/alexklibisz/isbi-2012/238a567ae843f7c03e5cc5d3f233cd3f93f0da17/data"
FISBE_MIPS = "https://zenodo.org/records/10875063/files/fisbe_v1.0_mips.zip?download=1"
FISBE_MIPS_SHA256 = "63e0ab093b47a0040bfdd9eacb0cda394cc762c13be1743f12dcec3672321678"

# --- the custom splits, as recorded with the data the paper used -------------------------------------------------
# SpheroidJ: 72 images of the Deep-Tumour-Spheroid repository (datasets/Images/test/<n>.png with the mask
# datasets/Labels/test/<n>.tif), named <microscope>_<n>; drawn with seed 42, 48 support and 24 test images.
SPHEROIDJ = {
    "support": """BL5S_280 BL5S_294 BL5S_315 BL5S_316 BL5S_321 BL5S_325 BL5S_335 BL5S_339 BL5S_345 BL5S_346 BL5S_347
        BL5S_355 BN2S_121 BN2S_122 BN2S_13 BN2S_133 BN2S_139 BN2S_140 BN2S_141 BN2S_166 BN2S_169 BN2S_176 BN2S_179
        BN2S_186 BN2S_19 BN2S_207 BN2S_227 BN2S_238 BN2S_248 BN2S_25 BN2S_254 BN2S_262 BN2S_263 BN2S_267 BN2S_27
        BN2S_270 BN2S_28 BN2S_29 BN2S_31 BN2S_4 BN2S_44 BN2S_56 BN2S_59 BN2S_61 BN2S_62 BN2S_80 BN2S_86
        BN2S_90""".split(),
    "test": """BL5S_298 BL5S_305 BL5S_309 BL5S_312 BL5S_338 BL5S_342 BN2S_119 BN2S_147 BN2S_159 BN2S_172 BN2S_18
        BN2S_180 BN2S_189 BN2S_192 BN2S_198 BN2S_24 BN2S_253 BN2S_259 BN2S_268 BN2S_269 BN2S_49 BN2S_53 BN2S_63
        BN2S_77""".split(),
}
# FISBe: the 30 completely labelled images. Test: the 7 of the official test split and 7 of the official training
# split; pool: the 5 of the official validation split and the other 11 of the official training split.
FISBE = {
    "train": """R22C03-20180918_66_J2 VT008647-20171222_63_D1 VT008647-20171222_63_D2 VT008647-20171222_63_E1
        VT012403-20171128_61_B2 VT019303-20171013_65_B6 VT019307-20171013_65_F1 VT033051-20171128_61_E2
        VT033051-20171128_61_E4 VT033614-20171124_64_H1 VT033614-20171124_64_H5 VT040433-20170919_63_D6
        VT041298-20171114_63_C3 VT047848-20171020_66_I2 VT047848-20171020_66_I3 VT047848-20171020_66_I5""".split(),
    "test": """JRC_SS04989-20160318_24_A2 R14A02-20180905_65_A6 R38F04-20181005_63_G3 R38F04-20181005_63_G5
        R38F04-20181005_63_H1 R53A10-20181019_64_A4 R54A09-20181019_64_H1 R75E01-20181030_64_D1
        VT011145-20171222_63_I1 VT027175-20171031_62_H3 VT027175-20171031_62_H4 VT047848-20171020_66_J2
        VT050157-20171110_61_C1 VT061467-20180911_62_E5""".split(),
}
BBBC010_POOL = 40          # the first 40 wells by name (A01-A24, B01-B16) are the pool, the other 60 the test split

DOWNLOADS = os.path.join(os.environ.get("ASG_DATA_ROOT", "data"), "_downloads")


# --- helpers -----------------------------------------------------------------------------------------------------
def fetch(url: str, name: str, sha256: str | None = None) -> str:
    """Download `url` to $DOWNLOADS/name once (atomically: an interrupted download leaves only a .part file)."""
    import requests

    dest = os.path.join(DOWNLOADS, name)
    if not (os.path.exists(dest) and os.path.getsize(dest) > 0):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"  downloading {url}", flush=True)
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(dest + ".part", "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        with open(dest + ".part", "rb") as f:
            if f.read(64).lstrip()[:1] == b"<":          # an HTML error page served with status 200
                raise RuntimeError(f"{url} returned an HTML page, not the file")
        os.replace(dest + ".part", dest)
    if sha256 is not None and sha256_of(dest) != sha256:
        print(f"  WARNING: {name} has another SHA-256 than the file the paper's data came from; "
              f"python -m benchmark.prepare_data --check tells whether the split is still the same", flush=True)
    return dest


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_member(z: zipfile.ZipFile, name: str) -> np.ndarray:
    data = z.read(name)
    if name.lower().endswith((".tif", ".tiff")):
        try:
            return tifffile.imread(io.BytesIO(data))
        except Exception:                               # e.g. LZW without imagecodecs; Pillow decodes it
            pass
    return np.asarray(Image.open(io.BytesIO(data)))


def mkdirs(root: str, *subs: str) -> None:
    for s in subs:
        os.makedirs(os.path.join(root, s), exist_ok=True)


def save_pair(root: str, split: str, stem: str, image: np.ndarray, label: np.ndarray) -> None:
    """An instance dataset's image and label map as uncompressed TIFF (tifffile reads them without imagecodecs)."""
    tifffile.imwrite(os.path.join(root, split, "images", stem + ".tif"), image)
    tifffile.imwrite(os.path.join(root, split, "masks", stem + ".tif"), label)


def colour_labels(rgb: np.ndarray) -> np.ndarray:
    """A colour-coded label image to a label map: label = index of the colour in np.unique's (sorted) order, so black
    is 0 and the colours count up from 1. The numbering is part of the fingerprint."""
    colours, inverse = np.unique(rgb.reshape(-1, rgb.shape[-1]), axis=0, return_inverse=True)
    if colours[0].any():
        raise ValueError("colour-coded labels without a black background")
    return inverse.reshape(rgb.shape[:2]).astype(np.uint16)


def round_robin(by_group: dict[str, list[str]]) -> list[tuple[str, str]]:
    """(group, item) taking the i-th item of every group (groups and items sorted) before any (i+1)-th."""
    groups = sorted(by_group)
    items = {g: sorted(by_group[g]) for g in groups}
    out, i = [], 0
    while any(i < len(items[g]) for g in groups):
        out += [(g, items[g][i]) for g in groups if i < len(items[g])]
        i += 1
    return out


# --- the converters: each writes the complete dataset into `out` --------------------------------------------------
def prep_spheroidj(out: str) -> None:
    for split, names in SPHEROIDJ.items():
        mkdirs(out, f"{split}/images", f"{split}/masks")
        for name in names:
            n = name.split("_")[1]
            img = fetch(f"{DTS}/Images/test/{n}.png", f"spheroidj/{n}.png")
            lbl = fetch(f"{DTS}/Labels/test/{n}.tif", f"spheroidj/{n}.tif")
            shutil.copyfile(img, os.path.join(out, split, "images", name + ".png"))     # the image unchanged
            m = np.asarray(Image.open(lbl))
            m = m[..., 0] if m.ndim == 3 else m
            Image.fromarray(((m > 0) * 255).astype(np.uint8)).save(os.path.join(out, split, "masks", name + ".png"))


def prep_rozpad(out: str) -> None:
    parts = [fetch(f"{HYPERBANK}/decay_dataset.zip.part-{p}", f"rozpad/decay_dataset.zip.part-{p}")
             for p in ("aa", "ab")]
    whole = os.path.join(DOWNLOADS, "rozpad", "decay_dataset.zip")
    if not os.path.exists(whole):
        with open(whole + ".part", "wb") as f:
            for p in parts:
                with open(p, "rb") as g:
                    shutil.copyfileobj(g, f)
        os.replace(whole + ".part", whole)
    if sha256_of(whole) != DECAY_SHA256:
        raise RuntimeError(f"{whole}: SHA-256 differs from the one the HyperBank repository publishes")
    with zipfile.ZipFile(whole) as z:
        for m in z.namelist():
            if m.startswith("decay/") and not m.endswith("/"):
                dest = os.path.join(out, m[len("decay/"):])
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "wb") as f:
                    f.write(z.read(m))


def prep_dsb2018(out: str) -> None:
    D.download_dsb2018(out)                              # StarDist's release, the same call the loader makes on a miss


def prep_monuseg(out: str) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit("MoNuSeg is distributed as parquet files: pip install pyarrow") from e
    for split in ("train", "test"):
        path = fetch(MONUSEG.format(split=split), f"monuseg/{split}.parquet")
        mkdirs(out, f"{split}/images", f"{split}/masks")
        d = pq.read_table(path).to_pydict()
        seen = set()
        for row, (pid, img, insts) in enumerate(zip(d["patient"], d["image"], d["instances"])):
            if pid in seen:
                raise ValueError(f"monuseg/{split}: image {pid} appears twice")
            seen.add(pid)
            im = np.asarray(Image.open(io.BytesIO(img["bytes"] if isinstance(img, dict) else img)))
            if im.ndim == 2:
                im = np.stack([im] * 3, -1)
            im = np.ascontiguousarray(im[..., :3]).astype(np.uint8)
            if not insts:
                raise ValueError(f"monuseg/{split}: row {row} ({pid}) has no nuclei")
            lbl = np.zeros(im.shape[:2], np.uint16)
            for k, inst in enumerate(insts, 1):      # one mask per nucleus, numbered in the file's order
                b = inst["bytes"] if isinstance(inst, dict) else inst
                lbl[np.asarray(Image.open(io.BytesIO(b))) > 0] = k
            save_pair(out, split, pid, im, lbl)


def prep_bbbc010(out: str) -> None:
    zi = zipfile.ZipFile(fetch(f"{BBBC010}/BBBC010_v2_images.zip", "bbbc010/BBBC010_v2_images.zip"))
    zw = zipfile.ZipFile(fetch(f"{BBBC010}/BBBC010_v1_foreground_eachworm.zip",
                               "bbbc010/BBBC010_v1_foreground_eachworm.zip"))
    w1 = {}                                              # well -> the w1 image (the channel the annotation fits)
    for n in zi.namelist():
        base = os.path.basename(n)
        if not n.startswith("__MACOSX") and "_w1_" in base and base.endswith(".tif"):
            w1[base.split("_w1_")[0].rsplit("_", 1)[1]] = n
    worms = {}                                           # well -> its worms' masks, by file name
    for n in sorted(zw.namelist()):
        base = os.path.basename(n)
        if not n.startswith("__MACOSX") and base.endswith("_ground_truth.png"):
            worms.setdefault(base[:3], []).append(n)
    wells = sorted(w1)
    if len(wells) != 100 or sorted(worms) != wells:
        raise RuntimeError(f"bbbc010: {len(wells)} wells with images, {len(worms)} with worm masks (expected 100)")
    for i, well in enumerate(wells):
        split = "train" if i < BBBC010_POOL else "test"
        mkdirs(out, f"{split}/images", f"{split}/masks")
        im = read_member(zi, w1[well])
        lbl = np.zeros(im.shape[:2], np.uint16)
        for k, n in enumerate(worms[well], 1):           # where worms overlap, the later worm keeps the pixel
            m = read_member(zw, n)
            lbl[(m[..., 0] if m.ndim == 3 else m) > 0] = k
        save_pair(out, split, well, im, lbl)


def prep_bacteria(out: str) -> None:
    with zipfile.ZipFile(fetch(BACT_PHASE, "bacteria/bact_phase.zip")) as z:
        files = {"train": {}, "test": {}}               # split -> species -> image stems
        for n in z.namelist():
            parts = n.split("/")
            if len(parts) == 3 and parts[0] in ("train_sorted", "test_sorted") and parts[2].endswith(".tif") \
                    and not parts[2].endswith(("_masks.tif", "_flows.tif")):
                files[parts[0][:-len("_sorted")]].setdefault(parts[1], []).append(parts[2][:-len(".tif")])

        def pair(split, species, stem):
            d = f"{split}_sorted/{species}/{stem}"
            return read_member(z, d + ".tif"), read_member(z, d + "_masks.tif").astype(np.uint16)

        mkdirs(out, "train/images", "train/masks", "test/images", "test/masks")
        # The pool is renamed round-robin by species, so that the first 20 by name span all 17 species.
        for i, (species, stem) in enumerate(round_robin(files["train"])):
            save_pair(out, "train", f"{i:03d}__{species}__{stem}", *pair("train", species, stem))
        for species, stems in files["test"].items():
            for stem in stems:
                save_pair(out, "test", f"{species}__{stem}", *pair("test", species, stem))
    if (len(glob.glob(f"{out}/train/images/*.tif")), len(glob.glob(f"{out}/test/images/*.tif"))) != (249, 148):
        raise RuntimeError("bacteria: expected 249 pool and 148 test images")


def prep_drive(out: str) -> None:
    mkdirs(out, "images", "masks")
    jobs = [(f"train/input/{i}.tif", f"train/label/{i}.png", f"{i:02d}") for i in range(21, 41)]     # official training
    jobs += [(f"val/input/{i:02d}.tif", f"val/label/{i:02d}_manual1.png", f"{i:02d}") for i in range(1, 21)]  # test
    for img, lbl, stem in jobs:
        shutil.copyfile(fetch(f"{DRIVE}/{img}", f"drive/{img}"), os.path.join(out, "images", stem + ".tif"))
        shutil.copyfile(fetch(f"{DRIVE}/{lbl}", f"drive/{lbl}"), os.path.join(out, "masks", stem + ".png"))


def prep_hrf(out: str) -> None:
    with zipfile.ZipFile(fetch(PANEL["hrf"].url, "hrf/all.zip")) as z:
        z.extractall(out)                                # images/, manual1/ (and mask/, which is not used)


def prep_isbi2012em(out: str) -> None:
    vol = tifffile.imread(fetch(f"{ISBI}/train-volume.tif", "isbi2012em/train-volume.tif"))
    lbl = tifffile.imread(fetch(f"{ISBI}/train-labels.tif", "isbi2012em/train-labels.tif"))
    if vol.shape != lbl.shape:
        raise RuntimeError(f"isbi2012em: volume {vol.shape} and labels {lbl.shape} differ")
    mkdirs(out, "images", "masks")
    for s in range(vol.shape[0]):
        Image.fromarray(vol[s]).save(os.path.join(out, "images", f"{s:02d}.png"))
        membrane = (lbl[s] == 0).astype(np.uint8) * 255    # foreground = the membrane, label 0 of the original
        Image.fromarray(membrane).save(os.path.join(out, "masks", f"{s:02d}.png"))
    frac = float((lbl == 0).mean())
    if not 0.1 < frac < 0.4:                             # the membrane is the thin minority (about 22 %)
        raise RuntimeError(f"isbi2012em: membrane fraction {frac:.2f}; is the label convention inverted?")


def prep_fisbe(out: str) -> None:
    with zipfile.ZipFile(fetch(FISBE_MIPS, "fisbe/fisbe_v1.0_mips.zip", FISBE_MIPS_SHA256)) as z:
        mips = {os.path.basename(n)[:-len(".png")]: n for n in z.namelist()
                if n.startswith("mips/completely/") and n.endswith(".png")}
        for split, names in FISBE.items():
            mkdirs(out, f"{split}/images", f"{split}/masks")
            for name in names:
                # Each projection is two panels side by side: the image left, the colour-coded neurons right.
                rgb = np.asarray(Image.open(io.BytesIO(z.read(mips[name]))).convert("RGB"))
                w = rgb.shape[1] // 2
                save_pair(out, split, name, np.ascontiguousarray(rgb[:, :w]), colour_labels(rgb[:, w:]))


PREP = {"spheroidj": prep_spheroidj, "rozpad": prep_rozpad, "dsb2018": prep_dsb2018, "monuseg": prep_monuseg,
        "bbbc010": prep_bbbc010, "bacteria": prep_bacteria, "drive": prep_drive,
        "hrf": prep_hrf, "isbi2012em": prep_isbi2012em, "fisbe": prep_fisbe}


def prepare(name: str, force: bool) -> None:
    out = PANEL[name].root
    if os.path.isdir(out) and os.listdir(out):
        if not force:
            print(f"[{name}] {out} exists, left alone (--force rebuilds it)", flush=True)
            return
        shutil.rmtree(out)
    print(f"[{name}] -> {out}", flush=True)
    tmp = out.rstrip("/") + ".partial"                   # written aside and moved in whole, never half a dataset
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    PREP[name](tmp)
    os.replace(tmp, out)


def check(names: list[str]) -> bool:
    from benchmark.registry import load_split
    from benchmark.run import split_fingerprint

    ok = True
    print(f"{'dataset':11s} {'pool':>4s} {'test':>4s}  {'test fingerprint':16s}  {'pool fingerprint':16s}  result")
    for name in names:
        try:
            pool, test = load_split(name)
        except Exception as e:                           # missing or incomplete data: report it, check the rest
            print(f"{name:11s}    -    -  {'-':16s}  {'-':16s}  NOT PREPARED ({type(e).__name__}: {e})")
            ok = False
            continue
        ft, fp = split_fingerprint(test), split_fingerprint(pool)
        good = ft == TEST_FP[name] and fp == POOL_FP[name]
        ok &= good
        what = "ok" if good else ("TEST SPLIT DIFFERS" if ft != TEST_FP[name] else "POOL DIFFERS")
        print(f"{name:11s} {len(pool):4d} {len(test):4d}  {ft:16s}  {fp:16s}  {what}", flush=True)
        del pool, test
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("datasets", nargs="*", default=["all"], help=f"'all' or any of: {' '.join(ALL_DATASETS)}")
    ap.add_argument("--check", action="store_true", help="compare the prepared splits with the paper's")
    ap.add_argument("--force", action="store_true", help="rebuild a dataset whose folder exists")
    a = ap.parse_args(argv)
    names = ALL_DATASETS if "all" in a.datasets else a.datasets
    unknown = [n for n in names if n not in PANEL]
    if unknown:
        ap.error(f"unknown dataset(s) {unknown}; valid: all {' '.join(ALL_DATASETS)}")
    if a.check:
        return 0 if check(names) else 1
    for name in names:
        prepare(name, a.force)
    print("done; check the splits with: python -m benchmark.prepare_data --check", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
