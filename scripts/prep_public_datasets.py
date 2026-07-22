#!/usr/bin/env python
"""Prepare 4 standard public biomedical few-shot benchmark datasets for the AutoSeg panel.

Run ON tulen (has disk + network); writes into ``$PANEL_DL_ROOT`` (default
``/disk1/prusek/panel_datasets``), the same root ``active_segmenter/eval/registry.py`` reads:

    PANEL_DL_ROOT=/disk1/prusek/panel_datasets \
        /home/prusek/dinov3_env/bin/python scripts/prep_public_datasets.py all

Canonical layouts produced (consumed by the registry kinds noted):
  drive/       images/NN.tif  masks/NN.png                    download kind, metric=cldice   binary vessel fg
  isbi2012em/  images/NN.png  masks/NN.png                    download kind, metric=cldice   binary membrane fg (=label 0)
  monuseg/     {train,test}/images/<id>.tif masks/<id>.tif    instance kind, metric=instance_ap  uint16 label map
  ctc_u373/    {train,test}/images/<id>.tif masks/<id>.tif    instance kind, metric=instance_ap  uint16 label map

Sources (all freely downloadable, NO registration):
  DRIVE     HF  Zomba/DRIVE-digital-retinal-images-for-vessel-extraction  (40 fundus imgs + 1st-manual vessel masks)
  ISBI2012  GH  alexklibisz/isbi-2012  train-volume.tif + train-labels.tif (30 EM slices, 512x512)
  MoNuSeg   HF  RationAI/MoNuSeg  parquet (train 37 / test 14) with per-nucleus instance masks
  CTC U373  celltrackingchallenge.net  PhC-C2DH-U373.zip  (gold SEG: seq01=15, seq02=19 frames)

CTC/MoNuSeg tifs are re-saved UNCOMPRESSED so registry load (tifffile.imread) needs no
imagecodecs (the CTC source tifs are LZW-compressed which tifffile cannot decode alone).
"""
from __future__ import annotations

import glob
import io
import os
import shutil
import sys
import zipfile

import numpy as np
import tifffile
from PIL import Image

ROOT = os.environ.get("PANEL_DL_ROOT", "/disk1/prusek/panel_datasets")
STAGE = os.path.join(ROOT, "_staging")

DRIVE_BASE = ("https://huggingface.co/datasets/"
              "Zomba/DRIVE-digital-retinal-images-for-vessel-extraction/resolve/main")
ISBI_BASE = "https://raw.githubusercontent.com/alexklibisz/isbi-2012/master/data"
MONUSEG = {
    "train": "https://huggingface.co/datasets/RationAI/MoNuSeg/resolve/main/data/train-00000-of-00001.parquet",
    "test": "https://huggingface.co/datasets/RationAI/MoNuSeg/resolve/main/data/test-00000-of-00001.parquet",
}
U373_URL = "https://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip"


def _get(url: str, timeout: int = 600) -> bytes:
    import requests
    last = None
    for _ in range(3):                              # retry only transient network errors
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            b = r.content
            if b[:64].lstrip()[:1] == b"<":         # HTML error/LFS-redirect body served as HTTP 200
                raise ValueError(f"got an HTML/redirect body (not binary) from {url}: {b[:60]!r}")
            return b
        except requests.RequestException as e:
            last = e
    raise RuntimeError(f"download failed after 3 tries: {url}") from last


def _fetch(url: str, dest: str) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"                            # atomic: a killed download leaves .part, never a
    with open(tmp, "wb") as f:                      # truncated `dest` that a later run trusts as complete
        f.write(_get(url))
    os.replace(tmp, dest)
    return dest


def _reset(d: str) -> None:
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)


def _decode(x) -> np.ndarray:
    b = x["bytes"] if isinstance(x, dict) else x
    return np.asarray(Image.open(io.BytesIO(b)))


def prep_drive() -> None:
    out = os.path.join(ROOT, "drive")
    imgd, mskd = os.path.join(out, "images"), os.path.join(out, "masks")
    _reset(imgd)
    _reset(mskd)
    jobs = []  # (image_url, mask_url, stem)
    for i in range(21, 41):  # HF "train" split ids -> label is NN.png
        jobs.append((f"{DRIVE_BASE}/train/input/{i}.tif", f"{DRIVE_BASE}/train/label/{i}.png", f"{i:02d}"))
    for i in range(1, 21):   # HF "val" split ids -> label is NN_manual1.png
        jobs.append((f"{DRIVE_BASE}/val/input/{i:02d}.tif", f"{DRIVE_BASE}/val/label/{i:02d}_manual1.png", f"{i:02d}"))
    n = 0
    for iurl, murl, stem in jobs:
        with open(os.path.join(imgd, f"{stem}.tif"), "wb") as f:
            f.write(_get(iurl, timeout=300))
        with open(os.path.join(mskd, f"{stem}.png"), "wb") as f:
            f.write(_get(murl, timeout=300))
        n += 1
    assert n == len(jobs) == 40, f"[drive] expected 40 pairs, wrote {n}"
    print(f"[drive] wrote {n} image/mask pairs -> {out}")


def prep_isbi() -> None:
    vol = tifffile.imread(_fetch(f"{ISBI_BASE}/train-volume.tif", os.path.join(STAGE, "isbi_volume.tif")))
    lbl = tifffile.imread(_fetch(f"{ISBI_BASE}/train-labels.tif", os.path.join(STAGE, "isbi_labels.tif")))
    out = os.path.join(ROOT, "isbi2012em")
    imgd, mskd = os.path.join(out, "images"), os.path.join(out, "masks")
    _reset(imgd)
    _reset(mskd)
    # ISBI2012 GT convention: 0 = membrane (~22% of pixels), 255 = cell interior.
    # Foreground = membrane (the thin tubular structure) = (label == 0).
    assert vol.shape == lbl.shape, f"[isbi] volume/label shape mismatch {vol.shape} vs {lbl.shape}"
    fracs = []
    for s in range(vol.shape[0]):
        Image.fromarray(vol[s]).save(os.path.join(imgd, f"{s:02d}.png"))
        mem = ((lbl[s] == 0).astype(np.uint8) * 255)
        fracs.append(float((mem > 0).mean()))
        Image.fromarray(mem).save(os.path.join(mskd, f"{s:02d}.png"))
    frac = float(np.mean(fracs))
    # membrane is the thin minority (~22%); if the mirror stores the complemented convention this catches
    # the silent inversion (else every isbi cldice number would be scored against an ~78%-fg blob).
    assert 0.1 < frac < 0.4, f"[isbi] membrane fg fraction {frac:.2f} out of [0.1,0.4] — GT convention flipped?"
    print(f"[isbi2012em] wrote {vol.shape[0]} slices (membrane fg = label 0, fg~{frac:.2f}) -> {out}")


def prep_monuseg() -> None:
    import pyarrow.parquet as pq
    out = os.path.join(ROOT, "monuseg")
    for split, url in MONUSEG.items():
        pth = _fetch(url, os.path.join(STAGE, f"monuseg_{split}.parquet"))
        imgd = os.path.join(out, split, "images")
        mskd = os.path.join(out, split, "masks")
        _reset(imgd)
        _reset(mskd)
        d = pq.read_table(pth).to_pydict()
        n = 0
        for row, (pid, img, insts) in enumerate(zip(d["patient"], d["image"], d["instances"])):
            im = _decode(img)
            if im.ndim == 2:
                im = np.stack([im] * 3, -1)
            im = np.ascontiguousarray(im[..., :3]).astype(np.uint8)
            if not insts:                          # H2: a null/empty row would write a BLANK GT — fail loud
                raise ValueError(f"[monuseg/{split}] row {row} ({pid}) has no instances")
            lbl = np.zeros(im.shape[:2], np.uint16)
            for k, inst in enumerate(insts, 1):
                lbl[_decode(inst) > 0] = k
            assert lbl.max() > 0, f"[monuseg/{split}] row {row} ({pid}) rasterised to a blank map"
            stem = f"{pid}_{row}"                   # H3: patient id can repeat across rows → unique per row
            tifffile.imwrite(os.path.join(imgd, f"{stem}.tif"), im)
            tifffile.imwrite(os.path.join(mskd, f"{stem}.tif"), lbl)
            n += 1
        got = len(glob.glob(os.path.join(imgd, "*.tif")))
        assert n > 0 and got == n, f"[monuseg/{split}] wrote {n} but {got} files on disk (id collision?)"
        print(f"[monuseg/{split}] wrote {n} images (uint16 per-nucleus instance maps)")


def prep_ctc_u373() -> None:
    zp = _fetch(U373_URL, os.path.join(STAGE, "u373.zip"))
    exdir = os.path.join(STAGE, "_u")
    base = os.path.join(exdir, "PhC-C2DH-U373")
    done = os.path.join(exdir, ".u373_extracted")          # C2: "dir exists" != "extraction finished"
    if not os.path.exists(done):
        if os.path.isdir(base):
            shutil.rmtree(base)                            # wipe a partial prior extraction
        with zipfile.ZipFile(zp) as z:
            z.extractall(exdir)
        open(done, "w").close()
    out = os.path.join(ROOT, "ctc_u373")
    # seq01 -> train, seq02 -> test : disjoint movies (no train/test leakage).
    for seq, split in (("01", "train"), ("02", "test")):
        imgd = os.path.join(out, split, "images")
        mskd = os.path.join(out, split, "masks")
        _reset(imgd)
        _reset(mskd)
        segs = sorted(glob.glob(os.path.join(base, f"{seq}_GT/SEG/man_seg*.tif")))
        n = 0
        for sp in segs:
            num = os.path.basename(sp)[len("man_seg"):-len(".tif")]  # e.g. "001"
            ip = os.path.join(base, seq, f"t{num}.tif")
            if not os.path.exists(ip):                     # C3: a SEG frame with no image is a real
                raise FileNotFoundError(f"[ctc/{seq}] SEG {sp} has no matching image {ip} "
                                        f"(padding mismatch / incomplete extract)")
            im = np.asarray(Image.open(ip))                  # PIL decodes LZW; re-save uncompressed
            sg = np.asarray(Image.open(sp)).astype(np.uint16)
            stem = f"{seq}_t{num}"
            tifffile.imwrite(os.path.join(imgd, f"{stem}.tif"), im)
            tifffile.imwrite(os.path.join(mskd, f"{stem}.tif"), sg)
            n += 1
        assert n == len(segs) and n > 0, f"[ctc/{seq}] matched {n}/{len(segs)} SEG frames"
        print(f"[ctc_u373/{split}] seq{seq}: wrote {n} frames (uint16 per-cell instance maps)")


# keys match the registry dataset names; "isbi" kept as an alias for convenience
FNS = {"drive": prep_drive, "isbi2012em": prep_isbi, "isbi": prep_isbi,
       "monuseg": prep_monuseg, "ctc_u373": prep_ctc_u373}


def main() -> None:
    targets = sys.argv[1:] or ["all"]
    if "all" in targets:
        targets = ["drive", "isbi2012em", "monuseg", "ctc_u373"]
    unknown = [t for t in targets if t not in FNS]
    if unknown:
        raise SystemExit(f"unknown target(s) {unknown}; valid: {sorted(set(FNS) | {'all'})}")
    for t in targets:
        FNS[t]()


if __name__ == "__main__":
    main()
