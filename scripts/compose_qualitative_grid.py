"""Compose the qualitative GRID figure (Fig. 2) from the ``.npz`` dumps written by
``scripts/dump_qualitative_grid.py``. No GPU needed — pure plotting, so the layout and styling
can be iterated freely.

Layout: pairs of [Ground truth | Ours] panels, ``PAIRS_PER_ROW`` pairs per row (2*PAIRS_PER_ROW image
panels), ``ceil(N/PAIRS_PER_ROW)`` rows, over N=6..8 examples from different datasets. The default
PAIRS_PER_ROW=3 gives a wide, short panel that fits a double-column float; QUAL_PAIRS_PER_ROW=2
restores the original tall layout. Each panel shows the
NATIVE-resolution input image with the segmentation as a translucent fill plus a crisp VECTOR
boundary contour (green = ground truth, red = our prediction). Saved as PDF: the microscopy images
are embedded at native resolution, while the contours, badges, and labels are vector.

Matplotlib embeds images with lossless Flate, so a native-resolution panel PDF is ~10 MB. Since the
crisp segmentation edges and labels are VECTOR (not part of the raster), the microscopy photos can be
safely JPEG-recompressed without touching them. The final figure is produced by piping this PDF
through Ghostscript, which keeps it crisp (500 dpi at the ~1.7 in panel is well above print) at ~2 MB:

    gs -q -o qualitative.pdf -sDEVICE=pdfwrite \
       -dDownsampleColorImages=true -dColorImageResolution=500 -dColorImageDownsampleType=/Bicubic \
       -dDownsampleGrayImages=true  -dGrayImageResolution=500  -dGrayImageDownsampleType=/Bicubic \
       -dAutoFilterColorImages=false -dColorImageFilter=/DCTEncode \
       -dAutoFilterGrayImages=false  -dGrayImageFilter=/DCTEncode -dJPEGQ=92 <native>.pdf
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DUMP = os.environ.get("QUAL_DUMP", os.path.join(os.path.dirname(__file__), "..", "qual_grid_dump"))
OUT = os.environ.get("QUAL_PDF",
                     os.path.join(os.path.dirname(__file__), "..", "paper", "isbi2027",
                                  "figures", "qualitative.pdf"))

# Ordered list of dumps to show (basename without .npz) and the display name for each pair.
# PAIRS_PER_ROW pairs per row; keep it 6 entries. Override with QUAL_SELECT="spheroidj_0,dsb2018_1,...".
# Best-IoU representative image per dataset; one example per dataset = a wide 2-row, six-morphology
# panel (three [GT|Ours] pairs per row) that fits as a double-column float. Decay replaces HRF
# (DRIVE already covers vessels). Override with QUAL_SELECT=...
SELECT = os.environ.get(
    "QUAL_SELECT",
    "spheroidj_0,dsb2018_2,monuseg_1,ctc_u373_1,drive_2,rozpad_0").split(",")
# pairs per row: 3 gives a wide, short 2-row panel (double-column figure*); 2 gives the old tall 3-row.
PAIRS_PER_ROW = int(os.environ.get("QUAL_PAIRS_PER_ROW", "3"))
DISPLAY = {"spheroidj": "Spheroids", "dsb2018": "Nuclei (DSB2018)", "monuseg": "H&E nuclei",
           "ctc_u373": "Phase-contrast cells", "drive": "Retinal vessels (DRIVE)",
           "hrf": "Retinal vessels (HRF)", "rozpad": "Decay spheroids"}
GT_COL = (0.15, 0.95, 0.25)
PR_COL = (1.0, 0.30, 0.12)
BG_DIM = 0.62                                                   # darken the image so overlays pop
# Images are embedded at native resolution; only a few pathologically large ones (HRF is 3504 px) are
# capped, because at the printed panel width (~1.7 in) 1600 px is already ~900 dpi — the full array is
# visually identical there but adds ~15 MB. Set QUAL_MAX_SIDE=0 to embed every image fully native.
MAX_SIDE = int(os.environ.get("QUAL_MAX_SIDE", "1600"))


def _cap(image: np.ndarray, gt: np.ndarray, pr: np.ndarray):
    h, w = image.shape[:2]
    s = max(h, w)
    if not MAX_SIDE or s <= MAX_SIDE:
        return image, gt, pr
    from skimage.transform import resize
    nh, nw = round(h * MAX_SIDE / s), round(w * MAX_SIDE / s)
    image = resize(image, (nh, nw), order=1, preserve_range=True, anti_aliasing=True)
    gt = resize(gt.astype(float), (nh, nw), order=0, preserve_range=True) > 0.5
    pr = resize(pr.astype(float), (nh, nw), order=0, preserve_range=True) > 0.5
    return image, gt, pr


def _disp(im: np.ndarray) -> np.ndarray:
    a = np.asarray(im, np.float32)
    a = a if a.ndim == 2 else a[..., :3]
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)          # robust contrast stretch
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def _panel(ax, img_disp, mask, colour, badge):
    # interpolation="none" makes the PDF backend embed the image at its NATIVE array resolution
    # (no resampling to the figure DPI); "nearest" would down-sample it to ~100 dpi and blur it.
    ax.imshow(img_disp * BG_DIM, cmap="gray" if img_disp.ndim == 2 else None,
              vmin=0, vmax=1, interpolation="none")
    m = np.asarray(mask) > 0
    if m.any():
        ov = np.zeros((*m.shape, 4), np.float32)
        ov[m] = (*colour, 0.30)                                  # translucent fill (flat colour)
        # The fill is a flat colour, so it may be down-sampled ("nearest") to keep the PDF small; the
        # crisp SEGMENTATION EDGE is drawn separately as a vector contour and the native-resolution
        # image underneath keeps interpolation="none". Only the base image needs to be native.
        ax.imshow(ov, interpolation="nearest")
        ax.contour(m.astype(float), levels=[0.5], colors=[colour], linewidths=0.8)  # vector edge
    ax.text(0.03, 0.97, badge, transform=ax.transAxes, va="top", ha="left", fontsize=6.5,
            color="white", bbox=dict(facecolor=colour, edgecolor="none", pad=1.2, alpha=0.92))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main() -> None:
    entries, missing = [], []
    for key in SELECT:
        p = os.path.join(DUMP, f"{key}.npz")
        if not os.path.exists(p):
            missing.append(key)
            continue
        entries.append((key, np.load(p, allow_pickle=True)))
    # FAIL LOUD on ANY missing dump, not just on all of them. With 5 of 6 present the grid composes into
    # a deliberate-looking layout with one panel silently absent, while the paper's caption still
    # describes every morphology -- a figure that quietly drops a dataset is worse than no figure.
    if missing:
        sys.exit(f"missing dumps for {missing} in {DUMP} — run dump_qualitative_grid.py (and pull the "
                 f".npz files) before composing; refusing to emit a figure with an absent panel")
    if not entries:
        sys.exit("no dumps found; run dump_qualitative_grid.py and pull the .npz files first")

    n = len(entries)
    ppr = PAIRS_PER_ROW
    nrows = (n + ppr - 1) // ppr
    # height per row scales with panel width; (2/ppr) reduces to the original 1.85/row at ppr=2.
    # A floor keeps single-row (ppr>=4) panels from collapsing below a legible height.
    row_h = max((2.0 / ppr) * 1.85, 0.95)
    fig, axes = plt.subplots(nrows, ppr * 2, figsize=(7.0, row_h * nrows), squeeze=False)
    for j, (key, d) in enumerate(entries):
        r, cpair = j // ppr, (j % ppr) * 2
        ds = str(d["ds"])
        image, gt, pr = _cap(d["image"], d["gt"] > 0, d["pred_fg"] > 0)
        img = _disp(image)
        _panel(axes[r][cpair], img, gt, GT_COL, "GT")
        _panel(axes[r][cpair + 1], img, pr, PR_COL, "Ours")
        # dataset name centred over the pair, anchored in the GT panel's axes fraction so it tracks
        # the final layout (x=1.0 is the shared GT/Ours edge = the pair centre); robust to bbox=tight.
        axes[r][cpair].text(1.0, 1.02, DISPLAY.get(ds, ds), transform=axes[r][cpair].transAxes,
                            ha="center", va="bottom", fontsize=8)
    # blank any unused trailing panels
    for j in range(n, nrows * ppr):
        r, cpair = j // ppr, (j % ppr) * 2
        axes[r][cpair].axis("off"); axes[r][cpair + 1].axis("off")

    fig.subplots_adjust(wspace=0.03, hspace=0.30, left=0.005, right=0.995, top=0.92, bottom=0.005)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
