"""Render the qualitative examples as INDIVIDUAL square panels (one PDF per GT/prediction), so the paper
can lay them out in LaTeX at a uniform size with the dataset name as a subcaption under each
[GT | Ours] pair. The panels are drawn WITHOUT a frame (axes spines are switched off) -- the frame in
the paper is LaTeX's \\fbox, so its weight stays adjustable in Overleaf without re-rendering.

What "vector, high resolution" does and does not mean here: the segmentation CONTOUR is a true vector
path, and the microscopy image is embedded at its native array resolution (interpolation="none", so the
PDF backend does not resample it) -- but the image itself and the translucent mask fill are raster, as
any photograph must be. Centre-cropping to a square is what makes every panel the same shape, so LaTeX
can give them all one width.

Output files are keyed on the DATASET (``{ds}_gt.pdf`` / ``{ds}_pred.pdf``) while QUAL_SELECT keys are
``{ds}_{index}`` -- selecting two indices of the same dataset therefore overwrites, last one wins.

No GPU: reads the .npz dumps written by dump_qualitative_grid.py.

  QUAL_DUMP=<dir with *.npz>  QUAL_SELECT=spheroidj_0,dsb2018_2,...  QUAL_PANEL_OUT=<out dir>
  python scripts/render_qual_panels.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DUMP = os.environ.get("QUAL_DUMP", "/disk1/prusek/qual_grid_dump")
OUT = os.environ.get("QUAL_PANEL_OUT", "/tmp/qual_panels")
SELECT = os.environ.get(
    "QUAL_SELECT", "spheroidj_0,dsb2018_2,monuseg_1,ctc_u373_1,drive_2,rozpad_0").split(",")
GT_COL = (0.15, 0.95, 0.25)
PR_COL = (1.0, 0.30, 0.12)
BG_DIM = 0.62                      # darken the image so the overlay pops


def disp(im):
    a = np.asarray(im, np.float32)
    a = a if a.ndim == 2 else a[..., :3]
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)      # robust contrast stretch
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def square(im, gt, pr):
    """Centre-crop to a square so every panel has the same aspect ratio (LaTeX then gives them the
    same display size)."""
    h, w = im.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    sl = (slice(y0, y0 + s), slice(x0, x0 + s))
    return im[sl], np.asarray(gt)[sl], np.asarray(pr)[sl]


def render(img_disp, mask, colour, path):
    fig, ax = plt.subplots(figsize=(2.0, 2.0), dpi=300)
    ax.imshow(img_disp * BG_DIM, cmap="gray" if img_disp.ndim == 2 else None,
              vmin=0, vmax=1, interpolation="none")            # native-res image, no resampling
    m = np.asarray(mask) > 0
    if m.any():
        ov = np.zeros((*m.shape, 4), np.float32)
        ov[m] = (*colour, 0.32)
        ax.imshow(ov, interpolation="nearest")
        ax.contour(m.astype(float), levels=[0.5], colors=[colour], linewidths=1.1)   # vector edge
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    # FAIL LOUD BEFORE writing anything. OUT is reused and never cleared, and LaTeX \includegraphics
    # fixed filenames, so a dump that has gone missing on a re-render would leave the PREVIOUS run's
    # panel in the figure -- a prediction from an older method version, indistinguishable from a
    # current one. Check every selection up front rather than skipping panels one at a time.
    missing = [k for k in SELECT if not os.path.exists(os.path.join(DUMP, f"{k}.npz"))]
    if missing:
        raise SystemExit(f"missing dumps for {missing} in {DUMP} — refusing to re-render a partial "
                         f"panel set (stale panels from a previous run would silently survive in {OUT})")
    for key in SELECT:
        p = os.path.join(DUMP, f"{key}.npz")
        d = np.load(p, allow_pickle=True)
        ds = str(d["ds"])
        img, gt, pr = square(d["image"], d["gt"] > 0, d["pred_fg"] > 0)
        imd = disp(img)
        render(imd, gt, GT_COL, os.path.join(OUT, f"{ds}_gt.pdf"))
        render(imd, pr, PR_COL, os.path.join(OUT, f"{ds}_pred.pdf"))
        print(f"wrote {ds}_gt.pdf, {ds}_pred.pdf")


if __name__ == "__main__":
    main()
