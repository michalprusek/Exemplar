# Copied from AutoSeg scripts/render_qual_grid.py (4cd0cd9) by tools/sync_paper.py; default paths adapted, internal notes rewritten in comments and docstrings.
"""Fig. 3: Exemplar's predictions on eight datasets (2 x 4 panels), as red contours on the grey image; only the model's
segmentation is marked.

Each panel is one test image of the qualitative dump (closed_form_bench.py --dump_qual: seed 0, K=8, the reported
readout). The crop is fixed per dataset. The crop is resampled to OUT_PX pixels before the contour is traced, so the
contour keeps a visible width at print size; the contour is the boundary of the predicted foreground at that size,
thickened to two pixels.

  python paper/render_qual_grid.py <dir with {ds}_{i}.npz> paper/out/figures/qual3
"""
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

SRC, OUT = sys.argv[1], sys.argv[2]
OUT_PX = 420
RED = np.array([230, 25, 25])
# dataset -> (dump stem, crop (y0, x0, side) or None = centred square)
PANELS = {
    "rozpad": ("rozpad_0", (515, 433, 1100)),       # the decaying spheroid and its fragments
    "spheroidj": ("spheroidj_0", (283, 223, 560)),  # the spheroid
    "dsb2018": ("dsb2018_2", None),
    "monuseg": ("monuseg_1", (0, 0, 500)),          # a quarter of the field, so single nuclei stay visible
    "ctc_u373": ("ctc_u373_1", None),
    "drive": ("drive_2", None),
    # ISBI2012-EM and FISBE: the dumped test image (of the first three) whose
    # seed-0 score is closest to the dataset's seed-0 median, so a typical case, not the best one
    "isbi2012em": ("isbi2012em_2", None),           # 0.916 against the median 0.929
    "fisbe": ("fisbe_1", (132, 133, 420)),          # 0.853 against 0.840; the centre at native scale, filaments stay visible
    # BBBC010 takes the panel of CTC-U373, which the paper does not report; chosen by the same rule (the dump is
    # bit-identical to ridge4_k8's seed 0): 0.770 against the median 0.696
    "bbbc010": ("bbbc010_2", None),
}
# ASG_DROP_DS=ctc_u373: a dataset that left the paper is not rendered (its image must not be shown)
_DROP = {x for x in os.environ.get("ASG_DROP_DS", "").split(",") if x}
PANELS = {ds: v for ds, v in PANELS.items() if ds not in _DROP}

os.makedirs(OUT, exist_ok=True)
for ds, (stem, crop) in PANELS.items():
    d = np.load(os.path.join(SRC, stem + ".npz"), allow_pickle=True)
    if str(d["ds"]) != ds or int(d["k"]) != 8 or int(d["seed"]) != 0:
        sys.exit(f"{stem}: expected {ds} at K=8 seed 0, got {d['ds']} K={d['k']} seed {d['seed']}")
    im = np.asarray(d["image"]).astype(np.float32)
    grey = im.mean(2) if im.ndim == 3 else im
    pr = np.asarray(d["pred_fg"]) > 0
    if crop is None:
        side = min(grey.shape)
        y0, x0 = (grey.shape[0] - side) // 2, (grey.shape[1] - side) // 2
    else:
        y0, x0, side = crop
    g, p = grey[y0:y0 + side, x0:x0 + side], pr[y0:y0 + side, x0:x0 + side]
    if g.shape != (side, side):
        sys.exit(f"{ds}: crop {g.shape} is not {side}x{side}")
    lo, hi = np.percentile(g, (0.5, 99.5))
    g8 = (np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    g8 = np.asarray(Image.fromarray(g8).resize((OUT_PX, OUT_PX), Image.BILINEAR))
    pm = np.asarray(Image.fromarray(p.astype(np.uint8) * 255).resize((OUT_PX, OUT_PX), Image.BILINEAR)) >= 128
    edge = pm & ~ndimage.binary_erosion(pm)
    edge = ndimage.binary_dilation(edge)                 # two pixels wide at the output size
    rgb = np.repeat(g8[..., None], 3, axis=2)
    rgb[edge] = RED
    Image.fromarray(rgb).save(os.path.join(OUT, f"{ds}.png"))
    print(f"{ds}: {stem}, crop y0={y0} x0={x0} side={side} of {grey.shape} -> {OUT_PX} px; "
          f"predicted foreground {p.mean():.3f} of the crop")
