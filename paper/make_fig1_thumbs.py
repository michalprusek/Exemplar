# Copied from AutoSeg scripts/make_fig1_thumbs.py (cb71f85) by tools/sync_paper.py; default paths adapted, internal notes rewritten in comments and docstrings.
"""Fig. 1's two thumbnails: a crop of one test image and Exemplar's predicted foreground on the same crop.

The image is a Decay test image, a decaying spheroid and its fragments, cropped closer than its Fig. 3 panel.

The prediction comes from a qualitative dump of the reported method (closed_form_bench.py --dump_qual, seed 0, K=8),
so the output thumbnail is a real prediction, never a drawing. Regenerate after the method's arms are re-run:
  python paper/make_fig1_thumbs.py results/c119/qual_v4/rozpad_0.npz paper/out/figures [y0,x0,side]
"""
import sys

import numpy as np
from PIL import Image

SRC, OUT = sys.argv[1], sys.argv[2]
# Decay: the core and the fragments around it, closer than Fig. 3's panel (which shows y0=515, x0=433, side 1100).
Y0, X0, SIDE = map(int, (sys.argv[3] if len(sys.argv) > 3 else "715,633,700").split(","))

d = np.load(SRC, allow_pickle=True)
if int(d["k"]) != 8:
    sys.exit(f"{SRC}: expected a dump at K=8, got K={d['k']}")
im = np.asarray(d["image"])[Y0:Y0 + SIDE, X0:X0 + SIDE]
if im.ndim == 3 and np.ptp(im.astype(int), axis=2).max() == 0:
    im = im[..., 0]                                       # a grey image stored in three equal channels
fg = np.asarray(d["pred_fg"])[Y0:Y0 + SIDE, X0:X0 + SIDE] > 0
if im.shape[:2] != (SIDE, SIDE) or not fg.any():
    sys.exit(f"{SRC}: crop {im.shape} or an empty prediction")
Image.fromarray(im.astype(np.uint8)).save(f"{OUT}/fig1_input.png")
mask = np.where(fg, 255, 34).astype(np.uint8)            # white foreground on near-black
Image.fromarray(mask).save(f"{OUT}/fig1_mask.png")
print(f"wrote {OUT}/fig1_input.png and {OUT}/fig1_mask.png from {SRC} ({d['method']}, seed {d['seed']}); "
      f"foreground {fg.mean():.3f} of the crop")
