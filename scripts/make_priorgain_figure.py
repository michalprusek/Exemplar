#!/usr/bin/env python
"""What the classical prior bank is worth, against how far structures fall below the patch stride.

The x axis is the MEDIAN distance-transform radius of the support foreground, divided by the
encoder's effective patch stride in native pixels (16 * native / 672). It answers "is the structure
this dataset asks for finer than one patch of the backbone?".

WHY THE MEDIAN AND NOT THE MEAN. The method's own descriptor is the mean (head_fusion_backend.py:653)
and the mean fails here: rozpad is a decaying spheroid, a thick core plus fine debris, so its radii
are strongly right-skewed and its mean (29.5 px) is five times its median (5.7 px). Registry C79
records that the pre-specified test used the mean and did NOT reach significance (rho = -0.573,
p = 0.066), that the median gives rho = -0.764 (p = 0.006), and that the median is the third
statistic tried on eleven points and therefore EXPLORATORY. This figure is descriptive: it is drawn
without a fitted line and without a p-value on purpose.

  ASG_SEM_TREE=results/final10 ASG_FIG_OUT=paper/isbi2027/figures/priorgain.pdf python scripts/make_priorgain_figure.py
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42        # never Type 3: IEEE PDF eXpress rejects it
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

ROOT = os.environ.get("ASG_SEM_TREE", "results/final10")
OUT = os.environ.get("ASG_FIG_OUT", "paper/isbi2027/figures/priorgain.pdf")

# median DT radius / effective patch stride, measured on each dataset's support pool by
# scripts/measure_structure_width.py (committed alongside; needs the datasets, so it runs on tulen)
WIDTH = {"hrf": 0.034, "drive": 0.072, "fisbe": 0.087, "rozpad": 0.117, "bbbc010": 0.135,
         "isbi2012em": 0.164, "monuseg": 0.168, "bacteria": 0.252, "dsb2018": 0.478,
         "ctc_u373": 0.573, "spheroidj": 1.236}
LABEL = {"hrf": "HRF", "drive": "DRIVE", "fisbe": "FISBE", "rozpad": "Decay", "bbbc010": "BBBC010",
         "isbi2012em": "ISBI2012-EM", "monuseg": "MoNuSeg", "bacteria": "Bacteria",
         "dsb2018": "DSB2018", "ctc_u373": "CTC-U373", "spheroidj": "SpheroidJ"}
METRIC = {"drive": "cldice", "hrf": "cldice", "isbi2012em": "cldice", "fisbe": "cldice"}


def seed_mean(arm, ds):
    met = METRIC.get(ds, "fg_iou")
    for f in sorted(glob.glob(os.path.join(ROOT, arm, f"*__{ds}.json"))):
        d = json.load(open(f))
        if d.get("metric") != met:
            continue
        ns, t = len(d["seeds"]), d["test_per_seed"]
        return np.asarray(d["per_image"], float).reshape(ns, t).mean(1)
    raise SystemExit(f"missing record: {arm}/{ds} ({met}) under {ROOT}")


x, y, err, names = [], [], [], []
for ds, w in WIDTH.items():
    full, nobank = seed_mean("stripv3_k8", ds), seed_mean("ablv3_nocls_k8", ds)
    d = full - nobank                                    # seed-paired difference
    x.append(w); y.append(d.mean()); err.append(d.std(ddof=1) / np.sqrt(len(d))); names.append(ds)

fig, ax = plt.subplots(figsize=(3.3, 2.05))
ax.axvline(1.0, color="0.75", lw=0.8, ls=(0, (4, 3)), zorder=1)
ax.text(1.02, 0.263, "one patch", fontsize=6, color="0.45", rotation=90, va="top")
ax.errorbar(x, y, yerr=err, fmt="o", ms=3.4, lw=0, elinewidth=0.8, capsize=1.6,
            color="#1f4e79", ecolor="#7f9dc0", zorder=3)
# Offsets in POINTS, not in data units: the x axis is logarithmic, so a multiplicative nudge moves
# a label by a different visual distance at each end of the axis. Hand-placed where points crowd.
OFF = {"hrf": (4, -9, "left"), "drive": (5, 2, "left"), "fisbe": (-4, -9, "right"),
       "rozpad": (5, 1, "left"), "bbbc010": (-4, 4, "right"), "isbi2012em": (4, -9, "left"),
       "monuseg": (4, 3, "left"), "bacteria": (4, 2, "left"), "dsb2018": (-5, 3, "right"),
       "ctc_u373": (-5, 2, "right"), "spheroidj": (-5, 3, "right")}
for xi, yi, n in zip(x, y, names):
    ox, oy, ha = OFF[n]
    ax.annotate(LABEL[n], (xi, yi), xytext=(ox, oy), textcoords="offset points",
                fontsize=5.6, color="0.25", ha=ha, va="bottom")
ax.set_xscale("log")
ax.set_xlabel("median foreground half-width  /  patch stride", fontsize=7)
ax.set_ylabel("gain from the prior bank", fontsize=7)
ax.tick_params(labelsize=6.5, length=2.5, pad=1.5)
ax.set_ylim(-0.025, 0.295)
ax.set_xlim(0.025, 2.1)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", lw=0.4, color="0.9", zorder=0)
fig.tight_layout(pad=0.25)
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
fig.savefig(OUT)
print(f"wrote {OUT}: {len(x)} datasets, gain {min(y):.3f}..{max(y):.3f}")
