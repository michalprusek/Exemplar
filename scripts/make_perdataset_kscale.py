"""One K-scaling panel per dataset: our curve against every baseline that has the same K coverage.

WHY. The paper's Fig. 2 plots panel MEANS, which is the right figure for an annotation-efficiency
claim but hides everything a per-dataset reader wants: where our curve is flat, where a baseline
crosses us, where nnU-Net overtakes and at which K. This draws the eleven curves that mean was
made of, so the shape of each dataset is visible rather than averaged away.

Arm hygiene: the method arm is read from ONE directory family (`stripv3_k*`). Mixing arms is a
shipped failure in this project -- a K-scaling figure once drew `ours_k*` beside `oursv3n_k8` and
understated our own curve by up to 0.082 -- so a missing K is left as a gap, never back-filled from
a neighbouring arm.

    ASG_SEM_TREE=results/final10 ASG_OUT=~/Desktop python scripts/make_perdataset_kscale.py
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42          # never Type 3
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.environ.get("ASG_SEM_TREE", "results/final10")
OUT = os.path.expanduser(os.environ.get("ASG_OUT", "~/Desktop"))
KS = (1, 4, 8, 16)

DATASETS = [("spheroidj", "fg_iou", "SpheroidJ"), ("rozpad", "fg_iou", "Decay"),
            ("dsb2018", "fg_iou", "DSB2018"), ("monuseg", "fg_iou", "MoNuSeg"),
            ("ctc_u373", "fg_iou", "CTC-U373"), ("bbbc010", "fg_iou", "BBBC010"),
            ("bacteria", "fg_iou", "Bacteria"), ("drive", "cldice", "DRIVE"),
            ("hrf", "cldice", "HRF"), ("isbi2012em", "cldice", "ISBI2012-EM"),
            ("fisbe", "cldice", "FISBe")]

# name -> (directory template, colour, marker, z-order). A tuple of templates means a baseline with
# several documented modes, read at the per-dataset better one, exactly as Table 1 prints it.
METHODS = [
    ("Exemplar",   ("stripv3_k{k}",),                          "#111111", "o", 3.0),
    ("nnU-Net",    ("nnunet_k{k}",),                           "#d62728", "s", 2.5),
    ("SegGPT",     ("seggpt_k{k}",),                           "#9467bd", "^", 1.5),
    ("Tyche",      ("tyche_k{k}",),                            "#1f77b4", "v", 1.5),
    ("UniverSeg",  ("universeg_k{k}",),                        "#2ca02c", "D", 1.5),
    ("INSID3", ("insid3_k{k}",), "#ff7f0e", "P", 1.5),
]


# `nnunet_k1` was written one level up from the other three K arms. Searching both is not laxity:
# omitting it would silently drop the single point where we lead nnU-Net, which is the opposite of
# a conservative default.
SEARCH = [ROOT, os.path.dirname(ROOT) or "."]


def score(dirname, ds, metric):
    """Seed-averaged mean for one cell, or None. Metric must match: a record scored as instance AP
    is a different quantity, and silently averaging it in is how a panel gets a wrong number."""
    cands = [g for base in SEARCH
             for g in sorted(glob.glob(os.path.join(base, dirname, f"*__{ds}.json")))]
    for f in cands:
        d = json.load(open(f))
        if d.get("metric") != metric:
            continue
        return float(np.asarray(d["per_image"], float).mean())
    return None


def best_of(templates, k, ds, metric):
    vals = [v for t in templates
            for v in [score(t.format(k=k), ds, metric)] if v is not None]
    return max(vals) if vals else None


os.makedirs(OUT, exist_ok=True)
drawn = []
for ds, metric, label in DATASETS:
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for name, templates, colour, marker, z in METHODS:
        xs, ys = [], []
        for k in KS:
            v = best_of(templates, k, ds, metric)
            if v is not None:
                xs.append(k); ys.append(v)
        if not xs:
            continue
        ax.plot(xs, ys, marker=marker, color=colour, zorder=z,
                lw=2.2 if name == "Exemplar" else 1.3,
                ms=6 if name == "Exemplar" else 4.5, label=name)
    ax.set_xscale("log", base=2)
    ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("Support masks $K$")
    ax.set_ylabel("Foreground IoU" if metric == "fg_iou" else "Centreline Dice")
    ax.set_title(f"{label}  ({'IoU' if metric == 'fg_iou' else 'clDice'})", fontsize=11)
    ax.set_ylim(0, 1); ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    p = os.path.join(OUT, f"kscale_{ds}.pdf")
    fig.savefig(p); fig.savefig(p.replace(".pdf", ".png"), dpi=160)
    plt.close(fig)
    drawn.append(label)
    print(f"  {label:<12} -> {os.path.basename(p)}")

print(f"\nwrote {len(drawn)} per-dataset figures to {OUT}")
