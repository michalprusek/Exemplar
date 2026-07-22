"""Emit the paper's component-ablation table (Table 2) from the clean campaign tree.

TWO blocks, because the paper makes two different claims and one table row cannot carry both:

  * ARCHITECTURE -- what the head is built from: backbone only -> + classical prior bank -> + gate ->
    + FiLM -> both (the full method). This answers "which component supplies the accuracy".
  * SELF-CONFIGURATION -- the paper's novelty axis, held at the FULL architecture and switching off one
    closed-form rule at a time (adaptive loss, colour/stain channel selection, then all of it). This
    answers "does configuring from the support masks actually buy anything", which the architecture
    block cannot show.

The Mean column is SEED-PAIRED: the six datasets are averaged within each seed and the deviation is
taken across those ten aggregates, so it is a spread over replications of the whole experiment rather
than a spread over datasets (which would mostly measure how different the datasets are).

Same fail-loud contract as make_semantic_tables.py: one metric-matching record per (arm, dataset),
uniform seed counts, all six datasets present, and no writing an incomplete table.

  ASG_SEM_TREE=<results/final10> ASG_SEM_OUT=<paper dir> python scripts/make_ablation_table.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.environ["ASG_SEM_TREE"]
OUT = os.environ.get("ASG_SEM_OUT", ".")
EXPECT_SEEDS = int(os.environ.get("ASG_SEM_SEEDS", "10"))

# The six datasets the ablation runs on: both instance-labelled and thin/vessel morphologies, so a
# component that helps only one of them cannot hide in the mean.
DATASETS = [("monuseg", "fg_iou"), ("drive", "cldice"), ("spheroidj", "fg_iou"),
            ("dsb2018", "fg_iou"), ("ctc_u373", "fg_iou"), ("hrf", "cldice")]
SHOWN = ("monuseg", "drive")                     # the two columns printed per-dataset

ARCH = [("abl_nocls_k8", r"Backbone only (no priors)"),
        ("abl_bank_k8", r"\quad+ classical prior bank"),
        ("abl_cgate_k8", r"\quad\quad+ competitive gate"),
        ("abl_film_k8", r"\quad\quad+ FiLM"),
        ("ours_k8", r"\quad\quad+ gate and FiLM (full)")]
SELFCFG = [("abl_sc_noloss_k8", r"\quad$-$ adaptive loss"),
           ("abl_sc_nocolor_k8", r"\quad$-$ colour selection"),
           ("abl_sc_none_k8", r"\quad$-$ all self-configuration")]
FULL_DIR = "ours_k8"


def per_seed(dirname, ds, metric):
    """Seed-mean vector for one (arm, dataset). Exactly one metric-matching record, or an error."""
    matches = []
    for f in sorted(glob.glob(os.path.join(ROOT, dirname, f"*__{ds}.json"))):
        d = json.load(open(f))
        if d.get("metric") != metric:
            continue
        ns, t = len(d["seeds"]), d["test_per_seed"]
        pi = np.asarray(d["per_image"], float)
        if ns * t != len(pi):
            raise ValueError(f"{f}: per_image {len(pi)} != {ns}*{t}")
        matches.append((f, pi.reshape(ns, t).mean(1), list(d["seeds"])))
    if not matches:
        raise SystemExit(f"missing record: {dirname}/{ds} ({metric}) under {ROOT} — refusing to write "
                         f"an ablation table with a hole in it")
    if len(matches) > 1:
        stem = dirname.rsplit("_k", 1)[0]
        based = [m for m in matches if os.path.basename(m[0]).startswith(f"{stem}__")]
        if len(based) != 1:
            raise ValueError(f"{dirname}/{ds}: ambiguous records {[os.path.basename(f) for f, *_ in matches]}")
        matches = based
    _, arr, sds = matches[0]
    if len(arr) != EXPECT_SEEDS:
        raise ValueError(f"{dirname}/{ds}: {len(arr)} seeds, expected {EXPECT_SEEDS}")
    return arr, sds


def row(dirname):
    """(per-dataset (mean,std) for SHOWN, seed-paired 6-dataset mean, its std)."""
    per, seedsets = {}, set()
    for ds, met in DATASETS:
        arr, sds = per_seed(dirname, ds, met)
        per[ds] = arr
        seedsets.add(tuple(sds))
    if len(seedsets) != 1:
        raise ValueError(f"{dirname}: datasets do not share one seed set; a seed-paired mean would "
                         f"average mismatched replications")
    agg = np.stack([per[ds] for ds, _ in DATASETS]).mean(0)      # (seeds,) seed-paired panel mean
    shown = [(float(per[ds].mean()), float(per[ds].std(ddof=1))) for ds in SHOWN]
    return shown, float(agg.mean()), float(agg.std(ddof=1))


def fmt(m, s, bold=False):
    body = f"{m:.3f}{{\\pm}}{s:.3f}"
    return f"{{\\boldmath${body}$}}" if bold else f"${body}$"


def main():
    best = {}
    data = {}
    for dirname, _ in ARCH + SELFCFG:
        data[dirname] = row(dirname)
    # bold the full method: it is the configuration the paper reports, and in both blocks it is the
    # reference every other row is measured against.
    for i, ds in enumerate(SHOWN):
        best[ds] = FULL_DIR
    lines = []
    for dirname, label in ARCH:
        shown, mn, sd = data[dirname]
        b = dirname == FULL_DIR
        lines.append(f"{label} & " + " & ".join(fmt(m, s, b) for m, s in shown)
                     + f" & {fmt(mn, sd, b)} \\\\")
    # No sub-header row: the "- X" labels already read as removals from the full method, the caption
    # says which block is which, and on a 4-page ISBI body one saved line is worth more than the label.
    lines.append(r"\midrule")
    for dirname, label in SELFCFG:
        shown, mn, sd = data[dirname]
        lines.append(f"{label} & " + " & ".join(fmt(m, s) for m, s in shown)
                     + f" & {fmt(mn, sd)} \\\\")
    body = "\n".join(lines)

    names = ", ".join(n.replace("_", r"\_") for n, _ in DATASETS)
    tex = r"""% tab:ablation — component ablation at K=8, SEMANTIC metric. Generated by
% scripts/make_ablation_table.py from the clean campaign tree; do not hand-edit values.
\begin{table}[t]
\centering
\caption{Component ablation at $K{=}8$ on the semantic metric, mean $\pm$ standard deviation over """ + f"{EXPECT_SEEDS}" + r""" seeds; the full method in bold. Columns are MoNuSeg (foreground intersection-over-union), DRIVE (centreline Dice), and the seed-paired mean over the six ablation datasets (""" + names + r"""). The upper block adds one architectural component at a time; the lower block holds the full architecture and switches off one closed-form self-configuration rule at a time.}
\label{tab:ablation}
{\scriptsize\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{l ccc}
\toprule
Configuration & MoNuSeg & DRIVE & Mean \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    open(os.path.join(OUT, "tab_ablation.tex"), "w").write(tex)
    print(f"wrote tab_ablation.tex ({len(ARCH)} architecture + {len(SELFCFG)} self-config rows, "
          f"all at {EXPECT_SEEDS} seeds)")


if __name__ == "__main__":
    main()
