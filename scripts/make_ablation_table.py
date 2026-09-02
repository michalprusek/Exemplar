"""Emit the paper's component-ablation table (Table 2) from the clean campaign tree.

TWO blocks, because the paper makes two different claims and one table row cannot carry both:

  * ARCHITECTURE -- what the head is built from: backbone only -> + classical prior bank (the full
    method). This answers "which component supplies the accuracy".
  * SELF-CONFIGURATION -- held at the FULL architecture and switching off one closed-form rule at a
    time (adaptive loss, colour/stain channel selection, then all of it). This answers "does
    configuring from the support masks buy anything on top", which the architecture block cannot show.

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

# ALL ELEVEN as of 2026-09-01 (user directive: the ablation is to be as complete as Table 1).
# The seven-dataset panel this replaced was inherited, not chosen: it was the set the FIRST
# ablation arms happened to cover back when each arm was expensive, and it survived three method
# changes without anyone re-asking. Its cost was real -- the Results paragraph quoted a
# seven-dataset mean for the same two arms whose eleven-dataset mean it quoted two sentences later,
# so the same configuration appeared under two different numbers on one page. Both architecture
# arms have always had all eleven cells; only the component arms had to be re-run.
DATASETS = [("spheroidj", "fg_iou"), ("rozpad", "fg_iou"), ("dsb2018", "fg_iou"),
            ("monuseg", "fg_iou"), ("ctc_u373", "fg_iou"), ("bbbc010", "fg_iou"),
            ("bacteria", "fg_iou"), ("drive", "cldice"), ("hrf", "cldice"),
            ("isbi2012em", "cldice"), ("fisbe", "cldice")]
NUMWORD = {5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
           11: "eleven"}
# THREE printed columns as of 2026-08-26. MoNuSeg and DRIVE are the COLOUR rule's datasets, so a
# reader comparing the "- adaptive loss" row against them alone would conclude the loss rule does
# nothing: it helps on SpheroidJ (+0.020) and CTC-U373 (+0.008) and on neither of those two.
# SpheroidJ also carries the paper's sharpest single number, the +0.002 the prior bank is worth
# there, which the introduction cites and which had no home in the table.
SHOWN = ("monuseg", "drive", "spheroidj")        # the columns printed per-dataset

# EVERY ARM IS THE REPORTED REGIME (`ablv3_*`, i.e. `_flat_es_ep500_wd4_do5_mix`). The previous
# `abl_*` arms were the BASE head -- a wide 3x3 stem, sixty fixed epochs, no regularisation -- while
# Table 1 reports best_v3. On DRIVE that was 0.690 against 0.771: a regime difference LARGER than
# the largest component effect the table measured, which is a legitimate "major issue" and the
# reason all seven arms were re-run.
#
# The full arm is NOT among them. `oursv3n_k8` already IS the full method in the reported regime at
# ten seeds over all eleven datasets, so pointing at it costs nothing and buys something better: the
# reference row here is numerically identical to Table 1's row, and the two tables cannot disagree.
#
# RE-BASED 2026-07-31, when the gate and FiLM were removed from the method (registry C65). Two of
# the arms did not have to move: `ablv3_nocls` and `ablv3_bank` carry neither `_cgate` nor `_film`,
# so they ALREADY measured the stripped architecture with and without the bank, and `ablv3_bank` is
# now the full method itself. The three self-configuration arms did have to move -- each carried one
# of the removed tokens, so each answered "what does this rule buy?" on top of two components the
# paper no longer has -- and were re-run as `ablv3s_sc_*`.
#
# The gate and FiLM arms were REMOVED from the table on 2026-08-24 (user decision). They were shown as
# a third "measured and rejected" block; with the paper no longer resting its contribution on the
# self-configuration axis, two rows of rejected alternatives cost body lines that the argument needs
# more. The arms themselves survive in the score tree (`ablv3_cgate_k8`, `ablv3_film_k8`) and in
# CONFIG-REGISTRY C64/C65, so the measurement is retrievable without being reported.
# Row label must match the prose name for this arm ("the frozen features alone"). It read
# "Backbone only" while the Results paragraph called the same score directory "features only",
# which made one arm look like two experiments printed 0.025 apart.
# THE BANK-ONLY ROW was added 2026-09-02. Its absence was a real hole: the paper's first
# contribution is that the classical bank ALONE outscores the frozen features alone on seven of
# eleven, and the table underneath it showed only "features" and "features + bank", so the claim had
# no tabular support and a reader could not see on which datasets the bank wins. The arm existed
# (`lean_bankonly_k8`, the same head with the DINO grids zeroed); only the row was missing.
ARCH = [("lean_nocls_k8", r"Features only"),
        ("lean_bankonly_k8", r"Priors only"),
        ("lean_k8", r"Both (Exemplar)")]
# The self-configuration rows are gone with the machinery they measured: the method now has a fixed
# objective and reads a fixed grayscale bank input, so there is nothing to switch off.
SELFCFG = []
# The SAME directory Table 1 reads, so the full row here is numerically identical to the row there.
# ADDED 2026-08-25 (registry C78). Three components the paper advertises and Table 2 never measured.
# They are NOT printed as table rows: the two columns this table shows are MoNuSeg and DRIVE, and the
# whole finding is that the two single-scale arms fail on HRF and CTC-U373 instead -- rows would hide
# it. Their panel deltas are emitted as macros for the prose to cite, so no number is hand-typed.
# RE-RUN IN THE LEAN CONFIGURATION 2026-09-02 (registry C87). The `ablv3_*` arms these replace are
# the stripv3 method: subtracting them from `lean_k8` measured the method change, not the component,
# and reversed every sign. `noup` is NOT among them -- the prior-guided upsampler was never quoted in
# the paper, and a fourth arm was not worth eleven more cells of GPU time.
COMPONENTS = [("lean_coarseonly_k8", "CoarseOnly"), ("lean_fineonly_k8", "FineOnly"),
              ("lean_nomix_k8", "NoMix")]

FULL_DIR = "lean_k8"

# Every record carries the fingerprint of the split it scored. The table SUBTRACTS arms from
# one another, so two arms scored on different test slices would produce a plausible table
# rather than an error -- and a shifted slice on a download-kind dataset is a documented
# failure here, not a hypothetical one.
_SEEN_FP: dict = {}


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
    _SEEN_FP.setdefault(ds, {})[dirname] = json.load(open(matches[0][0])).get("split_fp")
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
    # Deviations set \tiny and split off the mean, the same shape tab:fewshot uses. With four numeric
    # columns the inline `0.517{\pm}0.010` form ran 48 pt past \columnwidth and spilled into the
    # neighbouring column; this recovers the width without dropping a column or a digit.
    mean = f"{{\\boldmath${m:.3f}$}}" if bold else f"${m:.3f}$"
    return f"{mean}\\,{{\\tiny$\\pm{s:.3f}$}}"


def method_of(dirname):
    """The method string the arm was actually run with, read from its records (they must agree)."""
    names = set()
    for f in sorted(glob.glob(os.path.join(ROOT, dirname, "*.json"))):
        names.add(json.load(open(f))["method"])
    if len(names) != 1:
        raise ValueError(f"{dirname}: {len(names)} distinct method strings {sorted(names)}; "
                         f"an arm that mixes methods cannot be one column of an ablation")
    return names.pop()


def one_token_apart(full, ablated):
    """True iff `ablated` is `full` with exactly one lever token added or removed.

    That is the whole definition of a component ablation, and checking it is what stops the
    generator subtracting two different METHODS and printing the difference as a component's worth.
    """
    a, b = set(full.split("_")), set(ablated.split("_"))
    return len(a ^ b) == 1


def main():
    data = {}
    for dirname, _ in ARCH + SELFCFG:
        data[dirname] = row(dirname)
    # bold the full method: it is the configuration the paper reports, and in both blocks it is the
    # reference every other row is measured against.
    # BOLD MARKS THE BEST VALUE IN EACH COLUMN, not the reported method. With only two rows the two
    # conventions coincided and "Bold denotes Exemplar" was the honest caption; with a third row a
    # reader compares columns, and bolding one arm by name would hide any column it did not win.
    lines = []
    best_shown = [max(data[d][0][i][0] for d, _ in ARCH) for i in range(len(SHOWN))]
    best_mean = max(data[d][1] for d, _ in ARCH)
    for dirname, label in ARCH:
        shown, mn, sd = data[dirname]
        lines.append(f"{label} & "
                     + " & ".join(fmt(m, s, m == best_shown[i]) for i, (m, s) in enumerate(shown))
                     + f" & {fmt(mn, sd, mn == best_mean)} \\\\")
    # The caption's last two sentences -- that this table was measured on the pre-lean base head
    # and that the magnitudes are therefore indicative -- were HAND-ADDED to the .tex and were
    # absent here, so the next regeneration would have silently deleted the paper's honesty
    # disclosure about which head the ablation ran on. A caption that qualifies the numbers is
    # part of the numbers; it belongs with the generator that produces them.
    # No sub-header row: the "- X" labels already read as removals from the full method, the caption
    # says which block is which, and on a 4-page ISBI body one saved line is worth more than the label.
    # Only when there IS a second block. With SELFCFG empty (the self-configuration rows went with
    # the method they described) an unconditional rule renders as a doubled line above \bottomrule.
    if SELFCFG:
        lines.append(r"\midrule")
    for dirname, label in SELFCFG:
        shown, mn, sd = data[dirname]
        lines.append(f"{label} & " + " & ".join(fmt(m, s) for m, s in shown)
                     + f" & {fmt(mn, sd)} \\\\")
    for ds, per_dir in _SEEN_FP.items():
        fps = {fp for fp in per_dir.values() if fp}
        if len(fps) > 1:
            raise ValueError(f"{ds}: arms scored DIFFERENT test splits {per_dir}; subtracting "
                             f"them would compare different images and produce a plausible table")
    body = "\n".join(lines)

    tex = r"""% tab:ablation — component ablation at K=8, SEMANTIC metric. Generated by
% scripts/make_ablation_table.py from the clean campaign tree; do not hand-edit values.
\begin{table}[t]
\centering
\caption{Component ablation at $K{=}8$ (mean $\pm$ standard deviation over """ + f"{EXPECT_SEEDS}" + r""" seeds). MoNuSeg and SpheroidJ are scored by foreground IoU, DRIVE by centreline Dice, and Mean over all """ + NUMWORD[len(DATASETS)] + r""" datasets of \cref{tab:fewshot}, each by its own metric. Bold marks the best value in each column.}
\label{tab:ablation}
{\scriptsize\setlength{\tabcolsep}{1.2pt}
\begin{tabular}{l cccc}
\toprule
Configuration & MoNuSeg & DRIVE & SpheroidJ & Mean \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    open(os.path.join(OUT, "tab_ablation.tex"), "w").write(tex)

    # Component deltas for the prose (C78). Signed as "what the component is worth", i.e. FULL minus
    # the arm without it, so a positive number means the component helps.
    full_shown, full_mean, _ = data[FULL_DIR]
    lines_c = ["% Generated by scripts/make_ablation_table.py (registry C78). Do not hand-edit.",
               "% Panel deltas over the same eleven datasets as tab:ablation, i.e. the full panel.",
               "% Signed FULL - ablated, so positive = the component earns its place."]
    # A DELTA IS ONLY A COMPONENT DELTA IF BOTH ARMS ARE THE SAME METHOD. The `ablv3_*` cells were
    # run against `stripv3_k8`; once FULL_DIR moved to the lean arm, `full_mean - mn` subtracted two
    # DIFFERENT methods and every delta changed sign -- the paper printed "two scales are worth
    # -0.002", an artefact of the method change, not a measurement of the scales. Rather than emit a
    # plausible wrong number, refuse: a missing macro breaks the build loudly, which is the point.
    full_method = method_of(FULL_DIR)
    skipped = []
    for dirname, macro in COMPONENTS:
        if not one_token_apart(full_method, method_of(dirname)):
            skipped.append((macro, dirname))
            continue
        _, mn, _ = row(dirname)
        lines_c.append(f"\\newcommand{{\\d{macro}}}{{{full_mean - mn:.3f}}}")
    if skipped:
        lines_c.append("% NOT EMITTED -- ablated arm is a different method from " + FULL_DIR + ":")
        for macro, dirname in skipped:
            lines_c.append(f"%   \\d{macro} ({dirname}: {method_of(dirname)})")
        print("SKIPPED %d component delta(s): arms are a different method from %s"
              % (len(skipped), FULL_DIR))
    # the per-dataset collapses that are the actual finding
    for dirname, macro, ds in ([] if skipped else
                               [("lean_coarseonly_k8", "CoarseOnly", "hrf"),
                                ("lean_fineonly_k8", "FineOnly", "ctc_u373"),
                                ("lean_fineonly_k8", "FineOnly", "spheroidj")]):
        met = dict(DATASETS)[ds]
        a, _ = per_seed(FULL_DIR, ds, met)
        b, _ = per_seed(dirname, ds, met)
        # TeX control sequences are LETTERS ONLY -- "ctc_u373" would give \dFineOnlyCtcu373, which TeX
        # reads as \dFineOnlyCtcu followed by the text "373", i.e. a stray 373 in the preamble.
        tag = "".join(c for c in ds.replace("_", "").capitalize() if c.isalpha())
        lines_c.append(f"\\newcommand{{\\d{macro}{tag}}}{{{a.mean()-b.mean():.3f}}}")
    # The self-configuration total, quoted in the abstract. It was hand-typed there -- the one
    # number in that sentence that was not a macro, and the one whose scope (these seven datasets,
    # not the eleven-dataset panel the sentence opens on) went wrong.
    # The two ends of the architecture ablation, as macros: the Results paragraph quotes them and a
    # hand-typed pair is how a table and its prose drift apart.
    _, feat_seven, _ = row(ARCH[0][0])
    lines_c.append(f"\\newcommand{{\\featOnlyPanel}}{{{feat_seven:.3f}}}")
    lines_c.append(f"\\newcommand{{\\fullPanel}}{{{full_mean:.3f}}}")
    # What the bank is worth on the three datasets the table shows, which the Results prose quotes to
    # make the point that the gain orders inversely with structure width. These were hand-typed and
    # went stale the moment the reported arm changed -- they are differences between two rows of the
    # table directly above them, so there is no excuse for them not to be generated with it.
    for ds in ("drive", "monuseg", "spheroidj"):
        met = dict(DATASETS)[ds]
        a, _ = per_seed(FULL_DIR, ds, met)
        b, _ = per_seed(ARCH[0][0], ds, met)
        # Difference of the ROUNDED row values, not the rounded difference: the table prints three
        # decimals, and a reader who subtracts two of its rows must land on the number the prose
        # quotes. MoNuSeg differs in the third decimal between the two conventions.
        gain = round(a.mean(), 3) - round(b.mean(), 3)
        lines_c.append(f"\\newcommand{{\\bankGain{ds.capitalize()}}}{{{gain:.3f}}}")
    open(os.path.join(OUT, "numbers_components.tex"), "w").write("\n".join(lines_c) + "\n")
    print("wrote numbers_components.tex (%d component deltas)" % len(COMPONENTS))
    print(f"wrote tab_ablation.tex ({len(ARCH)} architecture + {len(SELFCFG)} self-config rows, "
          f"all at {EXPECT_SEEDS} seeds)")


if __name__ == "__main__":
    main()
