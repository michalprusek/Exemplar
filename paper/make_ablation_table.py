# Copied from AutoSeg scripts/make_ablation_table.py (cb71f85) by tools/sync_paper.py; default paths adapted, internal notes rewritten in comments and docstrings.
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

# ALL ELEVEN as of 2026-09-01 (the ablation is to be as complete as Table 1).
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
# ASG_DROP_DS=ctc_u373 (the paper's setting: CTC-U373 is not reported, because its conditions need the CTC
# organisers' permission for public use outside the challenge): the datasets named here are left out of every mean.
_DROP = {x for x in os.environ.get("ASG_DROP_DS", "").split(",") if x}
DATASETS = [d for d in DATASETS if d[0] not in _DROP]
NUMWORD = {5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
           11: "eleven"}
# THREE printed columns as of 2026-08-26. MoNuSeg and DRIVE are the COLOUR rule's datasets, so a
# reader comparing the "- adaptive loss" row against them alone would conclude the loss rule does
# nothing: it helps on SpheroidJ (+0.020) and CTC-U373 (+0.008) and on neither of those two.
# SpheroidJ also carries the paper's sharpest single number, the +0.002 the prior bank is worth
# there, which the introduction cites and which had no home in the table.
# THREE printed dataset columns, chosen 2026-10-09 when the two single-scale arms became rows (first
# author: "add the coarse and fine ablation to the table"). Each is where a finding the prose quotes
# lives: DRIVE is where the bank carries the method, SpheroidJ where the features do, and HRF is
# what the coarse scale alone loses. A fourth, CTC-U373 for what the fine scale alone loses, was
# tried and put the table 45 pt past the text column with the deviations in; that number stays in
# the prose as a macro. MoNuSeg left: after the same day's rewording no sentence quotes it.
SHOWN = ("drive", "spheroidj", "hrf")        # the columns printed per-dataset
DISPLAY = {"drive": "DRIVE", "spheroidj": "SpheroidJ", "hrf": "HRF", "ctc_u373": "CTC-U373"}

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
# RE-BASED 2026-07-31, when the gate and FiLM were removed from the method. Two of
# the arms did not have to move: `ablv3_nocls` and `ablv3_bank` carry neither `_cgate` nor `_film`,
# so they ALREADY measured the stripped architecture with and without the bank, and `ablv3_bank` is
# now the full method itself. The three self-configuration arms did have to move -- each carried one
# of the removed tokens, so each answered "what does this rule buy?" on top of two components the
# paper no longer has -- and were re-run as `ablv3s_sc_*`.
#
# The gate and FiLM arms were REMOVED from the table on 2026-08-24. They were shown as
# a third "measured and rejected" block; with the paper no longer resting its contribution on the
# self-configuration axis, two rows of rejected alternatives cost body lines that the argument needs
# more. The arms themselves survive in the score tree (`ablv3_cgate_k8`, `ablv3_film_k8`), so the
# measurement is retrievable without being reported.
# Row label must match the prose name for this arm ("the frozen features alone"). It read
# "Backbone only" while the Results paragraph called the same score directory "features only",
# which made one arm look like two experiments printed 0.025 apart.
# THE BANK-ONLY ROW was added 2026-09-02. Its absence was a real hole: the paper's first
# contribution is that the classical bank ALONE outscores the frozen features alone on seven of
# eleven, and the table underneath it showed only "features" and "features + bank", so the claim had
# no tabular support and a reader could not see on which datasets the bank wins. The arm existed
# (`lean_bankonly_k8`, the same head with the DINO grids zeroed); only the row was missing.
# The two single-scale rows are the full method with one DINOv3 reading removed and the bank kept
# (`_coarseonly`, `_fineonly`; written on the same machine as the full arm). The last row was
# "Both (Exemplar)" while the table had two inputs; with scales in it "both" no longer says which.
ARCH = [("lean_nocls_k8", r"Features only"),
        ("lean_bankonly_k8", r"Priors only"),
        ("lean_coarseonly_k8", r"Coarse scale + priors"),
        ("lean_fineonly_k8", r"Fine scale + priors"),
        ("lean_k8", r"Exemplar")]
# The self-configuration rows are gone with the machinery they measured: the method now has a fixed
# objective and reads a fixed grayscale bank input, so there is nothing to switch off.
SELFCFG = []
# The SAME directory Table 1 reads, so the full row here is numerically identical to the row there.
# ADDED 2026-08-25. Three components the paper advertises and Table 2 never measured.
# They were NOT printed as table rows while the table showed MoNuSeg and DRIVE, because the whole
# finding is that the two single-scale arms fail on HRF and CTC-U373 instead and rows would have
# hidden it. Since 2026-10-09 the two SCALE arms are rows of ARCH and those two datasets are columns;
# the fusion-layer arm stays here. All three still feed the prose macros below.
# RE-RUN IN THE LEAN CONFIGURATION 2026-09-02. The `ablv3_*` arms these replace are
# the stripv3 method: subtracting them from `lean_k8` measured the method change, not the component,
# and reversed every sign. `noup` is NOT among them: see SAME_HOST below.
COMPONENTS = [("lean_coarseonly_k8", "CoarseOnly"), ("lean_fineonly_k8", "FineOnly"),
              ("lean_nomix_k8", "NoMix")]

# ADDED 2026-10-09. The guided upsampler, as (control, ablated, macro). Its delta is
# NOT taken against FULL_DIR. `lean_noup_k8` was run on a second machine, and there the reported method itself
# moves by 0.0011 on the panel mean and by up to 0.0091 on one dataset (SpheroidJ), against an effect
# of 0.004. `lean_ctl_k8` is the reported method re-run in the same job, on the same GPU and from the
# same feature cache as the ablated arm, and is the only arm that delta may be subtracted from. It
# carries the SAME method string as `lean_k8` and must never be read in its place.
# WHERE THE OTHER ARMS RAN, verified 2026-10-09 from the write times of the records on each host and
# by md5 against this tree: `lean_k8`, `lean_nocls_k8` and the three COMPONENTS arms were all written
# on a first machine, so their deltas are same-MACHINE subtractions too. They are not known to be same-GPU:
# that machine has an A100 and an A5000, `lean_abl.sh` put coarse-only and fine-only on the first and no-mix
# on the second, and no log records which card wrote `lean_k8`. `lean_bankonly_k8` is the copy
# written on the SECOND machine, so the bank-only row of the table is cross-host against the full
# row; its margins run from 0.003 to 0.3, and only the smallest (DSB2018) is near the host effect.
SAME_HOST = [("lean_ctl_k8", "lean_noup_k8", "NoUp")]
# The datasets on which adding the bank to the features gains the most and the least, of all
# eleven. The Results prose named both until 2026-10-09, when the ablation paragraph was rewritten
# without scores (supervisor's version: the table carries them). The check stays: it is cheap, it
# is still true, and the macros it guards are what a sentence would quote if one came back. The
# same holds for the every-seed check and SAME_HOST_LARGEST below: their sentences are gone from
# main.tex, their facts are not.
BANK_GAIN_ENDS = ("drive", "spheroidj")
# The datasets the prose names as where that delta is largest; a guard in main() holds it to that.
SAME_HOST_LARGEST = ("drive", "hrf")

FULL_DIR = "lean_k8"

# Every record carries the fingerprint of the split it scored. The table SUBTRACTS arms from
# one another, so two arms scored on different test slices would produce a plausible table
# rather than an error -- and a shifted slice on a download-kind dataset is a documented
# failure here, not a hypothetical one.
_SEEN_FP: dict = {}
_SEEN_N: dict = {}


def assert_paired():
    """Every arm read so far scored the same test images on every dataset, or raise.

    Called twice, and the second call is the one that matters for the prose. The first sat inline
    after the table rows and so never saw the COMPONENT arms, which are loaded later and are exactly
    the arms a delta is subtracted from: a component record on a shifted slice (isbi2012em run at
    pool 20 instead of 16, or a stray data-root variable on the host) passed through and emitted a
    plausible macro. The image count is checked as well as the fingerprint, because a record
    written before fingerprints existed carries none and would otherwise match anything.
    """
    for ds, per_dir in _SEEN_FP.items():
        fps = {fp for fp in per_dir.values() if fp}
        if len(fps) > 1:
            raise ValueError(f"{ds}: arms scored DIFFERENT test splits {per_dir}; subtracting "
                             f"them would compare different images and produce a plausible table")
    for ds, per_dir in _SEEN_N.items():
        if len(set(per_dir.values())) > 1:
            raise ValueError(f"{ds}: arms scored a DIFFERENT NUMBER of test images {per_dir}; "
                             f"they cannot be the same split")


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
    _rec = json.load(open(matches[0][0]))
    _SEEN_FP.setdefault(ds, {})[dirname] = _rec.get("split_fp")
    _SEEN_N.setdefault(ds, {})[dirname] = _rec.get("test_per_seed")
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


def panel_per_seed(dirname):
    """((seeds,) panel mean per seed, the seed ids): what a seed-paired difference is taken over."""
    per, seedsets = [], set()
    for ds, met in DATASETS:
        arr, sds = per_seed(dirname, ds, met)
        per.append(arr)
        seedsets.add(tuple(sds))
    if len(seedsets) != 1:
        raise ValueError(f"{dirname}: datasets do not share one seed set")
    return np.stack(per).mean(0), seedsets.pop()


def macro_tag(ds):
    # TeX control sequences are LETTERS ONLY -- "ctc_u373" would give \dFineOnlyCtcu373, which TeX
    # reads as \dFineOnlyCtcu followed by the text "373", i.e. a stray 373 in the preamble.
    return "".join(c for c in ds.replace("_", "").capitalize() if c.isalpha())


def fmt(m, s, bold=False, short=False):
    # Deviations set \tiny and split off the mean, the same shape tab:fewshot uses. With four numeric
    # columns the inline `0.517{\pm}0.010` form ran 48 pt past \columnwidth and spilled into the
    # neighbouring column; this recovers the width without dropping a column or a digit.
    mean = f"{{\\boldmath${m:.3f}$}}" if bold else f"${m:.3f}$"
    if short:
        # The ridge table (three numeric columns) has room for the deviation at the table's own 7 pt instead of
        # \tiny (5 pt), without its leading zero and without the thin space, as tab:fewshot prints it.
        if not 0 <= s < 1:
            raise ValueError(f"deviation {s!r} is not below one; the short form would misprint it")
        return f"{mean}{{\\scriptsize$\\pm{s:.3f}$}}".replace("$\\pm0.", "$\\pm.", 1)
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


def _names(ds_list):
    names = [DISPLAY[d] for d in ds_list]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


# Which printed dataset is scored by which metric, read from DATASETS so the caption cannot
# describe a column the table no longer has.
_IOU = [d for d in SHOWN if dict(DATASETS)[d] == "fg_iou"]
_CLD = [d for d in SHOWN if dict(DATASETS)[d] == "cldice"]
METRIC_NOTE = (f"{_names(_IOU)} {'are' if len(_IOU) > 1 else 'is'} scored by foreground IoU, "
               f"{_names(_CLD)} by centreline Dice")


def main():
    data = {}
    for dirname, _ in ARCH + SELFCFG:
        data[dirname] = row(dirname)
    # bold the full method: it is the configuration the paper reports, and in both blocks it is the
    # reference every other row is measured against.
    # BOLD MARKS THE BEST VALUE IN EACH COLUMN, not the reported method. (The caption says it in four
    # words, "Best per column in bold": one word more put a one-word line under the table.) With only two rows the two
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
    assert_paired()
    body = "\n".join(lines)

    tex = r"""% tab:ablation — component ablation at K=8, SEMANTIC metric. Generated by
% scripts/make_ablation_table.py from the clean campaign tree; do not hand-edit values.
\begin{table}[t]
\centering
\caption{Component ablation at $K{=}8$ (mean $\pm$ standard deviation over """ + f"{EXPECT_SEEDS}" + r""" seeds). """ + METRIC_NOTE + r""", and Mean over all """ + NUMWORD[len(DATASETS)] + r""" datasets of \cref{tab:fewshot}, each by its own metric. Best per column in bold.}
\label{tab:ablation}
{\scriptsize\setlength{\tabcolsep}{1.2pt}
\begin{tabular}{l """ + "c" * (len(SHOWN) + 1) + r"""}
\toprule
Configuration & """ + " & ".join(DISPLAY[d] for d in SHOWN) + r""" & Mean \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    open(os.path.join(OUT, "tab_ablation.tex"), "w").write(tex)

    # Component deltas for the prose. Signed as "what the component is worth", i.e. FULL minus
    # the arm without it, so a positive number means the component helps.
    full_shown, full_mean, _ = data[FULL_DIR]
    lines_c = ["% Generated by scripts/make_ablation_table.py. Do not hand-edit.",
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
    for ctl, ablated, macro in SAME_HOST:
        # The control must BE the reported method and the ablated arm one token from it, or the
        # difference is not this component's. Both are fatal: a control that drifted from the
        # reported method would still subtract cleanly and print a plausible number.
        if method_of(ctl) != full_method:
            raise ValueError(f"{ctl} is {method_of(ctl)}, not the reported method {full_method}; "
                             f"it cannot stand in for {FULL_DIR} as a same-host control")
        if not one_token_apart(full_method, method_of(ablated)):
            raise ValueError(f"{ablated} ({method_of(ablated)}) is not one token from {full_method}")
        _, ctl_mean, _ = row(ctl)
        _, abl_mean, _ = row(ablated)
        lines_c.append(f"% \\d{macro} = {ctl} - {ablated}, both run on one host (registry C90). The same "
                       f"method on the two hosts: {ctl} - {FULL_DIR} = {ctl_mean - full_mean:+.4f}.")
        lines_c.append(f"\\newcommand{{\\d{macro}}}{{{ctl_mean - abl_mean:.3f}}}")
        # Where the upsampler pays, quoted in the prose as "largest on the two vessel datasets".
        # Emitted from the SAME pair as the panel delta, and refused if the named datasets stop
        # being the largest: the sentence would then be false with both of its numbers right.
        per_ds = {ds: per_seed(ctl, ds, met)[0].mean() - per_seed(ablated, ds, met)[0].mean()
                  for ds, met in DATASETS}
        top = sorted(per_ds, key=per_ds.get, reverse=True)[:len(SAME_HOST_LARGEST)]
        if set(top) != set(SAME_HOST_LARGEST):
            raise ValueError(f"\\d{macro} is largest on {top}, not on {list(SAME_HOST_LARGEST)}; the "
                             f"Results sentence naming the vessel datasets must change")
        for ds in SAME_HOST_LARGEST:
            lines_c.append(f"\\newcommand{{\\d{macro}{macro_tag(ds)}}}{{{per_ds[ds]:.3f}}}")
    # "Positive in every seed", which the prose says of all four panel deltas. Seed-paired: the same
    # seed is the same support draw in both arms. A delta that changes sign in one draw fails here
    # instead of leaving the sentence standing.
    pairs = ([] if skipped else [(FULL_DIR, d) for d, _ in COMPONENTS]) + [(c, a) for c, a, _ in SAME_HOST]
    for ref, ablated in pairs:
        (a, sa), (b, sb) = panel_per_seed(ref), panel_per_seed(ablated)
        if sa != sb:
            raise ValueError(f"{ref} and {ablated} were not run on the same seeds; no seed-paired delta")
        if not (a - b > 0).all():
            raise ValueError(f"{ref} - {ablated} is not positive in every seed "
                             f"({int((a - b > 0).sum())} of {len(a)}); the Results sentence must change")
    if pairs:
        lines_c.append(f"% Each of these {len(pairs)} panel deltas is positive in every one of the "
                       f"{EXPECT_SEEDS} seeds (seed-paired, checked at generation).")
    # the per-dataset collapses that are the actual finding
    for dirname, macro, ds in ([] if skipped else
                               [("lean_coarseonly_k8", "CoarseOnly", "hrf"),
                                ("lean_fineonly_k8", "FineOnly", "ctc_u373"),
                                ("lean_fineonly_k8", "FineOnly", "spheroidj")]):
        met = dict(DATASETS)[ds]
        a, _ = per_seed(FULL_DIR, ds, met)
        b, _ = per_seed(dirname, ds, met)
        # Both arms are rows of the table since 2026-10-09, so on a printed column the prose number
        # is the difference of the two ROUNDED cells, the convention the bank gains below already
        # follow: a reader who subtracts the rows must land on it.
        delta = (round(a.mean(), 3) - round(b.mean(), 3)) if ds in SHOWN else (a.mean() - b.mean())
        lines_c.append(f"\\newcommand{{\\d{macro}{macro_tag(ds)}}}{{{delta:.3f}}}")
    # The self-configuration total, quoted in the abstract. It was hand-typed there -- the one
    # number in that sentence that was not a macro, and the one whose scope (these seven datasets,
    # not the eleven-dataset panel the sentence opens on) went wrong.
    # The two ends of the architecture ablation, as macros: the Results paragraph quotes them and a
    # hand-typed pair is how a table and its prose drift apart.
    _, feat_seven, _ = row(ARCH[0][0])
    lines_c.append(f"\\newcommand{{\\featOnlyPanel}}{{{feat_seven:.3f}}}")
    lines_c.append(f"\\newcommand{{\\fullPanel}}{{{full_mean:.3f}}}")
    # What the bank is worth on the three datasets the table shows. These were hand-typed and went
    # stale the moment the reported arm changed -- they are differences between two rows of the
    # table directly above them, so there is no excuse for them not to be generated with it.
    # The prose used to read them as "greatest for thin structures", i.e. as a gain that orders
    # inversely with structure width. Over all eleven datasets it does not: HRF, the other vessel
    # dataset, is eighth (0.090), and Decay is second (0.199). What does hold over the eleven is the
    # two ENDS, and that is all the sentence now says ("adds most on DRIVE and least on SpheroidJ"),
    # so the ends are checked here. MoNuSeg's gain is no longer emitted: the table dropped that
    # column on 2026-10-09 and the prose had stopped quoting it (seventh of the eleven, not second).
    bank_gain = {ds: per_seed(FULL_DIR, ds, met)[0].mean() - per_seed(ARCH[0][0], ds, met)[0].mean()
                 for ds, met in DATASETS}
    most, least = max(bank_gain, key=bank_gain.get), min(bank_gain, key=bank_gain.get)
    if (most, least) != BANK_GAIN_ENDS or bank_gain[least] <= 0:
        raise ValueError(f"the bank's gain over the features is largest on {most} and smallest on "
                         f"{least} ({bank_gain[least]:+.4f}), not {BANK_GAIN_ENDS}; the Results "
                         f"sentence that names those two datasets must change")
    for ds in ("drive", "spheroidj"):
        met = dict(DATASETS)[ds]
        a, _ = per_seed(FULL_DIR, ds, met)
        b, _ = per_seed(ARCH[0][0], ds, met)
        # Difference of the ROUNDED row values, not the rounded difference: the table prints three
        # decimals, and a reader who subtracts two of its rows must land on the number the prose
        # quotes. MoNuSeg differs in the third decimal between the two conventions.
        gain = round(a.mean(), 3) - round(b.mean(), 3)
        lines_c.append(f"\\newcommand{{\\bankGain{ds.capitalize()}}}{{{gain:.3f}}}")
    assert_paired()   # again, now that the component arms have been read: see its docstring
    open(os.path.join(OUT, "numbers_components.tex"), "w").write("\n".join(lines_c) + "\n")
    print("wrote numbers_components.tex (%d component deltas)" % (len(COMPONENTS) + len(SAME_HOST)))
    print(f"wrote tab_ablation.tex ({len(ARCH)} architecture + {len(SELFCFG)} self-config rows, "
          f"all at {EXPECT_SEEDS} seeds)")


# ---------------------------------------------------------------------------------------------------------------
# ASG_ABL_SET=exemplar: the table for the method reported since 2026-10-14. The `lean` path above
# stays as it is: its prose guards (BANK_GAIN_ENDS, SAME_HOST_LARGEST, "positive in every seed") belong to sentences
# of the old text and would refuse this table over claims the rewritten text may not make.
#
# Every row is the full method with ONE change, read against `exemplar_k8` (the directory Table 1 reads, so the two
# tables cannot disagree). A change may move several method tokens -- "priors only" drops the magnification, the
# registration and block 12, which a head without an encoder grid cannot use -- so `one_token_apart` cannot define
# it. Each row DECLARES the tokens it removes from the full method and the ones it adds, and the generator asserts
# exactly that set difference: an arm run with any other string fails here instead of printing a plausible row.
# All arms ran on one machine from one code tree with the final arm's job-to-card layout, so every difference is a
# same-host subtraction (noise floor of one computation run twice: up to 0.009 on one dataset, 0.001 on the mean).
EXEMPLAR_FULL = "exemplar_k8"
EXEMPLAR_METHOD = "head_fusion_best_nobank_noloss_nocolor_flat_es_ep500_wd4_do5_mix_noaux_fzm_fam_lf12"
# (score dir, row label, tokens removed from the full method, tokens added, macro tag, required)
EXEMPLAR_BLOCKS = [
    [("exemplar_nocls_k8", r"Features only", set(), {"nocls"}, "FeaturesOnly", True),
     ("exemplar_bankonly_k8", r"Priors only", {"fzm", "fam", "lf12"}, {"bankonly"}, "PriorsOnly", True),
     ("exemplar_coarseonly_k8", r"Coarse view + priors", {"fzm", "fam"}, {"coarseonly"}, "CoarseOnly", True),
     ("exemplar_fineonly_k8", r"Fine view + priors", set(), {"fineonly"}, "FineOnly", True)],
    [("exemplar_nosel_k8", r"Fine view not magnified", {"fzm", "fam"}, set(), "NoZoom", True),
     ("exemplar_fz2_k8", r"Fixed $2{\times}$ magnification", {"fzm"}, {"fz2"}, "FixedTwo", True),
     ("exemplar_fza_k8", r"Width rule only", {"fzm"}, {"fza"}, "WidthRule", True)],
    # Printed only once all eleven records of BOTH exist.
    [("exemplar_nolf12_k8", r"Without the middle layer", {"lf12"}, set(), "NoMiddleLayer", False),
     ("exemplar_nofam_k8", r"Without registration", {"fam"}, set(), "NoRegistration", False)],
]
OVL_DS = [d for d, m in DATASETS if m == "fg_iou"]
CEN_DS = [d for d, m in DATASETS if m == "cldice"]


def declared_difference(full, ablated, removed, added):
    """Raise unless `ablated` is `full` minus exactly `removed` plus exactly `added` (as method tokens)."""
    a, b = set(full.split("_")), set(ablated.split("_"))
    if a - b != removed or b - a != added:
        raise ValueError(f"{ablated} is not {full} minus {sorted(removed)} plus {sorted(added)}: it removes "
                         f"{sorted(a - b)} and adds {sorted(b - a)}")


def has_all_records(dirname):
    return all(glob.glob(os.path.join(ROOT, dirname, f"*__{ds}.json")) for ds, _ in DATASETS)


def group_row(dirname):
    """Seed-paired (mean, std) for Overlap, Centreline and Mean: datasets averaged within a seed, spread across seeds."""
    per, seedsets = {}, set()
    for ds, met in DATASETS:
        arr, sds = per_seed(dirname, ds, met)
        per[ds] = arr
        seedsets.add(tuple(sds))
    if len(seedsets) != 1:
        raise ValueError(f"{dirname}: datasets do not share one seed set")
    out = []
    for grp in (OVL_DS, CEN_DS, OVL_DS + CEN_DS):
        agg = np.stack([per[d] for d in grp]).mean(0)
        out.append((float(agg.mean()), float(agg.std(ddof=1)), agg))
    return out, seedsets.pop()


def main_exemplar():
    preview = os.environ.get("ASG_ABL_PREVIEW", "") == "1"
    if preview and os.path.exists(os.path.join(OUT, "main.tex")):
        raise SystemExit("ASG_ABL_PREVIEW=1 writes a table with rows left out; refusing to write it into the paper directory")
    full_method = method_of(EXEMPLAR_FULL)
    if full_method != EXEMPLAR_METHOD:
        raise ValueError(f"{EXEMPLAR_FULL} is {full_method}, not the reported method {EXEMPLAR_METHOD}")
    full, full_seeds = group_row(EXEMPLAR_FULL)
    blocks, left_out = [], []
    for block in EXEMPLAR_BLOCKS:
        rows = []
        for dirname, label, removed, added, tag, required in block:
            if not has_all_records(dirname):
                if required and not preview:
                    raise SystemExit(f"missing records in {dirname} under {ROOT}: refusing to write an ablation table with a hole in it")
                left_out.append(dirname)
                continue
            declared_difference(full_method, method_of(dirname), removed, added)
            vals, seeds = group_row(dirname)
            if seeds != full_seeds:
                raise ValueError(f"{dirname} and {EXEMPLAR_FULL} were not run on the same seeds; no seed-paired difference")
            rows.append((dirname, label, tag, vals))
        if not all(r[5] for r in block) and len(rows) not in (0, len(block)):   # an optional block prints whole or not at all
            left_out.extend(r[0] for r in rows)
            rows = []
        if rows:
            blocks.append(rows)
    assert_paired()
    allrows = [r for b in blocks for r in b] + [(EXEMPLAR_FULL, r"Exemplar", "Full", full)]
    best = [max(r[3][i][0] for r in allrows) for i in range(3)]
    cell = lambda v, i: fmt(v[0], v[1], v[0] == best[i])
    lines = []
    for b in blocks:
        lines += [f"{label} & " + " & ".join(cell(v, i) for i, v in enumerate(vals)) + r" \\" for _, label, _, vals in b]
        lines.append(r"\midrule")
    lines.append(r"\textbf{Exemplar} & " + " & ".join(cell(v, i) for i, v in enumerate(full)) + r" \\")
    tex = ((r"% PREVIEW: rows left out " + ", ".join(left_out) + "\n") if left_out else "") + r"""% tab:ablation — ablation of the method reported since 2026-10-14, K=8, SEMANTIC metric. Generated by
% scripts/make_ablation_table.py (ASG_ABL_SET=exemplar) from the clean campaign tree; do not hand-edit values.
% Every row is Exemplar with one change, declared as method tokens removed and added and checked at generation.
\begin{table}[t]
\centering
\caption{Ablation at $K{=}8$, mean $\pm$ standard deviation over """ + f"{EXPECT_SEEDS}" + r""" seeds (for each seed the datasets are averaged first). Each row changes one thing in Exemplar. Overlap averages the datasets of \cref{tab:fewshot} scored by foreground IoU, Centreline those scored by centreline Dice, and Mean all """ + NUMWORD[len(DATASETS)] + r""". Best per column in bold.}
\label{tab:ablation}
{\scriptsize\setlength{\tabcolsep}{1.2pt}
\begin{tabular}{l ccc}
\toprule
Configuration & Overlap & Centreline & Mean \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    open(os.path.join(OUT, "tab_ablation.tex"), "w").write(tex)
    # What each change costs, signed FULL minus the row (positive = the full method is better), on the three
    # columns, and in how many seeds the full method is ahead on the panel mean. The text rework picks what it quotes.
    nums = ["% Generated by scripts/make_ablation_table.py (ASG_ABL_SET=exemplar). Do not hand-edit.",
            "% \\abl<Row><Col>: the row's value; \\abl<Row>Delta<Col>: Exemplar minus the row (positive = Exemplar better)."]
    cols = ("Overlap", "Centreline", "Mean")
    for _, label, tag, vals in [r for b in blocks for r in b] + [(EXEMPLAR_FULL, "Exemplar", "Full", full)]:
        for i, c in enumerate(cols):
            nums.append(f"\\newcommand{{\\abl{tag}{c}}}{{{vals[i][0]:.3f}}}")
            if tag != "Full":
                # difference of the ROUNDED cells, so a reader subtracting two rows of the table lands on it
                nums.append(f"\\newcommand{{\\abl{tag}Delta{c}}}{{{round(full[i][0], 3) - round(vals[i][0], 3):.3f}}}")
        if tag != "Full":
            ahead = int((full[2][2] - vals[2][2] > 0).sum())
            nums.append(f"\\newcommand{{\\abl{tag}SeedsAhead}}{{{ahead}}}  % of {EXPECT_SEEDS} seeds, panel mean")
    open(os.path.join(OUT, "numbers_ablation.tex"), "w").write("\n".join(nums) + "\n")
    print(f"wrote tab_ablation.tex and numbers_ablation.tex ({sum(len(b) for b in blocks)} rows + Exemplar"
          + (f"; PREVIEW, left out {left_out}" if left_out else "") + ")")


# ---------------------------------------------------------------------------------------------------------------
# ASG_ABL_SET=ridge: the table for the closed-form ridge, the method reported since
# 2026-10-15. Input and magnification rows are separate runs of scripts/closed_form_bench.py with one ablation flag
# (--blocks, --zoom_fixed), each record's protocol checked against what its row claims. The readout rows are other
# readouts of the SAME fits (`alt` of ridge_k8): same Gram matrices, same lambda rule, another weighting or cut.
# ASG_ABL_SET=ridge2: the same table for the final version of the method (v2, exact
# leave-one-out with per-fold standardisation and weights, single precision), arms `final10/ridge2_*`; every record must
# also carry loo_std=fold and precision=single.
# ASG_ABL_SET=ridge3: v3 = v2 + the unified magnification rule (feature token fzu), every arm
# run with the dataset's own held-out metric (--ex_metric dataset), arms `final10/ridge3_*`.
# ASG_ABL_SET=ridge4: v4 = v3 + ONE blend readout a*w_plain + (1-a)*w_bal cut at 1/2 (primary blend1_ex),
# arms `final10/ridge4_*`; the readout rows are the two ends of the blend (alt plain_mse = a 1, bal_mse = a 0).
RIDGE_STEM = os.environ.get("ASG_ABL_SET", "ridge") if os.environ.get("ASG_ABL_SET", "") in ("ridge", "ridge2", "ridge3", "ridge4") else "ridge"
RIDGE_FULL = f"{RIDGE_STEM}_k8"
_FM = ("head_fusion_best_nobank_noloss_nocolor_flat_es_ep500_wd4_do5_mix_noaux_"
       + ("fzu" if RIDGE_STEM in ("ridge3", "ridge4") else "fzm") + "_fam{}_natres")
RIDGE_PROTO = dict(readout="ridge_all", lam_search="newton", primary="blend1_ex" if RIDGE_STEM == "ridge4" else "sel1_ex",
                   ex_metric="dataset" if RIDGE_STEM in ("ridge3", "ridge4") else "gate", support=8,
                   feat_method=_FM.format("_lf12"),
                   **(dict(loo_std="fold", precision="single") if RIDGE_STEM in ("ridge2", "ridge3", "ridge4") else {}))
# (score dir, alt key or None for per_image, row label, macro tag, protocol fields the row must carry)
RIDGE_BLOCKS = [
    [(f"{RIDGE_STEM}_nocls_k8", None, r"Coarse + fine view only", "FeaturesOnly", dict(blocks="coarse,fine", zoom_fixed=0)),
     (f"{RIDGE_STEM}_bankonly_k8", None, r"Priors only", "PriorsOnly", dict(blocks="bank", zoom_fixed=0)),
     (f"{RIDGE_STEM}_coarseonly_k8", None, r"Coarse view + priors", "CoarseOnly", dict(blocks="coarse,bank", zoom_fixed=0)),
     (f"{RIDGE_STEM}_fineonly_k8", None, r"Fine view + priors", "FineOnly", dict(blocks="fine,bank", zoom_fixed=0))],
    # the reading of the backbone: the last block only, without block 12 (feature string without _lf12)
    [(f"{RIDGE_STEM}_nolf12_k8", None, r"Last layer only", "LastLayerOnly", dict(blocks="coarse,fine,bank", zoom_fixed=0, feat_method=_FM.format(""))),
     (f"{RIDGE_STEM}_nozoom_k8", None, r"Fine view not magnified", "NoZoom", dict(blocks="coarse,fine,bank", zoom_fixed=1))],
    # the readout with a fixed threshold (alt of ridge_k8; restored after a review: no fixed
    # setting serves both groups). The best held-out choice without the SE rule stays out of the table (prose).
    [(RIDGE_FULL, "plain_mse", r"Plain readout, threshold $0.5$", "PlainHalf", None),
     (RIDGE_FULL, "bal_mse", r"Balanced readout, threshold $0.5$", "BalancedHalf", None)],
]

# ASG_ABL_FOREST=views,priors: a block with a random forest in place of the ridge, on the views (final10/views_rf_k8,
# a copy of results/c114/views_rf_panel) and on the priors (final10/ilastik_k8 = Table 1's Prior-bank RF). With
# "Coarse + fine view only" and "Priors only" above it, Table 2 then holds the 2x2 inputs x readout. Each record must
# carry the dataset's metric, the seeds of the full arm and the split fingerprint of the full arm's record, or the
# table is refused.
# For ridge3 the views row reads final10/views_rf3_k8 (the forest on v3's views), and every views record
# must carry the stem's feature string, so a forest on another magnification rule never fills the cell.
FOREST_ROWS = {"views": ("views_rf3_k8" if RIDGE_STEM in ("ridge3", "ridge4") else "views_rf_k8", "views_rf", r"Forest on the views",
                         "ForestViews"),
               "priors": ("ilastik_k8", "ilastik_rf", r"Forest on the priors", "ForestPriors")}
FOREST = [s for s in os.environ.get("ASG_ABL_FOREST", "").split(",") if s]


def forest_group_row(dirname, method):
    per, seedsets = {}, set()
    for ds, met in DATASETS:
        fs = sorted(glob.glob(os.path.join(ROOT, dirname, f"{method}__{ds}.json")))
        ref = sorted(glob.glob(os.path.join(ROOT, RIDGE_FULL, f"*__{ds}.json")))
        if len(fs) != 1 or len(ref) != 1:
            raise SystemExit(f"{dirname}/{ds}: {len(fs)} records ({len(ref)} for {RIDGE_FULL}); refusing the forest row")
        d, r = json.load(open(fs[0])), json.load(open(ref[0]))
        if d.get("metric") != met or d.get("method") != method:
            raise ValueError(f"{fs[0]}: metric {d.get('metric')} / method {d.get('method')}, expected {met} / {method}")
        if d["split_fp"] != r["split_fp"] or d["seeds"] != r["seeds"]:
            raise ValueError(f"{fs[0]}: split {d['split_fp']} / seeds {d['seeds']} differ from {RIDGE_FULL}'s")
        if method == "views_rf" and d["protocol"].get("feat_method") != _FM.format("_lf12"):
            raise ValueError(f"{fs[0]}: views of {d['protocol'].get('feat_method')}, the table needs {_FM.format('_lf12')}")
        pi = np.asarray(d["per_image"], float)
        per[ds] = pi.reshape(len(d["seeds"]), d["test_per_seed"]).mean(1)
        seedsets.add(tuple(d["seeds"]))
    if len(seedsets) != 1:
        raise ValueError(f"{dirname}: datasets do not share one seed set")
    out = []
    for grp in (OVL_DS, CEN_DS, OVL_DS + CEN_DS):
        agg = np.stack([per[d] for d in grp]).mean(0)
        out.append((float(agg.mean()), float(agg.std(ddof=1)), agg))
    return out, seedsets.pop()


def ridge_per_seed(dirname, ds, metric, key, expect):
    """Seed-mean vector of one arm on one dataset from its single ridge record (per_image, or alt[key])."""
    fs = sorted(glob.glob(os.path.join(ROOT, dirname, f"*__{ds}.json")))
    if len(fs) != 1:
        raise SystemExit(f"{dirname}/{ds}: {len(fs)} records under {ROOT}; refusing to write an ablation table")
    d = json.load(open(fs[0]))
    if d.get("metric") != metric or d.get("method") != "ridge":
        raise ValueError(f"{fs[0]}: metric {d.get('metric')} / method {d.get('method')}, expected {metric} / ridge")
    p = d["protocol"]
    for k, v in {**RIDGE_PROTO, **(expect or dict(blocks="coarse,fine,bank", zoom_fixed=0))}.items():
        if p.get(k) != v:
            raise ValueError(f"{fs[0]}: protocol {k}={p.get(k)!r}, the row needs {v!r}")
    ns, t = len(d["seeds"]), d["test_per_seed"]
    pi = np.asarray(d["per_image"] if key is None else d["alt"][key], float)
    if ns * t != len(pi) or ns != EXPECT_SEEDS:
        raise ValueError(f"{fs[0]}: {len(pi)} scores for {ns} seeds x {t} images (expected {EXPECT_SEEDS} seeds)")
    _SEEN_FP.setdefault(ds, {})[dirname] = d.get("split_fp")
    _SEEN_N.setdefault(ds, {})[dirname] = t
    return pi.reshape(ns, t).mean(1), tuple(d["seeds"])


def ridge_group_row(dirname, key, expect):
    per, seedsets = {}, set()
    for ds, met in DATASETS:
        per[ds], sds = ridge_per_seed(dirname, ds, met, key, expect)
        seedsets.add(sds)
    if len(seedsets) != 1:
        raise ValueError(f"{dirname}: datasets do not share one seed set")
    out = []
    for grp in (OVL_DS, CEN_DS, OVL_DS + CEN_DS):
        agg = np.stack([per[d] for d in grp]).mean(0)
        out.append((float(agg.mean()), float(agg.std(ddof=1)), agg))
    return out, seedsets.pop()


def main_ridge():
    full, full_seeds = ridge_group_row(RIDGE_FULL, None, None)
    blocks = []
    for block in RIDGE_BLOCKS:
        rows = []
        for dirname, key, label, tag, expect in block:
            vals, seeds = ridge_group_row(dirname, key, expect)
            if seeds != full_seeds:
                raise ValueError(f"{dirname} and {RIDGE_FULL} were not run on the same seeds; no seed-paired difference")
            rows.append((dirname, label, tag, vals))
        blocks.append(rows)
    if FOREST:
        rows = []
        for key in FOREST:
            dirname, method, label, tag = FOREST_ROWS[key]
            vals, seeds = forest_group_row(dirname, method)
            if tuple(seeds) != tuple(full_seeds):
                raise ValueError(f"{dirname} and {RIDGE_FULL} were not run on the same seeds")
            rows.append((dirname, label, tag, vals))
        blocks.append(rows)
    assert_paired()
    allrows = [r for b in blocks for r in b] + [(RIDGE_FULL, r"Exemplar", "Full", full)]
    best = [max(r[3][i][0] for r in allrows) for i in range(3)]
    cell = lambda v, i: fmt(v[0], v[1], v[0] == best[i], short=True)
    lines = []
    for b in blocks:
        lines += [f"{label} & " + " & ".join(cell(v, i) for i, v in enumerate(vals)) + r" \\" for _, label, _, vals in b]
        lines.append(r"\midrule")
    lines.append(r"\textbf{Exemplar} & " + " & ".join(cell(v, i) for i, v in enumerate(full)) + r" \\")
    tex = r"""% tab:ablation — ablation of the closed-form ridge (the method reported since 2026-10-15), K=8, SEMANTIC metric.
% Generated by scripts/make_ablation_table.py (ASG_ABL_SET=ridge) from the clean campaign tree;
% do not hand-edit values. Input and magnification rows: separate runs, protocol checked; readout rows: alt of ridge_k8.
\begin{table}[t]
\centering
\caption{Ablation at $K{=}8$, mean $\pm$ standard deviation over """ + f"{EXPECT_SEEDS}" + r""" seeds of the mean over the datasets. Best per column in bold.}
\label{tab:ablation}
{\scriptsize\setlength{\tabcolsep}{1.2pt}
\begin{tabular}{l ccc}
\toprule
Configuration & Overlap & Centreline & Mean \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    tex = tex.replace("(ASG_ABL_SET=ridge)", f"(ASG_ABL_SET={RIDGE_STEM})").replace(" of ridge_k8.", f" of {RIDGE_FULL}.")
    open(os.path.join(OUT, "tab_ablation.tex"), "w").write(tex)
    nums = [f"% Generated by scripts/make_ablation_table.py (ASG_ABL_SET={RIDGE_STEM}). Do not hand-edit.",
            "% \\abl<Row><Col>: the row's value; \\abl<Row>Delta<Col>: Exemplar minus the row (positive = Exemplar better)."]
    cols = ("Overlap", "Centreline", "Mean")
    for _, label, tag, vals in [r for b in blocks for r in b] + [(RIDGE_FULL, "Exemplar", "Full", full)]:
        for i, c in enumerate(cols):
            nums.append(f"\\newcommand{{\\abl{tag}{c}}}{{{vals[i][0]:.3f}}}")
            if tag != "Full":
                nums.append(f"\\newcommand{{\\abl{tag}Delta{c}}}{{{round(full[i][0], 3) - round(vals[i][0], 3):.3f}}}")
        if tag != "Full":
            ahead = int((full[2][2] - vals[2][2] > 0).sum())
            nums.append(f"\\newcommand{{\\abl{tag}SeedsAhead}}{{{ahead}}}  % of {EXPECT_SEEDS} seeds, panel mean")
    open(os.path.join(OUT, "numbers_ablation.tex"), "w").write("\n".join(nums) + "\n")
    print(f"wrote tab_ablation.tex and numbers_ablation.tex ({sum(len(b) for b in blocks)} rows + Exemplar)")


if __name__ == "__main__":
    _set = os.environ.get("ASG_ABL_SET", "lean")
    main_exemplar() if _set == "exemplar" else main_ridge() if _set in ("ridge", "ridge2", "ridge3", "ridge4") else main()
