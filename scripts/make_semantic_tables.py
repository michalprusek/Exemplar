"""Emit the MERGED main semantic-comparison table for the paper, from the clean campaign tree.

One Table 1 places our method against the budget-matched few-shot baselines (which see the
same eight support masks) AND the trained specialists (which use no support but were trained on thousands
of objects), on the SAME per-dataset semantic metric (foreground IoU for blob/nucleus/worm/bacteria/decay
fields, centreline Dice for the vessel/membrane/filament fields). Eleven datasets, FLAT (no development/
held-out split). Bold = best in the row across ALL methods (few-shot + specialists together) -- NOT per
group, because a per-group best would hide exactly the cross-group comparisons the caption invites.

The method columns sit in three labelled blocks, split by HOW a method consumes the K support masks --
the same taxonomy the manuscript argues in prose. A method that FITS a model on the eight masks (ours,
ilastik-RF, nnU-Net) is not doing what a forward-pass in-context method does, and printing all eight in
one undifferentiated "same-budget" block made nnU-Net's nine row-bests read as a loss to a peer rather
than as the fit-cost trade the text spends a paragraph on.

FAIL-LOUD CONTRACT. This is the last hop between the score tree and the paper's headline table, so every
way a partial or ambiguous tree could masquerade as a measurement is an error, not a default:
  * a directory holding two metric-matching records (two method-name generations, a stray variant) is
    AMBIGUOUS -> raise, never silently take the better one (``make_final_kscale.find`` does the same);
  * files present but none carrying the scored metric is a harness misconfiguration -> raise;
  * every rendered cell must carry the SAME seed count (the caption asserts it) -> raise otherwise;
  * a cell printed WITHOUT a deviation must actually be deterministic -> verified, not assumed;
  * the panel Mean is computed only over the FULL dataset list -- a mean over a method-specific subset
    is not comparable to a full-panel mean and must never be bolded against one;
  * the two caption claims that depend on the numbers (the fine-tuned-specialist delta, and which
    datasets the specialists win) are DERIVED from the tree, not hand-typed;
  * an empty/unresolvable tree must never overwrite a good table.

Writes:
  tab_fewshot.tex  -- the merged 11-dataset, 11-method main results table (\\label{tab:fewshot}).

The dataset order is frozen to match the paper. Run:
  ASG_SEM_TREE=/disk1/prusek/active-segmenter/results/final10 \
  ASG_SEM_OUT=<paper dir> python scripts/make_semantic_tables.py
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.environ["ASG_SEM_TREE"]                      # dir holding {method}_k{K}/ and {specialist}/
OUT = os.environ.get("ASG_SEM_OUT", ".")
# The caption states this; every rendered cell is checked against it.
EXPECT_SEEDS = int(os.environ.get("ASG_SEM_SEEDS", "10"))

# Eleven datasets, FLAT, in the frozen paper order (SpheroidJ … FISBE). Each carries its semantic metric
# (foreground IoU for blob/nucleus/worm/bacteria/decay fields, centreline Dice for vessels/membranes/
# filaments). No development/held-out grouping.
DATASETS = [
    ("spheroidj", "SpheroidJ", "fg_iou"),
    ("rozpad", "Decay", "fg_iou"),
    ("dsb2018", "DSB2018", "fg_iou"),
    ("monuseg", "MoNuSeg", "fg_iou"),
    ("ctc_u373", "CTC-U373", "fg_iou"),
    ("bbbc010", "BBBC010", "fg_iou"),
    ("bacteria", "Bacteria", "fg_iou"),
    ("drive", "DRIVE", "cldice"),
    ("hrf", "HRF", "cldice"),
    ("isbi2012em", "ISBI2012-EM", "cldice"),
    ("fisbe", "FISBE", "cldice"),
]

# (dirname, dataset) -> seed count of the record actually rendered; checked for uniformity before write.
_SEEN_SEEDS = {}


def stat(dirname, ds, metric):
    """(mean, std, n_seeds) on the semantic metric for one (method-dir, dataset), or None if absent.

    A directory must resolve to exactly ONE metric-matching record. Two generations of a method name in
    one dir (this repo has already renamed its best method once: ``head_fusion_best_cgate_film`` ->
    ``..._nobank``, CLAUDE.md C13) would otherwise let a silent max() report whichever scored higher --
    i.e. publish a number from a method the paper says was dropped. When a dir legitimately holds a base
    and a variant record (PerSAM writes ``persam__`` and ``persam_f__``), the base variant wins; anything
    still ambiguous raises.
    """
    matches, other_metrics = [], set()
    for f in sorted(glob.glob(os.path.join(ROOT, dirname, f"*__{ds}.json"))):
        d = json.load(open(f))
        if d.get("metric") != metric:
            other_metrics.add(d.get("metric"))
            continue
        pi = np.asarray(d["per_image"], float)
        ns, t = len(d["seeds"]), d["test_per_seed"]
        if ns * t != len(pi):
            raise ValueError(f"{f}: per_image {len(pi)} != {ns}*{t}; malformed/partial score file")
        sm = pi.reshape(ns, t).mean(1)
        matches.append((f, (float(sm.mean()), float(sm.std(ddof=1)) if ns > 1 else float("nan"), ns)))
    if not matches:
        if other_metrics:
            # Files exist but none is scored on the metric this table reports. Silently returning "--"
            # here is the documented "metric mismatch skipped half a comparison" failure.
            raise ValueError(f"{dirname}/{ds}: no record with metric={metric!r}; "
                             f"found {sorted(m for m in other_metrics if m)}")
        return None
    if len(matches) > 1:
        stem = dirname.rsplit("_k", 1)[0]
        based = [m for m in matches if os.path.basename(m[0]).startswith(f"{stem}__")]
        if len(based) != 1:
            raise ValueError(f"{dirname}/{ds}: {len(matches)} metric-matching records "
                             f"{[os.path.basename(f) for f, _ in matches]} -- ambiguous, refusing to guess")
        matches = based
    _SEEN_SEEDS[(dirname, ds)] = matches[0][1][2]
    return matches[0][1]


def insid3(ds, metric):
    """INSID3 -- the AUTHORS' released implementation, replacing our own read-out (2026-09-02).

    Until now this column was a training-free correspondence read-out we wrote ourselves following
    INSID3's premise, and the caption had to say so. It is now their code (github.com/visinf/INSID3),
    run through `scripts/insid3_bench.py` on our split, our seeds and the same K=8 draw, with the
    CRF refinement their published Chest X-ray number reproduces from -- 78.7 against their 78.8,
    registry C88. Their code default is bilinear and would understate them by 1.5 points.

    The old two-mode steelman (`insid3_guided_k8` / `insid3_dense_k8`, better of two CRF settings
    per dataset) is gone with the construction it steelmanned: there is one configuration now, the
    one the authors report.
    """
    return stat("insid3_k8", ds, metric)


def resolve(dirname, ds, metric):
    """Dispatch to the INSID3 helper, else a plain directory stat."""
    return insid3(ds, metric) if dirname == "insid3" else stat(dirname, ds, metric)


def cell(dirname, ds, metric, det=False):
    """One rendered table cell. ``det`` marks a column the caption calls deterministic -- that is
    VERIFIED here, not merely expressed by dropping the deviation."""
    r = resolve(dirname, ds, metric)
    if r is None:
        return "--"
    m, s, n = r
    if det:
        if n > 1 and not (np.isnan(s) or s < 1e-6):
            raise ValueError(f"{dirname}/{ds}: rendered without a deviation (the caption calls these "
                             f"deterministic) but std={s:.4g} over {n} seeds")
        return f"{m:.3f}"
    if n < 2 or np.isnan(s):
        # The bare format is reserved for the deterministic specialists; a 1-seed in-context cell must
        # not borrow it, or it reads as a deterministic measurement under a caption that says so.
        raise ValueError(f"{dirname}/{ds}: only {n} seed(s) -- refusing to render an in-context cell in "
                         f"the deterministic (no-deviation) format")
    # A deviation below half a milli-unit rounds to "0.000", which reads as "this method is
    # deterministic" -- the very thing the caption reserves for the specialist columns. It is real
    # but tiny: Matcher on MoNuSeg spans 0.2157-0.2165 across ten seeds (sd 3.6e-4), because it
    # personalises to one salient object and fails the same way on dense H&E whichever mask it gets.
    # Print the bound instead of a zero that claims something stronger than the data.
    if s < 5e-4:
        return f"{m:.3f}\\pm{{<}}0.001"
    return f"{m:.3f}\\pm{s:.3f}"


def bold(x):
    return f"{{\\boldmath${x}$}}" if "--" not in x else x


def render(v, is_best):
    """Typeset one cell, with the deviation two sizes down.

    Eleven datasets by twelve columns overflowed the text block by 112pt at a uniform size, which is a
    formatting defect on the page, not a tight fit. The deviation is the part to shrink: it qualifies
    the precision of the mean beside it, and no claim in the paper compares deviations across columns,
    whereas every claim compares means. The mean stays at the table's size and carries the bolding.
    """
    if v == "--":
        return "--"
    if "\\pm" not in v:
        return bold(v) if is_best else f"${v}$"
    m, s = v.split("\\pm")
    core = f"{{\\boldmath${m}$}}" if is_best else f"${m}$"
    return core + r"\,{\tiny$\pm" + s + "$}"


def col_mean(dirname):
    """Panel mean over ALL datasets, or None when the method is missing any.

    A mean over a method-specific subset is systematically easier (the missing runs are usually the hard
    datasets) and would still be bolded against full-panel columns, so a partial column gets no Mean.
    """
    vals = [resolve(dirname, ds, m) for ds, _, m in DATASETS]
    if any(v is None for v in vals):
        missing = [ds for (ds, _, _), v in zip(DATASETS, vals) if v is None]
        print(f"  ! {dirname}: Mean suppressed -- missing {missing}", file=sys.stderr)
        return None
    return sum(v[0] for v in vals) / len(vals)


def _subset_mean(dirname, subset):
    """Mean over a metric-defined subset, or None if the column is missing any of it."""
    vals = [resolve(dirname, ds, m) for ds, _, m in DATASETS if m == subset]
    if any(v is None for v in vals):
        return None
    return sum(v[0] for v in vals) / len(vals)


def metric_split_rows(cols):
    """Two aggregation rows: the mean over the overlap-scored datasets and over the centreline ones.

    WHY THE TABLE NEEDS THEM. The panel Mean averages two different quantities and says so, which
    makes it the wrong statistic for this paper's central comparison. Split by metric at K=8 the two
    families separate completely: ours is ahead on the seven datasets scored by foreground overlap
    and behind on the four scored by centreline Dice, and the entire panel-mean difference is the
    second group. A reader who sees only the combined mean concludes we are a little behind
    everywhere, which is not what the numbers say.
    """
    out = []
    for subset, label in (("fg_iou", "\\quad overlap"),
                          ("cldice", "\\quad centreline")):
        n = sum(1 for _, _, m in DATASETS if m == subset)
        means = {nm: _subset_mean(d, subset) for nm, d, _ in cols}
        best = max((m for m in means.values() if m is not None), default=None)
        cells = []
        for nm, _, _ in cols:
            v = means[nm]
            cells.append("--" if v is None else
                         render("%.3f" % v, best is not None and abs(v - best) < 1e-9))
        out.append(label + " & " + " & ".join(cells) + " \\\\")
    return "\n".join(out)


def mean_row(cols):
    """A panel-mean aggregation row (best in bold), placed under the table's last rule."""
    means = {n: col_mean(d) for n, d, _ in cols}
    best = max((m for m in means.values() if m is not None), default=None)
    cells = []
    for n, _, _ in cols:
        m = means[n]
        if m is None:
            cells.append("--")
        else:
            s = f"{m:.3f}"
            cells.append(render(s, best is not None and abs(m - best) < 1e-9))
    return (r"\midrule" + "\n" + r"\textbf{Mean} & " + " & ".join(cells) + r" \\"
            + "\n" + metric_split_rows(cols))


def build_body(cols):
    """One row per dataset, FLAT. ``cols`` = [(name, dir, det)]; the best score per row is bolded on the
    numeric mean across ALL methods (few-shot + specialists), not per group."""
    out = []
    for ds, name, metric in DATASETS:
        raw = {n: cell(d, ds, metric, det) for n, d, det in cols}
        means = {n: float(v.split("\\pm")[0].strip("{}$\\boldmath ")) for n, v in raw.items() if v != "--"}
        best = max(means.values()) if means else None
        cells = []
        for n, _, _ in cols:
            v = raw[n]
            is_best = v != "--" and best is not None and abs(means[n] - best) < 1e-9
            cells.append(render(v, is_best))
        out.append(f"{name:12} & " + " & ".join(cells) + r" \\")
    return "\n".join(out)


# The eleven method columns in THREE labelled blocks, split by COMPUTE REGIME -- by how a method
# consumes the K support masks, which is the taxonomy the text already argues. Third tuple field =
# deterministic (verified in ``cell``); the block a column belongs to is now simply the group it is
# listed under, so a column cannot end up under a heading that does not describe it.
# Ours, ilastik-RF and nnU-Net all FIT a model on the same eight masks; SegGPT/UniverSeg/INSID3/Tyche/
# Matcher read them in a forward pass. Printing all eight in one "same-budget few-shot" block asserted
# an equivalence the paper never claims and made nnU-Net's nine row-bests read as a loss to a peer
# instead of the fit-cost trade the text spends a paragraph on. The specialists stay last because
# they get no support at all: they are the block the reader should compare against differently.
GROUPS = [
    ("Forward-pass few-shot", [
        ("SegGPT", "seggpt_k8", False),
        ("UniverSeg", "universeg_k8", False), ("INSID3", "insid3", False),
        ("Tyche", "tyche_k8", False),
        # Matcher is ONE-SHOT by design, so it is run at K=1 -- its own setting, not a handicap. That
        # makes its column the one place in this table where the support size differs, which the caption
        # has to say outright: a reader assuming K=8 everywhere would misread it as a weaker result than
        # it is. Included because the full panel now measures it at the same ten seeds as every other
        # column, and describing a measured baseline as merely "task-mismatched" would be dismissing by
        # definition what we can dismiss by measurement.
        ("Matcher$^{\\dagger}$", "matcher_k1", False),
    ]),
    ("Fitted on the support", [
        # `lean_k8` (was `stripv3_k8` until the method was simplified on 2026-09-01), NOT `oursv3n_k8`. The latter is best_v3 WITH the competitive gate and FiLM,
        # which the method no longer has (registry C65): it scores 0.791 against the reported
        # 0.789, so pointing here at it puts a different method in Table 1 from the one every
        # number in the prose describes -- the exact defect the ablation re-run was undertaken
        # to remove, reintroduced in the other table.
        ("Exemplar", "lean_k8", False), ("Prior-bank RF", "ilastik_k8", False),
        ("nnU-Net", "nnunet_k8", False),
    ]),
    ("Trained specialists", [
        ("Cellpose-SAM", "cellpose_sam", True), ("StarDist", "stardist", True),
        ("micro-SAM", "microsam", True),
    ]),
]
# The flat left-to-right column order. Every consumer below still sees plain (name, dir, det) triples,
# so the grouping is a presentation layer only -- it must not reach the numbers, the bolding or the
# completeness check, all of which stay defined over the whole panel.
COLS = [c for _, entries in GROUPS for c in entries]
# Column name -> bib key. Two columns are deliberately absent, and _head() keys off that absence
# rather than off a special case. "Exemplar" is ours, so it has nothing to cite. "Prior-bank RF"
# used to carry \cite{ilastik}, which read as ilastik's own score for a baseline the Baselines
# paragraph explicitly says is NOT ilastik: it is a random forest over OUR bank, deliberately
# strengthened. That paragraph carries the ilastik citation, in the sentence that draws the
# distinction; the header must not re-attribute the number.
# The citation came OFF this column on 2026-09-02 and went back ON the same day, and both moves
# were right. It came off while the column was our own read-out: a citation in a header reads as
# "these are that method's numbers", which the caption then had to deny. It went back when the
# column became the authors' released implementation, verified against their published anchor.
# "Prior-bank RF" still carries no citation, for the reason this comment originally recorded.
CITE = {"SegGPT": "seggpt", "UniverSeg": "universeg", "Tyche": "tyche", "INSID3": "insid3",
        "nnU-Net": "nnunet",
        "Matcher$^{\\dagger}$": "matcher", "Cellpose-SAM": "cellposesam", "StarDist": "stardist",
        "micro-SAM": "microsam"}
# Header, column spec, group headings and group rules are ALL derived from GROUPS. They were hardcoded
# while the ROWS came from COLS, so adding a column produced ten values under nine headings -- which
# LaTeX either refuses or, worse, absorbs by shifting every number one method to the left. A hand-typed
# \cmidrule span fails even more quietly: it compiles, and just underlines the wrong methods.
def _spans():
    """(label, first_col, last_col) per block, 1-based with the Dataset column occupying column 1."""
    out, c = [], 2
    for label, entries in GROUPS:
        out.append((label, c, c + len(entries) - 1))
        c += len(entries)
    return out


COLSPEC = "l " + " ".join("c" * len(entries) for _, entries in GROUPS)
GROUP_ROW = " & ".join([""] + [f"\\multicolumn{{{b - a + 1}}}{{c}}{{{lab}}}" for lab, a, b in _spans()])
GROUP_RULES = " ".join(f"\\cmidrule(lr){{{a}-{b}}}" for _, a, b in _spans())
# Citations sit on a second header line rather than beside the name. With eleven method columns the
# inline "~\cite{}" was part of why the tabular ran 112pt past the text block; stacking costs one row
# of height and buys back the width, and a reader looking for the reference still finds it in place.
def _head(n):
    name = n.replace("$^{\\dagger}$", "\\textsuperscript{$\\dagger$}")
    if n not in CITE:
        return name
    return (r"\begin{tabular}{@{}c@{}}" + name + r"\\[-1.5pt]\cite{" + CITE[n]
            + r"}\end{tabular}")


HEADER = "Dataset & " + " & ".join(_head(n) for n, _, _ in COLS)

# Off-the-shelf specialist dir -> its fine-tuned counterpart, for the caption's FT delta.
FT_OF = {"cellpose_sam": "cellpose_ft_k8", "stardist": "stardist_ft_k8", "microsam": "microsam_ft_k8"}


def check_seeds():
    """Every rendered cell must carry the seed count the caption asserts."""
    if not _SEEN_SEEDS:
        raise SystemExit(f"no cells resolved under {ROOT!r} -- refusing to overwrite the paper table")
    bad = {k: v for k, v in _SEEN_SEEDS.items() if v != EXPECT_SEEDS}
    if bad:
        raise ValueError(f"caption asserts {EXPECT_SEEDS} seeds, but {len(bad)} cell(s) differ: {bad}")


def specialist_won(cols):
    """Display names of the datasets where a trained specialist takes the row-best.

    The caption claims this is exactly the standard-cell datasets; deriving it means a regenerated
    number can never leave the caption asserting something the table no longer shows.
    """
    spec = {n for n, _, det in cols if det}
    won = []
    for ds, name, metric in DATASETS:
        present = {n: r[0] for n, d, _ in cols if (r := resolve(d, ds, metric)) is not None}
        if present and max(present, key=present.get) in spec:
            won.append(name)
    return won


def ft_delta(ds="monuseg", metric="fg_iou"):
    """(off-the-shelf, fine-tuned) for the caption's 'fine-tuning barely moves it' example, read from
    the tree rather than hand-typed. Returns None if the fine-tuned arm was not run."""
    off = resolve("cellpose_sam", ds, metric)
    ft = stat(FT_OF["cellpose_sam"], ds, metric)
    return (off[0], ft[0]) if off and ft else None


def _require_complete(cols):
    """Refuse to write when any column is missing any dataset.

    `cell()` renders "--" for an absent record, and a reader takes "--" as "not applicable", not as
    "this baseline may have won here". Deleting only the two records where a baseline beats us moves
    the bold onto our row, writes cleanly and exits 0, under a caption that says the bold is the best
    score in the row. The recorded final10 data loss makes a partially-resynced tree the live
    trigger, so a hole in the tree has to stop the table rather than decorate it.
    """
    holes = []
    for name, dirname, det in cols:
        missing = [ds for ds, _, metric in DATASETS if resolve(dirname, ds, metric) is None]
        if missing:
            holes.append(f"{name} ({dirname}) missing {missing}")
    if holes:
        raise SystemExit("REFUSING to write a table with missing cells -- a rendered \"--\" reads as "
                         "'not applicable' and silently hands the bold to whoever is left:\n  "
                         + "\n  ".join(holes))


def emit_merged():
    _require_complete(COLS)
    body = build_body(COLS)
    won = specialist_won(COLS)
    ft = ft_delta()
    # check_seeds AFTER ft_delta: the fine-tuned record behind the caption's delta is only
    # added to _SEEN_SEEDS by ft_delta(), so running the check first left that one record
    # unvalidated while the success line still printed 'all at N seeds' over the enlarged set.
    check_seeds()
    # The caption spells small counts as words, matching the manuscript's style, so the generated
    # file is byte-identical to what the paper compiles -- a reader who re-runs this script must get
    # the committed table back, not a near-miss they have to eyeball.
    _WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
             8: "eight", 9: "nine", 10: "ten", 11: "eleven"}
    # The caption no longer NAMES the datasets specialists win (the manuscript dropped them for
    # space), so the derivation would become decorative. Assert it instead: if the set ever changes,
    # this fails rather than printing a sentence the numbers no longer support.
    if sorted(won) != sorted(["DSB2018", "MoNuSeg", "CTC-U373"]):
        raise ValueError(f"main.tex says specialists lead on the three standard-cell datasets they "
                         f"were built for, but the numbers say {won or 'none'}")
    if ft is None:
        raise ValueError("main.tex cites the fine-tuned-specialist delta (Cellpose-SAM on MoNuSeg) "
                         "but no such record was found")
    # A CAPTION MAKES THE FLOAT READABLE ALONE; IT DOES NOT RE-ARGUE THE PAPER (user, 2026-07-29).
    # The previous caption restated six things the body already says: the seed count (Experiments),
    # that the specialists lead on the three standard-cell datasets and the Cellpose fine-tuning
    # delta (both in the specialist paragraph, with the same numbers), that the whole nnU-Net gap is
    # topology (the nnU-Net paragraph), and the entire forward-pass/fitting taxonomy including
    # ilastik-RF's steelman provenance (the Baselines paragraph). On a four-page format that is paid
    # for twice, in reader patience and in the page budget that then squeezes real content.
    #
    # What stays is only what a reader cannot get from the table itself: the metric and K, the seed
    # count and why the specialist columns carry no deviation, what bold spans, that the Mean mixes
    # two metrics, INSID3's oracle mode choice, the dagger key, and how specialist foreground is
    # defined. Each of those stops a number being read as something it is not.
    #
    # The two guards above are KEPT even though their sentences moved into main.tex: they verify
    # claims the paper still makes, and a claim is no less worth checking for living in the prose.
    # Short caption adopted verbatim from the co-author's Overleaf revision (2026-08-10) and
    # kept HERE rather than only in the .tex, so a regeneration cannot silently restore the
    # long version. Seed count still comes from EXPECT_SEEDS.
    caption = (
        rf"Per-dataset performance at $K{{=}}8$ over {EXPECT_SEEDS} seeds using each dataset's "
        r"designated metric. Bold denotes the best result per row. The Mean combines "
        r"seven foreground-IoU and four centreline-Dice datasets, also reported separately. "
        r"$^{\dagger}$Matcher uses its documented $K{=}1$ setting. INSID3 is the authors' released implementation, whose published anchor reproduces here; see Baselines."
    )
    tex = (
        r"""% tab:fewshot — MERGED main results: ours vs few-shot in-context baselines AND trained specialists,
% 11 datasets, FLAT, in three blocks by how each method uses the K support masks (forward pass / fitted
% on the support / no support at all). Generated by scripts/make_semantic_tables.py from the clean
% campaign tree; do not hand-edit values. Bold = best in the row across ALL THREE blocks.
\begin{table*}[t]
\centering
\caption{""" + caption + r"""}
\label{tab:fewshot}
{\scriptsize\setlength{\tabcolsep}{1.2pt}
\begin{tabular}{""" + COLSPEC + r"""}
\toprule
""" + GROUP_ROW + r""" \\
""" + GROUP_RULES + r"""
""" + HEADER + r""" \\
\midrule
""" + body + "\n" + mean_row(COLS) + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""
    )
    open(os.path.join(OUT, "tab_fewshot.tex"), "w").write(tex)
    print(f"wrote tab_fewshot.tex ({len(_SEEN_SEEDS)} cells, all at {EXPECT_SEEDS} seeds; "
          f"specialists lead on {won})")


if __name__ == "__main__":
    emit_merged()
