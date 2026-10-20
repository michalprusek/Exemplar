# Copied from AutoSeg scripts/make_semantic_tables.py (cb71f85) by tools/sync_paper.py; default paths adapted, internal notes rewritten in comments and docstrings.
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
  ASG_SEM_TREE=/path/to/results/final10 \
  ASG_SEM_OUT=<paper dir> python paper/make_semantic_tables.py
"""
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.environ["ASG_SEM_TREE"]                      # dir holding {method}_k{K}/ and {specialist}/
OUT = os.environ.get("ASG_SEM_OUT", ".")
# The caption states this; every rendered cell is checked against it.
EXPECT_SEEDS = int(os.environ.get("ASG_SEM_SEEDS", "10"))
# The support size the table is built at. 8 is the paper's Table 1 and the default, and at 8 the
# output is byte-identical to what this script wrote before the parameter existed. Any other value
# writes `tab_fewshot_k{K}.tex`, a drop-in for the same \label, and never touches `tab_fewshot.tex`.
# Added to render the table at one mask as well.
K = int(os.environ.get("ASG_SEM_K", "8"))
# Which specialist records Table 1 prints.
#   "ft"   each specialist fine-tuned on the eight support masks of every dataset, ten seeds: the
#          default, so that every column of the table has used the same annotation.
#   "ots"  as its authors released it, with no support masks: what the table printed until then.
# Fine-tuned records exist at K=8 only, so any other K falls back to "ots" and says so.
#   "ftsem" the same fine-tunes scored on the network's own foreground map:
#          the default since 2026-10-12. Every metric of this table is semantic, and the instance
#          read-out understated these tools by up to 0.87 on thin structures. In this mode the table is
#          laid out by what each method does with the eight masks: forward pass / a head fitted on
#          frozen features / a network trained on the support.
SPEC_MODE = os.environ.get("ASG_SPEC_MODE", "ftsem")
if SPEC_MODE not in ("ftsem", "ft", "ots"):
    raise SystemExit(f"ASG_SPEC_MODE must be 'ftsem', 'ft' or 'ots', not {SPEC_MODE!r}")
if K != 8 and SPEC_MODE in ("ft", "ftsem"):
    print(f"  ! K={K}: no fine-tuned specialist table at this K, printing them off the shelf", file=sys.stderr)
    SPEC_MODE = "ots"
FT_MODES = ("ft", "ftsem")
SPECIALISTS = ("Cellpose-SAM", "StarDist", "micro-SAM")
# At ONE mask each fine-tuned specialist ran at two budgets: the rule
# in epochs over the support, and the same number of gradient steps as at eight masks. The row prints
# the budget with the higher mean over the eleven datasets, a selection in the baseline's favour that
# the caption states. Both directories must be complete or the table is not written.
# The score-directory stem of OUR column (`<stem>_k{K}` under ASG_SEM_TREE). `lean` is the method the paper
# reported until 2026-10-14; `exemplar` is the method decided that day: the
# fine view at the magnification decided from the support masks, registered to the patch grid, block 12 read
# beside the last. The K=1 row derives its directory from the same stem, so the column never mixes methods.
OURS_STEM = os.environ.get("ASG_OURS_STEM", "lean")
# Each metric group ends with its own mean, so the table shows on which group our method is better. Block layout only;
# the older layouts keep their rows.
SPLIT = SPEC_MODE == "ftsem"
# ASG_SEM_NO_K1=1 leaves out the panel mean at ONE mask (and its caption sentence). First used while our K=1 arm was
# still running (the row must never print another method's K=1 under our header); the paper's Table 1 leaves the
# one-mask row out (Fig. 2 shows K = 1), so the exemplar table is generated with it.
NO_K1 = os.environ.get("ASG_SEM_NO_K1", "") == "1"
# ASG_SEM_FIT_ROW=1: a last row with each
# column's fit time from the eight masks on one RTX A5000, from the smallest to the largest images (DSB2018 to HRF). Its
# cells are the macros of numbers_timing.tex (make_timing_numbers.py), never numbers, so the row and the prose cannot
# round differently. A forward-pass column fits nothing and prints "--". Any other column without a timing stops the
# table, unless it is named in ASG_SEM_FIT_UNTIMED (comma-separated column names): it then prints "not timed".
# ASG_TYCHE_STEM=tyche_bestof: Tyche scored as in its paper, the one of its 16 candidates that best matches the test
# mask by the dataset's metric (scripts/tyche_bench.py). Default `tyche` = the mean of the candidates, the earlier
# column.
TYCHE_STEM = os.environ.get("ASG_TYCHE_STEM", "tyche")
FIT_ROW = os.environ.get("ASG_SEM_FIT_ROW", "") == "1"
FIT_UNTIMED = [s for s in os.environ.get("ASG_SEM_FIT_UNTIMED", "").split(",") if s]
# ASG_SEM_FITINF=1: the last row prints, for EVERY column, the fit time and the inference time per image on the RTX
# A5000, each the median over the eleven datasets (scripts/c119/timing_table.py -> numbers_fitinf.tex), "--" for the
# fit of a forward-pass method. Overrides the DSB2018-to-HRF fit row.
FITINF = os.environ.get("ASG_SEM_FITINF", "") == "1"
FITINF_KEYS = {"SegGPT": "SegGPT", "UniverSeg": "UniverSeg", "INSID3": "INSID", "Tyche": "Tyche", "Exemplar": "Exemplar",
               "Prior-bank RF": "PriorRF", "ilastik": "ilastik", "Convpaint": "Convpaint", "nnU-Net": "nnUNet",
               "Cellpose-SAM": "CellposeSAM", "StarDist": "StarDist", "micro-SAM": "microSAM"}
FIT_CELLS = {"Exemplar": (r"\fitSmall--\fitLarge\,s", ("fitSmall", "fitLarge")),
             "nnU-Net": (r"\nnSmallMin--${>}$\nnLargeMin\,min", ("nnSmallMin", "nnLargeMin")),
             "Cellpose-SAM": (r"\cpFitSmall--\cpFitLarge\,min", ("cpFitSmall", "cpFitLarge")),
             "StarDist": (r"\sdFitSmall--\sdFitLarge\,min", ("sdFitSmall", "sdFitLarge")),
             "Convpaint": (r"\cvpFitSmall--\cvpFitLarge\,min", ("cvpFitSmall", "cvpFitLarge")),
             "micro-SAM": (r"\msFitSmall--\msFitLarge\,min", ("msFitSmall", "msFitLarge"))}
# ASG_SEM_SIGMARKS=1 (off by default; the paper's table does not use it): a superscript asterisk on
# every baseline cell that differs from Exemplar at a Holm-adjusted p < 0.05 in either direction, from the same families
# the Results report (make_stats.table1_significance). Untested columns carry none, and the caption says which.
SIGMARKS = os.environ.get("ASG_SEM_SIGMARKS", "") == "1"
SIG = {}
if SIGMARKS:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import make_stats  # noqa: E402  (reads ASG_SEM_TREE / ASG_OURS_STEM / ASG_TYCHE_STEM like this script)
    if make_stats.OURS_STEM != OURS_STEM or make_stats.TYCHE_STEM != TYCHE_STEM:
        raise ValueError("make_stats and the table read different arms; the marks would test another column")
    SIG = make_stats.table1_significance()
    SIG_TESTED = set(make_stats.FORWARD_PASS) | {"nnU-Net", "Cellpose-SAM"} | (
        {"Convpaint"} if os.environ.get("ASG_SEM_CONVPAINT", "") == "1" else set())
def _base(n):
    """A column name without its header mark (Tyche$^{\\dagger}$ -> Tyche), as make_stats names it."""
    return re.sub(r"\$\^\{[^}]*\}\$", "", n)


FTSEM_K1 = {"Cellpose-SAM": ("cellpose_ft11sem_k1_e800", "cellpose_ft11sem_k1_e6400"),
            "micro-SAM": ("microsam_ft11sem_k1_e50", "microsam_ft11sem_k1_e400"),
            "StarDist": ("stardist_ft11sem_k1_e1600", "stardist_ft11sem_k1_e12800")}
K1_CHOICE = {}


def kdir(stem, k=None):
    """Score directory of a K-dependent arm. nnU-Net's one-mask records live beside the tree, not
    in it, which is where make_stats.py, the audit and the K-scaling figure already read them."""
    k = K if k is None else k
    if stem == "nnunet" and k == 1:
        return "../nnunet_k1"
    return f"{stem}_k{k}"

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
    ("fisbe", "FISBe", "cldice"),
]
# ASG_DROP_DS=ctc_u373 (the paper's setting: CTC-U373 is not reported, because its conditions need the CTC
# organisers' permission for public use outside the challenge): the datasets named here are left out of every row and mean.
_DROP = {x for x in os.environ.get("ASG_DROP_DS", "").split(",") if x}
DATASETS = [d for d in DATASETS if d[0] not in _DROP]

# (dirname, dataset) -> seed count of the record actually rendered; checked for uniformity before write.
_SEEN_SEEDS = {}
# (dirname, ds) -> (seeds, per-seed mean over the test images), for the deviation of the summary rows
_SEED_MEANS = {}


def stat(dirname, ds, metric):
    """(mean, std, n_seeds) on the semantic metric for one (method-dir, dataset), or None if absent.

    A directory must resolve to exactly ONE metric-matching record. Two generations of a method name in
    one dir (this repo has already renamed its best method once: ``head_fusion_best_cgate_film`` ->
    ``..._nobank``) would otherwise let a silent max() report whichever scored higher --
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
        matches.append((f, (float(sm.mean()), float(sm.std(ddof=1)) if ns > 1 else float("nan"), ns),
                        (tuple(d["seeds"]), sm)))
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
                             f"{[os.path.basename(f) for f, _, _ in matches]} -- ambiguous, refusing to guess")
        matches = based
    _SEEN_SEEDS[(dirname, ds)] = matches[0][1][2]
    _SEED_MEANS[(dirname, ds)] = matches[0][2]
    return matches[0][1]


def insid3(ds, metric):
    """INSID3 -- the AUTHORS' released implementation, replacing our own read-out (2026-09-02).

    Until now this column was a training-free correspondence read-out we wrote ourselves following
    INSID3's premise, and the caption had to say so. It is now their code (github.com/visinf/INSID3),
    run through `scripts/insid3_bench.py` on our split, our seeds and the same K=8 draw, with the
    CRF refinement their published Chest X-ray number reproduces from -- 78.7 against their 78.8.
    Their code default is bilinear and would understate them by 1.5 points.

    The old two-mode steelman (`insid3_guided_k8` / `insid3_dense_k8`, better of two CRF settings
    per dataset) is gone with the construction it steelmanned: there is one configuration now, the
    one the authors report.
    """
    return stat(kdir("insid3"), ds, metric)


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


def mark(cell_tex, underline):
    """Underline a rendered cell (the best in the row, in the block layout where bold is per block)."""
    return r"\underline{" + cell_tex + "}" if underline and cell_tex != "--" else cell_tex


def render(v, is_best, sig=False):
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
    if SIGMARKS:
        # The asterisk needs room: at full width it widened seven columns and the table ran 20pt past the text block,
        # and set with zero width it touched the next column's number. Every score is below one, so the means drop
        # their leading zero as the deviations already do, which gives the width back (2026-10-17).
        if not m.strip().startswith("0."):
            raise ValueError(f"mean {m!r} is not below one; the short form would misprint it")
        m = m.replace("0.", ".", 1)
    core = (f"{{\\boldmath${m}$}}" if is_best else f"${m}$") + (r"$^{*}$" if sig else "")
    # With fine-tuned specialists three more columns carry a deviation, and the tabular ran 27pt past
    # the text block. A deviation is always below one here, so its leading zero carries nothing:
    # ".014" reads as 0.014 and gives back one digit per cell, which is the 27pt. The MEAN keeps its
    # zero, because means are what a reader compares and copies.
    if SPEC_MODE in FT_MODES:
        if "0." not in s:
            raise ValueError(f"deviation {s!r} is not below one; the short form would misprint it")
        s = s.replace("0.", ".", 1)
    if SPEC_MODE == "ftsem":
        # The deviation is set at the table's own 7 pt instead of \tiny (5 pt); the width comes from dropping the thin
        # space before it, a tabcolsep of 0.8 pt and the fit row's bare slash. The tabular stays inside the text block.
        return core + r"{\scriptsize$\pm" + s + "$}"
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


def agg(dirname, subset=None):
    """(mean, deviation) of a summary row: the mean over the datasets (all, or one metric group) and its standard
    deviation over seeds, the datasets averaged within each seed first, as in Table 2 (the Mean and the group means
    get a deviation too). None when the column misses a dataset."""
    rows = [(ds, m) for ds, _, m in DATASETS if subset is None or m == subset]
    vals = [resolve(dirname, ds, m) for ds, m in rows]
    if any(v is None for v in vals):
        return None
    key = kdir("insid3") if dirname == "insid3" else dirname
    seeds = {_SEED_MEANS[(key, ds)][0] for ds, _ in rows}
    if len(seeds) != 1:
        raise ValueError(f"{dirname}: the datasets were drawn with different seed lists {seeds}; no per-seed mean")
    per_seed = np.mean([_SEED_MEANS[(key, ds)][1] for ds, _ in rows], axis=0)
    return sum(v[0] for v in vals) / len(vals), float(per_seed.std(ddof=1))


def agg_tex(m, s):
    """The summary cell in the body's format (the deviation below 0.0005 printed as a bound, as in `cell`)."""
    return f"{m:.3f}\\pm{{<}}0.001" if s < 5e-4 else f"{m:.3f}\\pm{s:.3f}"


def _subset_mean(dirname, subset):
    """Mean over a metric-defined subset, or None if the column is missing any of it."""
    vals = [resolve(dirname, ds, m) for ds, _, m in DATASETS if m == subset]
    if any(v is None for v in vals):
        return None
    return sum(v[0] for v in vals) / len(vals)


# THE TWO METRIC-SPLIT ROWS were removed on 2026-10-09 and came back on 2026-10-14 (both the first
# author's decisions), now as the last row of each metric group rather than under the panel mean, so a
# reader sees each group's mean directly under its datasets. The panel mean at ONE mask stays last.
GROUP_LABEL = {"fg_iou": r"\textit{Overlap}", "cldice": r"\textit{Centreline}"}   # short: the label column is no wider than ISBI2012-EM


def subset_row(cols, subset):
    """The mean over one metric group, marked like every other row (best per block bold, row best
    underlined). `_require_complete` has already refused a column with a missing dataset."""
    means = {n: _subset_mean(d, subset) for n, d, _ in cols}
    if any(v is None for v in means.values()):
        raise SystemExit(f"REFUSING to write the {subset} mean row: {[n for n, v in means.items() if v is None]} incomplete")
    means = {n: round(v, 3) for n, v in means.items()}        # rounded as printed (see mean_row)
    overall, per_block = _bests(means, cols)
    tex = {n: agg_tex(*agg(d, subset)) for n, d, _ in cols} if SPLIT else {n: f"{means[n]:.3f}" for n in means}
    cells = [render_cell(n, tex[n], means[n], overall, per_block) for n, _, _ in cols]
    return GROUP_LABEL[subset] + " & " + " & ".join(cells) + r" \\"


def split_fp(dirname, ds, metric):
    """Split fingerprint of the one metric-matching record, or None. `stat` returns numbers only."""
    fps = {json.load(open(f)).get("split_fp")
           for f in sorted(glob.glob(os.path.join(ROOT, dirname, f"*__{ds}.json")))
           if json.load(open(f)).get("metric") == metric}
    return fps.pop() if len(fps) == 1 else None


def require_one_split(dirs, what):
    """Every column of a row must have scored the same test images.

    The table sets columns side by side and bolds the best, which is a comparison; `stat` never
    looked at the fingerprint, so a column computed on a different split would have printed without
    complaint. A record that carries no fingerprint is reported, not waved through.
    """
    bad = []
    for ds, _, metric in DATASETS:
        fps = {d: split_fp(d, ds, metric) for d in dirs}
        if len(set(fps.values())) != 1 or None in fps.values():
            bad.append(f"{ds}: {fps}")
    if bad:
        raise ValueError(f"{what}: columns were not scored on one test split:\n  " + "\n  ".join(bad))


def k1_dirs(cols):
    """The K=1 score directory behind each column of the K=8 table, in column order.

    A column whose method takes no support masks (the specialists off the shelf) has no K: its
    directory is the same one, and its mean at one mask is the mean it has at eight. Matcher is at
    K=1 already. A FINE-TUNED specialist has no record at one mask, and its entry is None: the row
    prints a dash there. It must never fall back to the off-the-shelf mean, which would put another
    method's number under a fine-tuned header.
    """
    out = []
    for name, d, det in cols:
        if name in SPECIALISTS and SPEC_MODE == "ftsem":
            out.append(k1_choice(name))
        elif name in SPECIALISTS and SPEC_MODE == "ft":
            out.append(None)
        elif det or d == "matcher_k1":
            out.append(d)
        elif d == "insid3":
            out.append(kdir("insid3", 1))
        else:
            stem = d.rsplit("_k", 1)[0]
            out.append(kdir(stem, 1))
    return out


def k1_choice(name):
    """The K=1 directory of a fine-tuned specialist: the budget with the higher eleven-dataset mean."""
    if name in K1_CHOICE:
        return K1_CHOICE[name]
    means = {}
    for d in FTSEM_K1[name]:
        m = col_mean(d)
        if m is None:
            raise SystemExit(f"REFUSING to write the K=1 row: {name} at {d} is missing datasets")
        means[d] = m
    pick = max(means, key=means.get)
    print(f"  K=1 {name}: " + ", ".join(f"{d} {m:.4f}" for d, m in means.items()) + f" -> {pick}", file=sys.stderr)
    K1_CHOICE[name] = pick
    return pick


def _bests(values, cols):
    """(overall best value, {block label: best value in that block}) over the numeric entries."""
    nums = {n: v for n, v in values.items() if v is not None}
    overall = max(nums.values()) if nums else None
    per_block = {}
    for n, v in nums.items():
        b = BLOCK[n]
        per_block[b] = max(per_block.get(b, v), v)
    return overall, per_block


def render_cell(n, v_tex, v_num, overall, per_block, sig=False):
    """Bold the best in the row (old layouts), or the best in each block, with the row's best
    underlined (block layout). A reader then sees both who leads the row and who leads its regime."""
    if v_num is None:
        return "--"
    if SPEC_MODE == "ftsem":
        tex = render(v_tex, abs(v_num - per_block[BLOCK[n]]) < 1e-9, sig)
        return mark(tex, abs(v_num - overall) < 1e-9)
    return render(v_tex, abs(v_num - overall) < 1e-9, sig)


def mean_row(cols):
    """The aggregation rows under the table's last rule, best in bold: the panel mean, and at K=8
    the panel mean at ONE support mask beneath it."""
    # rounded as printed, so two columns that print the same value are both marked, as in the dataset rows (whose
    # values are read back from the rendered cells)
    means = {n: (None if v is None else round(v, 3)) for n, d, _ in cols for v in [col_mean(d)]}
    overall, per_block = _bests(means, cols)
    cells = [render_cell(n, (agg_tex(*agg(d)) if SPLIT else f"{means[n]:.3f}") if means[n] is not None else "--",
                         means[n], overall, per_block) for n, d, _ in cols]
    out = r"\midrule" + "\n" + r"\textbf{Mean} & " + " & ".join(cells) + r" \\"
    if K != 8 or NO_K1:
        return out
    # The mean at one mask. Fail-loud like the body: a column with a missing dataset gets no
    # number, and a number that is not there must stop the table, not print as a dash under a
    # caption that calls the row a mean.
    dirs = k1_dirs(cols)
    require_one_split([d for d in dirs if d is not None], "the K=1 mean row")
    m1 = []
    for (n, _, _), d in zip(cols, dirs):
        if d is None:                      # a fine-tuned specialist: not run at one mask
            m1.append(None)
            continue
        v = col_mean(d)
        if v is None:
            raise SystemExit(f"REFUSING to write the K=1 mean row: {n} ({d}) is missing datasets")
        m1.append(v)
    vals1 = {n: v for (n, _, _), v in zip(cols, m1)}
    overall1, per_block1 = _bests(vals1, cols)
    cells1 = [render_cell(n, f"{v:.3f}" if v is not None else "--", v, overall1, per_block1)
              for (n, _, _), v in zip(cols, m1)]
    return out + "\n" + r"Mean, $K{=}1$ & " + " & ".join(cells1) + r" \\"


def fit_row(cols):
    """The fit-time row (ASG_SEM_FIT_ROW): macros only, each one checked to be defined in OUT/numbers_timing.tex, so
    a table can never reach the paper with a cell that compiles to nothing or to another method's time."""
    fp = {n for lab, entries in GROUPS if lab.startswith("Forward pass") for n, _, _ in entries}
    slash = "/" if SPEC_MODE == "ftsem" else r"\,/\,"    # bare in ftsem: its width pays for the 7 pt deviations
    if FITINF:                    # every column: median fit / median inference per image (c119/timing_table.py)
        path = os.path.join(OUT, "numbers_fitinf.tex")
        defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", open(path).read())) if os.path.exists(path) else set()
        cells, missing = [], []
        for n, _, _ in cols:
            key = FITINF_KEYS.get(_base(n))          # the header may carry a mark (Tyche$^{\dagger}$)
            need = ([] if n in fp else [f"tfit{key}"]) + [f"tinf{key}"]
            if key is None or any(m not in defined for m in need):
                missing.append(f"{n}: {need if key else 'no FITINF_KEYS entry'}")
                continue
            cells.append(("--" if n in fp else rf"\tfit{key}") + slash + rf"\tinf{key}")
        if missing:
            raise SystemExit(f"REFUSING to write the fit / inference row ({path}):\n  " + "\n  ".join(missing))
        return "\n" + r"\midrule" + "\n" + r"Fit\,/\,image & " + " & ".join(cells) + r" \\"
    path = os.path.join(OUT, "numbers_timing.tex")
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", open(path).read())) if os.path.exists(path) else set()
    cells, missing = [], []
    for n, _, _ in cols:
        if n in fp:
            cells.append("--")
        elif n in FIT_UNTIMED:       # named explicitly: wins over a FIT_CELLS entry whose timing is not measured yet
            cells.append("not timed")
        elif n in FIT_CELLS:
            tex, macros = FIT_CELLS[n]
            missing += [f"{n}: \\{m}" for m in macros if m not in defined]
            cells.append(tex)
        else:
            missing.append(f"{n}: no timing and not in ASG_SEM_FIT_UNTIMED")
    if missing:
        raise SystemExit(f"REFUSING to write the fit-time row ({path}):\n  " + "\n  ".join(missing))
    return "\n" + r"\midrule" + "\n" + r"Fit time & " + " & ".join(cells) + r" \\"


def build_body(cols):
    """One row per dataset, FLAT. ``cols`` = [(name, dir, det)]; the best score per row is bolded on the
    numeric mean across ALL methods (few-shot + specialists), not per group."""
    out = []
    for i, (ds, name, metric) in enumerate(DATASETS):
        raw = {n: cell(d, ds, metric, det) for n, d, det in cols}
        means = {n: float(v.split("\\pm")[0].strip("{}$\\boldmath ")) for n, v in raw.items() if v != "--"}
        overall, per_block = _bests({n: means.get(n) for n, _, _ in cols}, cols)
        cells = [render_cell(n, raw[n], means.get(n), overall, per_block, ds in SIG.get(_base(n), ())) for n, _, _ in cols]
        out.append(f"{name:12} & " + " & ".join(cells) + r" \\")
        last_of_group = i + 1 == len(DATASETS) or DATASETS[i + 1][2] != metric
        if SPLIT and last_of_group:
            out += [r"\midrule", subset_row(cols, metric)]
            if i + 1 < len(DATASETS):                      # the panel mean below brings its own rule
                out.append(r"\midrule")
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
MATCHER = "Matcher" if K == 1 else "Matcher$^{\\dagger}$"
# ASG_SEM_NO_MATCHER=1 drops the Matcher column with its dagger and caption key: it was the only column run at another
# support size than the table's, so the caption's "Every method uses the same masks" was false while it stood, and it
# was the lowest column on every row.
NO_MATCHER = os.environ.get("ASG_SEM_NO_MATCHER", "") == "1"
# Tyche scored by its best candidate against the test mask (ASG_TYCHE_STEM=tyche_bestof) is the most favourable condition
# in the table, so its header carries a mark and the caption a key (an external review asked for it): an oracle number
# must be visible where a reader scans the columns, not only in Baselines.
# ASG_SEM_NO_TYCHE_MARK=1 (the caption repeated the Baselines paragraph): no mark on Tyche's header and no caption
# note; the text says how Tyche is scored.
TYCHE_MARK = "" if os.environ.get("ASG_SEM_NO_TYCHE_MARK", "") == "1" else \
    (("$^{\\dagger}$" if NO_MATCHER else "$^{\\ddagger}$") if TYCHE_STEM == "tyche_bestof" else "")
TYCHE = "Tyche" + TYCHE_MARK
# ASG_SEM_CONVPAINT=1 (Prior-bank RF stays, Convpaint is added beside it): Convpaint (napari-convpaint 1.1.0,
# combo_dino_vgg) as a column of the "fitted without gradient descent" block, score dir final10/convpaint_k8.
CONVPAINT = os.environ.get("ASG_SEM_CONVPAINT", "") == "1"
# ASG_SEM_ILASTIK=1 (the paper's setting): the REAL ilastik 1.4.2 (final10/ilastik142_k8,
# mean 0.634) takes the place of Prior-bank RF (final10/ilastik_k8, 0.595), which understates the ilastik workflow it
# stood in for. Prior-bank RF keeps its Table 2 row ("Forest on the priors").
ILASTIK = os.environ.get("ASG_SEM_ILASTIK", "") == "1"
_RF_COL = ("ilastik", kdir("ilastik142"), False) if ILASTIK else ("Prior-bank RF", kdir("ilastik"), False)
_FP_BLOCK = [
        ("SegGPT", kdir("seggpt"), False),
        ("UniverSeg", kdir("universeg"), False), ("INSID3", "insid3", False),
        (TYCHE, kdir(TYCHE_STEM), False),
] + ([] if NO_MATCHER else [(MATCHER, "matcher_k1", False)])
if SPEC_MODE == "ftsem":
    # The block layout: every column used the same
    # eight masks, and the blocks say what each method does with them. nnU-Net sits with the specialists
    # because both train a whole network on the support.
    GROUPS = [
        ("Forward pass, nothing fitted", _FP_BLOCK),
        # the ridge (since 2026-10-15) is solved, not trained; the random forest is grown on the priors
        ("Head fitted on frozen features" if not OURS_STEM.startswith("ridge") else "Fitted without gradient descent", [
            ("Exemplar", kdir(OURS_STEM), False), _RF_COL,
        ] + ([("Convpaint", kdir("convpaint"), False)] if CONVPAINT else [])),
        ("Network trained on the support", [
            ("nnU-Net", kdir("nnunet"), False),
            ("Cellpose-SAM", "cellpose_ft11sem_k8", False), ("StarDist", "stardist_ft11sem_k8", False),
            ("micro-SAM", "microsam_ft11sem_k8", False),
        ]),
    ]
else:
  GROUPS = [
    ("Forward-pass few-shot", [
        ("SegGPT", kdir("seggpt"), False),
        ("UniverSeg", kdir("universeg"), False), ("INSID3", "insid3", False),
        (TYCHE, kdir(TYCHE_STEM), False),
        # Matcher is ONE-SHOT by design, so it is run at K=1 -- its own setting, not a handicap. That
        # makes its column the one place in this table where the support size differs, which the caption
        # has to say outright: a reader assuming K=8 everywhere would misread it as a weaker result than
        # it is. Included because the full panel now measures it at the same ten seeds as every other
        # column, and describing a measured baseline as merely "task-mismatched" would be dismissing by
        # definition what we can dismiss by measurement.
        # At K=1 every column is at Matcher's own setting, so the dagger and its caption key go.
        (MATCHER, "matcher_k1", False),
    ]),
    ("Fitted on the support", [
        # `lean_k8` (was `stripv3_k8` until the method was simplified on 2026-09-01), NOT `oursv3n_k8`. The latter is best_v3 WITH the competitive gate and FiLM,
        # which the method no longer has: it scores 0.791 against the reported
        # 0.789, so pointing here at it puts a different method in Table 1 from the one every
        # number in the prose describes -- the exact defect the ablation re-run was undertaken
        # to remove, reintroduced in the other table.
        ("Exemplar", kdir(OURS_STEM), False), ("Prior-bank RF", kdir("ilastik"), False),
        ("nnU-Net", kdir("nnunet"), False),
    ]),
    # Since 2026-10-10 the specialist columns print the model FINE-TUNED on the same eight support
    # masks, so every column of the table has used the same annotation. Their cells
    # then carry a deviation over the ten support draws like every other column. Off the shelf
    # ("ots") the same tools take no support and are deterministic, which `cell` verifies.
    ("Fine-tuned specialists", [
        ("Cellpose-SAM", "cellpose_ft11_k8", False), ("StarDist", "stardist_ft11_k8", False),
        ("micro-SAM", "microsam_ft11_k8", False),
    ]) if SPEC_MODE == "ft" else
    ("Trained specialists", [
        ("Cellpose-SAM", "cellpose_sam", True), ("StarDist", "stardist", True),
        ("micro-SAM", "microsam", True),
    ]),
  ]
BLOCK = {n: label for label, entries in GROUPS for n, _, _ in entries}
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
CITE = {"SegGPT": "seggpt", "UniverSeg": "universeg", TYCHE: "tyche", "INSID3": "insid3",
        "nnU-Net": "nnunet", "Convpaint": "convpaint", "ilastik": "ilastik",
        MATCHER: "matcher", "Cellpose-SAM": "cellposesam", "StarDist": "stardist",
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
# Our own column carries "(ours)" where the others carry a citation, and its name is set bold, so a
# reader finds it among eleven methods without scanning for the one header with no reference.
# Bold in the BODY still means best in the row; the caption says so, and a header is not a result.
OURS = "Exemplar"


def _head(n):
    name = n.replace("$^{\\dagger}$", "\\textsuperscript{$\\dagger$}")
    if n == OURS:
        return r"\begin{tabular}{@{}c@{}}\textbf{" + name + r"}\\[-1.5pt](ours)\end{tabular}"
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
    spec = set(SPECIALISTS)
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
    require_one_split([d if d != "insid3" else kdir("insid3") for _, d, _ in COLS], f"the K={K} table")
    # Built here and not inside the template below: its K=1 cells register their seed counts, and
    # check_seeds() must see them.
    aggregate = mean_row(COLS)
    if FIT_ROW:
        if SPEC_MODE != "ftsem" or K != 8:
            raise SystemExit("the fit-time row exists only in the K=8 table of fine-tuned specialists (ftsem)")
        aggregate += fit_row(COLS)
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
    # Both guards check sentences main.tex makes about the K=8 table, so they bind at K=8 only. At
    # another K the set is printed below instead: it is what the prose would have to say.
    if K == 8 and SPEC_MODE != "ftsem" and sorted(won) != sorted(["DSB2018", "MoNuSeg", "CTC-U373"]):
        raise ValueError(f"main.tex says specialists lead on the three standard-cell datasets they "
                         f"were built for, but the numbers say {won or 'none'}")
    # Only the off-the-shelf table needs the July three-dataset fine-tune behind it: the text beside
    # THAT table quoted its delta. The fine-tuned table prints the fine-tuned scores themselves.
    if K == 8 and SPEC_MODE == "ots" and ft is None:
        raise ValueError("main.tex cites the fine-tuned-specialist delta (Cellpose-SAM on MoNuSeg) "
                         "but no such record was found")
    # A CAPTION MAKES THE FLOAT READABLE ALONE; IT DOES NOT RE-ARGUE THE PAPER.
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
    # The short caption is kept HERE rather than only in the .tex, so a regeneration cannot
    # silently restore the long version. Seed count still comes from EXPECT_SEEDS.
    # The sentence "INSID3 is the authors' released implementation, whose published anchor
    # reproduces here; see Baselines" left the caption on 2026-10-09. The Baselines
    # paragraph says both things, so the caption was saying them a second time.
    if SPEC_MODE == "ftsem":
        # One sentence per thing a reader cannot get from the table: what the numbers are, what bold and
        # underline mean, what the two summary rows average, how the specialists are read out, and the
        # one selection made in a baseline's favour (its K=1 training budget).
        caption = (
            rf"Scores with $K{{=}}{K}$ support masks, mean $\pm$ standard deviation over {EXPECT_SEEDS} "
            r"seeds. "   # 2026-10-17: "(in the mean rows, of the dataset average)" left (reviewer: opaque; page budget)
            r"The upper datasets are scored by foreground IoU (overlap), the lower by centreline Dice"
            r". Bold: best "   # 2026-10-17: "each group ends with its own mean" left (page budget; the italic rows say it)
            r"in each group of methods; underline: best in the row. The Mean averages "
            + (rf"all {_WORD[len(DATASETS)]} datasets. " if NO_K1 else
               rf"all {_WORD[len(DATASETS)]} datasets; the last row gives it at $K{{=}}1$, where each specialist is shown at the "
               r"better of two training budgets. ")
            # (ridge, 2026-10-15: "Specialists are scored on their foreground probability map" is said in Baselines; the
            # page budget moves it out of the caption)
            + ("" if OURS_STEM.startswith("ridge") else r"Specialists are scored on their foreground probability map.")
            + ("" if NO_MATCHER else r" $^{\dagger}$Matcher uses its documented $K{=}1$ setting.")
            + (f" {TYCHE_MARK}Best of 16 candidates against the test mask." if TYCHE_MARK else "")
            + ((r" $^{*}$Differs significantly from Exemplar (Wilcoxon, Holm); not tested: "
                + ", ".join(_base(n) for n, _, _ in COLS if _base(n) != "Exemplar" and _base(n) not in SIG_TESTED) + ".")
               if SIGMARKS else "")
            # what the fit-time row shows; how each method is clocked is said in the Implementation paragraph
            + ((r" Last row: median over the datasets of the fit time\,/\,time per image on one RTX A5000 (first support draw, five test"
                r" images), encoding included." if FITINF else
                r" Last row: fit time on an RTX A5000, DSB2018 to HRF.") if FIT_ROW else "")
        )
    else:
      caption = (
        rf"Per-dataset performance at $K{{=}}{K}$ over {EXPECT_SEEDS} seeds using each dataset's "
        r"designated metric. Bold denotes the best result per row. The Mean combines "
        r"seven foreground-IoU and four centreline-Dice datasets"
        + ((r"; the last row gives it at $K{=}1$, where the specialists were not fine-tuned. "
            if SPEC_MODE == "ft" else r"; the last row gives it at $K{=}1$. ") if K == 8 else ". ")
        + ("" if K == 1 else r"$^{\dagger}$Matcher uses its documented $K{=}1$ setting.")
      )
    tex = (
        r"""% tab:fewshot — MERGED main results: ours vs few-shot in-context baselines AND trained specialists,
% the panel's datasets, FLAT, in three blocks by how each method uses the K support masks (forward pass / fitted
% on the support / specialists: fine-tuned on the support, or off the shelf under ASG_SPEC_MODE=ots).
% Generated by scripts/make_semantic_tables.py from the clean
% campaign tree; do not hand-edit values. """ + (f"Our column: {OURS_STEM}_k*. Bold = best within each group of methods, "
        "underline = best in the row; each metric group ends with its mean." if SPEC_MODE == "ftsem"
        else "Bold = best in the row across ALL THREE blocks.") + r"""
\begin{table*}[t]
\centering
\caption{""" + caption + r"""}
\label{tab:fewshot}
{\scriptsize\setlength{\tabcolsep}{""" + ("0.8pt" if SPEC_MODE == "ftsem" else "1.2pt") + r"""}
\begin{tabular}{""" + COLSPEC + r"""}
\toprule
""" + GROUP_ROW + r""" \\
""" + GROUP_RULES + r"""
""" + HEADER + r""" \\
\midrule
""" + body + "\n" + aggregate + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""
    )
    fname = "tab_fewshot.tex" if K == 8 else f"tab_fewshot_k{K}.tex"
    open(os.path.join(OUT, fname), "w").write(tex)
    print(f"wrote {fname} ({len(_SEEN_SEEDS)} cells, all at {EXPECT_SEEDS} seeds; "
          f"specialists lead on {won})")


if __name__ == "__main__":
    emit_merged()
