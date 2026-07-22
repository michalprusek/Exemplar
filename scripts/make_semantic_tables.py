"""Emit the MERGED main semantic-comparison table for the paper, from the clean campaign tree.

One Table 1 places our method against the paradigm-matched few-shot in-context baselines (which see the
same eight support masks) AND the trained specialists (which use no support but were trained on thousands
of objects), on the SAME per-dataset semantic metric (foreground IoU for blob/nucleus/worm/bacteria/decay
fields, centreline Dice for the vessel/membrane/filament fields). Eleven datasets, FLAT (no development/
held-out split). Bold = best in the row across ALL methods (few-shot + specialists together).

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
  tab_fewshot.tex  -- the merged 11-dataset, 8-method main results table (\\label{tab:fewshot}).

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
    """INSID3 at the per-dataset better of its two documented CRF modes (this repo's steelman)."""
    a, b = stat("insid3_guided_k8", ds, metric), stat("insid3_dense_k8", ds, metric)
    cands = [x for x in (a, b) if x]
    return max(cands, key=lambda c: c[0]) if cands else None


def resolve(dirname, ds, metric):
    """Dispatch to the INSID3 better-of-two-modes helper, else a plain directory stat."""
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
    return f"{m:.3f}\\pm{s:.3f}"


def bold(x):
    return f"{{\\boldmath${x}$}}" if "--" not in x else x


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
            cells.append(bold(s) if best is not None and abs(m - best) < 1e-9 else f"${s}$")
    return r"\midrule" + "\n" + r"\textbf{Mean} & " + " & ".join(cells) + r" \\"


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
            cells.append(bold(v) if is_best else (f"${v}$" if v != "--" else "--"))
        out.append(f"{name:12} & " + " & ".join(cells) + r" \\")
    return "\n".join(out)


# The five few-shot in-context columns, then the three trained-specialist columns (deterministic).
COLS = [
    ("Ours", "ours_k8", False), ("SegGPT", "seggpt_k8", False),
    ("UniverSeg", "universeg_k8", False), ("INSID3", "insid3", False),
    ("Tyche", "tyche_k8", False),
    ("Cellpose", "cellpose_sam", True), ("StarDist", "stardist", True),
    ("micro-SAM", "microsam", True),
]
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


def emit_merged():
    body = build_body(COLS)
    check_seeds()
    won = specialist_won(COLS)
    ft = ft_delta()
    # The caption spells small counts as words, matching the manuscript's style, so the generated
    # file is byte-identical to what the paper compiles -- a reader who re-runs this script must get
    # the committed table back, not a near-miss they have to eyeball.
    _WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
             8: "eight", 9: "nine", 10: "ten", 11: "eleven"}
    # The caption no longer NAMES the datasets specialists win (the manuscript dropped them for
    # space), so the derivation would become decorative. Assert it instead: if the set ever changes,
    # this fails rather than printing a sentence the numbers no longer support.
    if sorted(won) != sorted(["DSB2018", "MoNuSeg", "CTC-U373"]):
        raise ValueError(f"caption says specialists lead on the three standard-cell datasets, but "
                         f"the numbers say {won or 'none'}")
    if ft is None:
        raise ValueError("caption cites the fine-tuned-specialist delta but no such record was found")
    won_txt = f"the {_WORD[len(won)]} standard-cell datasets they were built for"
    ft_txt = (f", and fine-tuning a specialist on the eight masks barely changes its cell numbers "
              f"(e.g.\\ Cellpose on MoNuSeg ${ft[0]:.3f}\\rightarrow{ft[1]:.3f}$)")
    caption = (
        r"Per-dataset accuracy on each dataset's semantic metric (\cref{sec:exp}), against paradigm-matched "
        r"few-shot in-context methods that see the same eight support masks and against trained specialists "
        r"that use no support but were trained on thousands of objects. All methods are evaluated over the "
        rf"same {_WORD[EXPECT_SEEDS]} seeds: Ours and the few-shot baselines report mean $\pm$ standard "
        r"deviation, whereas the specialists ignore the support masks and are deterministic (std $=0$ across "
        rf"the {_WORD[EXPECT_SEEDS]} seeds), so they are shown without a deviation. The best score in each "
        rf"row is bold; the trained specialists, which use far more supervision, lead only on {won_txt}. The "
        r"Mean row averages the per-dataset scores across two metrics (foreground IoU and centreline Dice) "
        r"and is therefore a summary, not a single measured quantity. INSID3 is shown at the per-dataset "
        r"better of its two conditional-random-field modes, an oracle choice deliberately conservative "
        r"toward that baseline; specialist foreground is the union of predicted instances"
        rf"{ft_txt}."
    )
    tex = (
        r"""% tab:fewshot — MERGED main results: ours vs few-shot in-context baselines AND trained specialists,
% 11 datasets, FLAT. Generated by scripts/make_semantic_tables.py from the clean campaign tree; do not
% hand-edit values. Bold = best in the row across ALL methods (few-shot + specialists).
\begin{table*}[t]
\centering
\caption{""" + caption + r"""}
\label{tab:fewshot}
{\footnotesize\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{l ccccc ccc}
\toprule
 & \multicolumn{5}{c}{Few-shot in-context} & \multicolumn{3}{c}{Trained specialists} \\
\cmidrule(lr){2-6} \cmidrule(lr){7-9}
Dataset & Ours & SegGPT~\cite{seggpt} & UniverSeg~\cite{universeg} & INSID3~\cite{insid3} & Tyche~\cite{tyche} & Cellpose~\cite{cellpose} & StarDist~\cite{stardist} & micro-SAM~\cite{microsam} \\
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
