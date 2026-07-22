"""Build the final annotation-efficiency figure: mean SEMANTIC score over the ten datasets with full
K coverage (all eleven but ctc_u373, whose pool of fifteen cannot supply K=16), for every method at
K=1,4,8,16, with +/-1 std error bands across seeds. Blob/nucleus/worm/bacteria fields are scored on
foreground IoU (matching Table 1), vessels/membranes/filaments on centreline Dice. Per (method,K,dataset)
we search a prioritised dir list and accept a file ONLY if its metric matches the required semantic
metric, so an AP-scored file can never leak in."""
import glob
import json
import os

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.environ.get("ASG_REPO_ROOT", "/disk1/prusek/active-segmenter")
ROOT = os.environ.get("ASG_RESULTS_ROOT", f"{_REPO}/results")
# Ten of the eleven datasets. ONLY ctc_u373 is excluded: its support pool holds only fifteen images,
# so it cannot supply K=16, and averaging over a set that shrinks at K=16 would put a fake jump in the
# curve. Every method has full K=1,4,8,16 coverage on these ten (Decay/rozpad included).
FG_METRIC = {"spheroidj": "fg_iou", "dsb2018": "fg_iou", "monuseg": "fg_iou",
             "bbbc010": "fg_iou", "bacteria": "fg_iou", "rozpad": "fg_iou",
             "drive": "cldice", "hrf": "cldice",
             "isbi2012em": "cldice", "fisbe": "cldice"}
COMMON = list(FG_METRIC)

# ONE dir per (method, K), all from the single clean tree the campaign writes.
#
# This was a prioritised FALLBACK list, and that silently mixed harnesses: `Ours` at K=8 resolved to
# `scores_fact/cgate_film` (written by `head_fusion_best_cgate_film` — bank-unfreeze ON, the lever
# CLAUDE.md C13 dropped) while K=1/4/16 came from `_v2`, so the headline curve was not one method; and
# after the harness fixes those old dirs still won over the recomputed ones, making the recomputation
# invisible. A missing dir must now be a missing POINT, not a silent substitution.
FINAL = "final10"
DIRS = {
    "Ours":      {k: [f"{FINAL}/ours_k{k}"] for k in (1, 4, 8, 16)},
    "SegGPT":    {k: [f"{FINAL}/seggpt_k{k}"] for k in (1, 4, 8, 16)},
    "Tyche":     {k: [f"{FINAL}/tyche_k{k}"] for k in (1, 4, 8, 16)},
    "UniverSeg": {k: [f"{FINAL}/universeg_k{k}"] for k in (1, 4, 8, 16)},
    # The campaign runs BOTH crf modes and writes them to separate directories; there is no plain
    # `insid3_k*`. Naming one would have found nothing and dropped INSID3 from the figure entirely,
    # silently, exactly as the hardcoded method list dropped SegGPT and the fine-tuned specialists.
    # Both are listed so `find` takes the better per dataset, which is this repo's documented
    # steelman and matches what make_table_data.py reports.
    "INSID3":    {k: [f"{FINAL}/insid3_guided_k{k}", f"{FINAL}/insid3_dense_k{k}"]
                  for k in (1, 4, 8, 16)},
    "Cellpose-FT":  {k: [f"{FINAL}/cellpose_ft_k{k}"] for k in (1, 4, 8, 16)},
    "StarDist-FT":  {k: [f"{FINAL}/stardist_ft_k{k}"] for k in (1, 4, 8, 16)},
    "microSAM-FT":  {k: [f"{FINAL}/microsam_ft_k{k}"] for k in (1, 4, 8, 16)},
    # One-shot BY CONSTRUCTION (PerSAM = "Personalize SAM with One Shot"); the multi-shot prototype is
    # our adaptation, so both points are produced and the figure marks them as one-shot operating points.
    # Matcher K=4 (our multi-shot adaptation) was dropped from the campaign: its dense
    # correspondence cost 5+ h per dense/high-res cell and the paper needs only the K=1
    # one-shot operating point. Matcher stays a single one-shot marker.
    "Matcher":   {1: [f"{FINAL}/matcher_k1"]},
    "PerSAM":    {1: [f"{FINAL}/persam_k1"], 8: [f"{FINAL}/persam_k8"]},
    # Off-the-shelf specialists: ONE directory, no _k suffix, because they ignore the support masks.
    # The same directory at every K is what makes them a flat reference line, which is the honest
    # rendering: they do not scale with K because they never see K. They were absent from this figure
    # entirely while `make_table_data.py` reported them, so the figure and the table disagreed about
    # which baselines had even been measured — and the common set includes three datasets where
    # CLAUDE.md records that the specialists win.
    "Cellpose":  {k: [f"{FINAL}/cellpose_sam"] for k in (1, 4, 8, 16)},
    "StarDist":  {k: [f"{FINAL}/stardist"] for k in (1, 4, 8, 16)},
    "microSAM":  {k: [f"{FINAL}/microsam"] for k in (1, 4, 8, 16)},
}
# Drawn as flat reference lines, and labelled as such, so nobody reads them as a K-scaling trend.
SUPPORT_BLIND = {"Cellpose", "StarDist", "microSAM"}
KS = [1, 4, 8, 16]

# Methods that train ON the support masks rather than conditioning on them. They belong on this
# figure -- the question it answers is "what does K labels buy you", and a specialist fine-tuned on
# those same K labels is the most demanding answer to that question -- but they are not in-context
# methods, so they are drawn dashed and the caption must say what the distinction is.
FINETUNED = {"Cellpose-FT", "StarDist-FT", "microSAM-FT"}
# Methods whose published contribution is a SINGLE support example. Drawing a line through their
# points would imply a K-scaling claim their authors never made, so they get markers only.
ONE_SHOT = {"Matcher", "PerSAM"}

# A K-scaling curve is only meaningful for methods that consume the K support masks and can therefore
# improve with more of them. The off-the-shelf specialists never see the support (flat by
# construction) and the one-shot methods take a single mask by design, so neither belongs on an
# annotation-efficiency plot -- they are reported in the tables instead. Only the in-context few-shot
# methods (ours, SegGPT, UniverSeg, Tyche, INSID3) are drawn here.
NO_SCALING = SUPPORT_BLIND | ONE_SHOT


def per_seed(path):
    d = json.load(open(path))
    pi = np.asarray(d["per_image"], float)
    ns, t = len(d["seeds"]), d["test_per_seed"]
    if ns * t != len(pi):                            # FAIL-LOUD: writer guarantees this shape; a mismatch
        raise ValueError(f"{path}: per_image len {len(pi)} != len(seeds)*test_per_seed "
                         f"({ns}*{t}); malformed/partial score file, refusing to guess a mean")
    # The seed LIST is returned, not just the count: `agg` pairs datasets by seed index, so two
    # datasets re-run over different seed VALUES of equal length would be silently mis-paired and the
    # figure's CI would be computed over mismatched replications.
    return d["metric"], pi.reshape(ns, t).mean(1), list(d["seeds"])


def find(dirs, ds):
    """Best metric-matching file for dataset ds across the listed dirs, or None.

    When a method lists more than one directory they are its documented VARIANTS (INSID3's two crf
    modes), and the reported number is the better of them per dataset. This used to return the FIRST
    directory that matched, which for a two-variant method silently reported whichever name happened
    to be written first rather than the steelman the table reports, so the figure and the table could
    disagree about the same baseline. Errors if a single dir holds more than one metric-matching
    candidate, since a dir is assumed method-pure.
    """
    best = None
    for d in dirs:
        matches = []
        for f in sorted(glob.glob(os.path.join(ROOT, d, f"*__{ds}.json"))):  # sorted = deterministic
            m, arr, sds = per_seed(f)
            if m == FG_METRIC[ds]:
                matches.append((f, arr, sds))
        # A dir can hold two variant records (PerSAM writes persam__ AND persam_f__). Keep the base
        # variant, whose filename starts with the dir's method stem. A dir left genuinely ambiguous
        # (two base-looking files) is still an error rather than a silent pick.
        if len(matches) > 1:
            stem = os.path.basename(d).rsplit("_k", 1)[0]
            based = [t for t in matches if os.path.basename(t[0]).startswith(f"{stem}__")]
            if len(based) == 1:
                matches = based
            else:
                raise ValueError(f"ambiguous {ds}/{FG_METRIC[ds]} in {d}: {[f for f, *_ in matches]}")
        if matches and (best is None or matches[0][1].mean() > best[0].mean()):
            best = (matches[0][1], matches[0][2])            # (seed-means, seed values)
    return best


def agg(method, k):
    """Aggregate = per seed, average the COMMON datasets' seed-means (seed-paired: the same seed VALUE
    across datasets, which is asserted below -- equal seed COUNTS alone would silently mis-pair a
    dataset re-run over a different seed set). This gives one aggregate value per seed; those n seeds
    are i.i.d. replications of the whole experiment, so mean +/- t*SD/sqrt(n) is a proper 95% CI."""
    dirs = DIRS[method].get(k)
    if not dirs:
        return None
    arrs = [find(dirs, ds) for ds in COMMON]
    have = [a for a in arrs if a is not None]
    if len(have) < len(COMMON):                     # RIGOR: only plot a point over the FULL common set
        miss = [ds for ds, a in zip(COMMON, arrs) if a is None]
        print(f"  ! {method} K={k}: skipped (incomplete, missing {miss})")
        return None
    lens = {len(a[0]) for a in have}
    if len(lens) != 1:                               # FAIL-LOUD: no silent min()-truncation of seeds
        raise ValueError(f"{method} K={k}: unequal seed counts across datasets "
                         f"{list(zip(COMMON, [len(a[0]) for a in have]))}; refusing to truncate")
    seedsets = {tuple(a[1]) for a in have}
    if len(seedsets) != 1:                           # FAIL-LOUD: equal COUNTS but different seed VALUES
        raise ValueError(f"{method} K={k}: datasets do not share one seed set "
                         f"{ {ds: a[1] for ds, a in zip(COMMON, have)} }; index-pairing them would "
                         f"average mismatched replications")
    M = np.stack([a[0] for a in have])               # (datasets, seeds) — same seed values, index-paired
    ps = M.mean(0)                                   # (seeds,) seed-paired aggregate
    ns = len(ps)
    mean = float(ps.mean())
    if ns >= 2:
        sem = float(ps.std(ddof=1)) / np.sqrt(ns)          # sample SD -> SEM
        ci = float(stats.t.ppf(0.975, ns - 1)) * sem       # 95% CI half-width (t, df=n-1)
    else:
        ci = 0.0
    return mean, ci, ns


STYLE = {"Ours": ("#111111", "o", 2.4), "Tyche": ("#1f77b4", "s", 1.4),
         "UniverSeg": ("#2ca02c", "^", 1.4), "INSID3": ("#ff7f0e", "D", 1.4),
         "SegGPT": ("#9467bd", "v", 1.4), "Matcher": ("#d62728", "*", 1.4),
         "PerSAM": ("#8c564b", "P", 1.4), "Cellpose-FT": ("#17becf", "s", 1.4),
         "StarDist-FT": ("#bcbd22", "^", 1.4), "microSAM-FT": ("#e377c2", "D", 1.4),
         "Cellpose": ("#7f7f7f", "", 1.0), "StarDist": ("#aaaaaa", "", 1.0),
         "microSAM": ("#c7c7c7", "", 1.0)}

# The method list is DERIVED from DIRS, never written out again. It used to be a second hardcoded
# literal, and the two drifted: SegGPT, PerSAM and all three fine-tuned specialists had directories
# here and were simply never drawn, so the figure showed our method beating four baselines while the
# campaign had measured nine. A missing style now fails here, loudly, instead of raising KeyError
# halfway through drawing or -- worse -- quietly leaving a measured competitor off the plot.
MISSING_STYLE = sorted(set(DIRS) - set(STYLE))
if MISSING_STYLE:
    raise SystemExit(f"no plot style for {MISSING_STYLE}: every method with a score directory must "
                     f"be drawn, or the figure understates the competition. Add a STYLE entry.")

fig, ax = plt.subplots(figsize=(5.4, 3.0))
print(f"=== final K-scaling (mean foreground over the {len(COMMON)} common datasets; "
      f"band = 95% CI over seeds) ===")
_top = 0.0                                           # highest drawn value, for the axis-clip guard
_drawn = []                                          # methods actually plotted, for the empty-figure guard
for method in DIRS:
    if method in NO_SCALING:                         # not a scaling method -> reported in the tables
        continue
    # Collect over ALL of KS, keeping a hole as NaN rather than closing the gap. `agg` refuses to
    # average an incomplete common set and prints a warning, but dropping that K from `xs` made
    # `ax.plot` draw a straight segment across the hole, passing through a K position that was never
    # measured and looking exactly like a measured trend. NaN breaks the line instead.
    ys_all, es_all, have = [], [], []
    for k in KS:
        r = agg(method, k) if k in DIRS[method] else None
        if r is None:
            ys_all.append(np.nan); es_all.append(np.nan)
        else:
            ys_all.append(r[0]); es_all.append(r[1]); have.append((k, r[0], r[1], r[2]))
    if not have:
        print(f"  ! {method}: NO points -- absent from the figure")
        continue
    print(f"{method:12}", " ".join(f"K{k}={y:.3f}±{e:.3f}(n={n})" for k, y, e, n in have))
    _drawn.append(method)
    scheduled = [k for k in KS if k in DIRS[method]]
    gaps = [k for k in scheduled if np.isnan(ys_all[KS.index(k)])]
    if gaps:
        print(f"  ! {method}: gap at K={gaps} -- the curve is BROKEN there, not interpolated")
    c, mk, lw = STYLE[method]
    ys, es = np.array(ys_all, float), np.array(es_all, float)
    _top = max(_top, float(np.nanmax(ys + es)))
    if method in ONE_SHOT:                           # markers only: no K-scaling claim is implied
        xs = [k for k, _, _, _ in have]
        ax.errorbar(xs, [y for _, y, _, _ in have], yerr=[e for _, _, e, _ in have],
                    fmt=mk, color=c, ms=11, capsize=2, label=f"{method} (one-shot)", zorder=4)
    elif method in SUPPORT_BLIND:
        # Flat by construction: these never see the support. Drawn thin and dotted so they read as a
        # reference level rather than as a competitor whose accuracy happens not to vary with K.
        ax.plot(KS, ys, color=c, lw=lw, ls=":", label=f"{method} (no support)", zorder=2)
    else:
        ls = "--" if method in FINETUNED else "-"
        ax.plot(KS, ys, marker=mk, color=c, lw=lw, ls=ls, ms=5,
                label=f"{method} (fine-tuned)" if method in FINETUNED else method, zorder=3)
        ax.fill_between(KS, ys - es, ys + es, color=c, alpha=0.15, lw=0)

ax.set_xscale("log", base=2)
ax.set_xticks(KS); ax.set_xticklabels(KS)
ax.set_xlabel("Support masks $K$")
ax.set_ylabel("Mean foreground score")
# The upper limit was FIXED at 0.85. A baseline scoring above that would have been drawn outside the
# axes and simply not appeared, which on this figure means a competitor that beats us goes missing --
# and smoke tests already put a fine-tuned micro-SAM above 0.85 on one dataset. The limit now follows
# the data and says so when it had to grow.
if _top > 0.85:
    print(f"  ! y-limit raised to fit the data (max drawn {_top:.3f} > the default 0.85)")
ax.set_ylim(0, max(0.85, _top * 1.04))
ax.grid(True, which="major", ls=":", alpha=0.4)
# legend OUTSIDE the axes (right) so it never overlaps the curves
ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.tight_layout()
out = os.environ.get("ASG_KSCALE_OUT", f"{_REPO}/paper/isbi2027/figures/kscale.pdf")
# EMPTY-FIGURE GUARD. Every guard above fails loud on a bad RECORD, but a wrong ROOT (running on a
# machine without the tree, or after a directory rename) makes every method resolve to "NO points" and
# execution would fall straight through to savefig -- overwriting the real figure with a perfectly
# well-formed EMPTY one, exit 0. A figure with no curves, or without ours, is never worth writing.
if "Ours" not in _drawn:
    raise SystemExit(f"refusing to overwrite {out}: drew {_drawn or 'nothing'} — 'Ours' produced no "
                     f"points under ASG_RESULTS_ROOT={ROOT!r}")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print("wrote", out)
