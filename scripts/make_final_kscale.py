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
# TrueType, not matplotlib's default Type 3. Type 3 is a bitmap font and IEEE PDF eXpress
# rejects it; the figure carried two DejaVuSans Type 3 faces into the manuscript until 2026-08-25.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
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
    # CORRECTED 2026-07-28. This read `ours_k{k}`, which is `head_fusion_best_cgate_film_nobank`
    # -- the configuration BEFORE the lean regularised head. Table 1 reports
    # `..._flat_es_ep500_wd4_do5_mix` (best_v3), so the figure and the table were plotting two
    # different methods, differing on all eleven datasets by up to 0.082 (drive) and always in
    # the figure's disfavour: the published curve understated our own K-scaling. Verbatim the
    # failure this file's own header warns about ("the headline curve was not one method").
    # K=8 lists two directories because the v3 K=8 arm was written in two passes -- the four
    # centreline datasets landed in `oursv3n_k8` -- and they agree exactly on the seven they
    # share, so this is a COMPLETION of one arm, not the better-of-variants steelman that the
    # INSID3 entry below uses the same mechanism for.
    # RE-POINTED 2026-07-31 at the stripped method, `head_fusion_best_nobank_flat_es_ep500_wd4_do5_mix`,
    # after the competitive gate and FiLM were removed from the method. This is the SAME class of
    # correction the 2026-07-28 note below records, and it has to be made in the same breath as
    # Table 1 and \Cref{tab:ablation}: a curve drawn from `oursv3_k*` beside a table drawn from
    # `stripv3_k*` would again be two methods under one label. The K=8 entry is a single directory
    # because the seven datasets first written to `ablv3_bank_k8` were copied into `stripv3_k8`
    # after verifying every record carries the same method string, support 8 and resolution 672.
    # RE-POINTED 2026-09-01 at `lean_k*`, the arms Table 1 and the ablation now read, after the
    # method dropped self-configuration, the colour rule and the auxiliary outline head. Third time
    # this line has had to move, and for the third time for the same reason: the figure and the
    # table must name the same method or the curve is a different segmenter from the one the paper
    # reports. Until every cell lands this raises rather than drawing a short curve.
    "Exemplar":  {k: [f"{FINAL}/lean_k{k}"] for k in (1, 4, 8, 16)},
    # nnU-Net TRAINS on the support masks rather than conditioning on them, so it is drawn
    # dashed with the fine-tuned specialists. It belongs on an annotation-efficiency plot for
    # exactly the reason they do: the question is what K labels buy, and a network trained from
    # scratch on those same K labels is the most demanding answer. K=1 lives outside the
    # campaign tree because it was run later. K=4 and K=16 are queued and will appear here as
    # they land; until then nnU-Net is a two-point curve and the caption says so.
    # COMPLETE 2026-08-05: the band finished at 210/210 seeds, so nnU-Net is drawn at all four
    # support sizes and its curve no longer has a gap for the generator to break.
    "nnU-Net":   {1: ["nnunet_k1"], 4: [f"{FINAL}/nnunet_k4"], 8: [f"{FINAL}/nnunet_k8"],
                  16: [f"{FINAL}/nnunet_k16"]},
    "SegGPT":    {k: [f"{FINAL}/seggpt_k{k}"] for k in (1, 4, 8, 16)},
    "Tyche":     {k: [f"{FINAL}/tyche_k{k}"] for k in (1, 4, 8, 16)},
    "UniverSeg": {k: [f"{FINAL}/universeg_k{k}"] for k in (1, 4, 8, 16)},
    # The campaign runs BOTH crf modes and writes them to separate directories; there is no plain
    # `insid3_k*`. Naming one would have found nothing and dropped INSID3 from the figure entirely,
    # silently, exactly as the hardcoded method list dropped SegGPT and the fine-tuned specialists.
    # Both are listed so `find` takes the better per dataset, which is this repo's documented
    # steelman and matches what make_table_data.py reports.
    # REPLACED 2026-09-02: was our own correspondence read-out at the better of two CRF modes.
    # It is now the authors' released INSID3 in the configuration their published anchor
    # reproduces from (registry C88). CTC-U373 cannot supply K=16, which is why this figure
    # averages ten datasets anyway.
    "INSID3": {k: [f"{FINAL}/insid3_k{k}"] for k in (1, 4, 8, 16)},
    # The fine-tuned specialists are NOT in the figure, and this is now explicit rather than an
    # accident. They were only ever run at K=8, and only on three datasets each -- the K=1/4/16
    # directories are empty -- so `agg` refused every point and the old code dropped them silently,
    # leaving a figure that named four baselines while claiming to survey the field. They stay out
    # of DIRS until the campaign completes them; the caption already says specialists are discussed
    # separately. Re-add all three the moment the runs finish; the empty-figure guard below will
    # then hold them to full coverage like everything else.
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
# Dashed = trains ON the support masks (as opposed to conditioning on them in a forward pass).
# nnU-Net is from-scratch rather than fine-tuned, so it shares the line style but not the label.
# nnU-Net was drawn DASHED with the fine-tuned specialists, on the argument that it trains on the
# support rather than conditioning on it. That distinction is real and the caption still makes it
# in words, but the line style was the wrong place for it: nnU-Net is measured at all four support
# sizes on the same seeds and the same test images as we are, it BEATS us from four masks on, and
# drawing the one baseline that wins in a demoted style reads as downplaying it. Solid, 2026-08-05.
TRAINS_ON_SUPPORT = FINETUNED
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


STYLE = {"Exemplar": ("#111111", "o", 2.4), "Tyche": ("#1f77b4", "s", 1.4),
         "UniverSeg": ("#2ca02c", "^", 1.4), "INSID3": ("#ff7f0e", "D", 1.4),
         "SegGPT": ("#9467bd", "v", 1.4), "Matcher": ("#d62728", "*", 1.4),
         "PerSAM": ("#8c564b", "P", 1.4), "Cellpose-FT": ("#17becf", "s", 1.4),
         "StarDist-FT": ("#bcbd22", "^", 1.4), "microSAM-FT": ("#e377c2", "D", 1.4),
         "Cellpose": ("#7f7f7f", "", 1.0), "StarDist": ("#aaaaaa", "", 1.0),
         "microSAM": ("#c7c7c7", "", 1.0),
         # nnU-Net TRAINS on the support, so it is dashed with the fine-tuned specialists, but
         # weighted like a headline comparison because it is the method the paper concedes
         # leads at K=8. Its absence here made this whole script unrunnable: MISSING_STYLE
         # raised at import the moment nnU-Net entered DIRS, so the correction that commit
         # carried -- pointing the curve at best_v3 instead of best_v2 -- had never once run.
         "nnU-Net": ("#c7522b", "X", 2.0)}

# The method list is DERIVED from DIRS, never written out again. It used to be a second hardcoded
# literal, and the two drifted: SegGPT, PerSAM and all three fine-tuned specialists had directories
# here and were simply never drawn, so the figure showed our method beating four baselines while the
# campaign had measured nine. A missing style now fails here, loudly, instead of raising KeyError
# halfway through drawing or -- worse -- quietly leaving a measured competitor off the plot.
MISSING_STYLE = sorted(set(DIRS) - set(STYLE))
if MISSING_STYLE:
    raise SystemExit(f"no plot style for {MISSING_STYLE}: every method with a score directory must "
                     f"be drawn, or the figure understates the competition. Add a STYLE entry.")

fig, ax = plt.subplots(figsize=(5.4, 2.5))
print(f"=== final K-scaling (mean foreground over the {len(COMMON)} common datasets; "
      f"band = 95% CI over seeds) ===")
_top = 0.0                                           # highest drawn value, for the axis-clip guard
_drawn = []                                          # methods actually plotted, for the empty-figure guard
_gappy = []   # (method, missing K) -- a partial curve is a wrong figure, not a short one
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
        _gappy.append((method, gaps))
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
        ls = "--" if method in TRAINS_ON_SUPPORT else "-"
        ax.plot(KS, ys, marker=mk, color=c, lw=lw, ls=ls, ms=5,
                label=f"{method} (fine-tuned)" if method in FINETUNED else method, zorder=3)
        ax.fill_between(KS, ys - es, ys + es, color=c, alpha=0.15, lw=0)

ax.set_xscale("log", base=2)
ax.set_xticks(KS); ax.set_xticklabels(KS)
ax.set_xlabel("Support masks $K$")
# "foreground score" is wrong for the four centreline-scored datasets in this mean, and the
# phrase appears nowhere in the paper. Match the caption, which says designated metric.
ax.set_ylabel("Mean designated metric")
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
# Every method with a score directory must contribute at least one point. Guarding only "Ours"
# protected the one absence a human would notice instantly, while a figure showing our curve
# against NOTHING -- every baseline silently dropped for one missing K -- writes cleanly, exits
# 0, and looks more publishable than an empty one.
_expected = {m for m in DIRS if m not in NO_SCALING}
_absent = sorted(_expected - set(_drawn))
if _absent:
    raise SystemExit(f"refusing to write {out}: {_absent} have score directories but drew no "
                     f"points. An annotation-efficiency plot missing its competitors is worse "
                     f"than no figure. Fix the tree, or remove them from DIRS deliberately.")
# A method that drew SOME of its scheduled points is more dangerous than one that drew none:
# matplotlib joins what it has, so a curve missing K=4 and K=16 renders as a clean straight line
# between the two points that survived and nothing on the page says so. Caught 2026-09-01, when
# repointing Exemplar at the lean arms mid-campaign produced exactly that and the file was written
# anyway because every baseline was complete.
if _gappy:
    raise SystemExit("refusing to write {}: {} — a partial curve is drawn as a straight line "
                     "between the points that survived, which is a claim the data does not make. "
                     "Wait for the cells, or drop the K from DIRS.".format(
                         out, "; ".join(f"{m} missing K={g}" for m, g in _gappy)))
if "Exemplar" not in _drawn:
    raise SystemExit(f"refusing to overwrite {out}: drew {_drawn or 'nothing'} — 'Exemplar' produced no "
                     f"points under ASG_RESULTS_ROOT={ROOT!r}")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print("wrote", out)
