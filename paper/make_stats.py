# Copied from AutoSeg scripts/make_stats.py (cb71f85) by tools/sync_paper.py; default paths adapted, internal notes rewritten in comments and docstrings.
"""Paired per-image Wilcoxon with Holm correction, emitted as LaTeX macros for the manuscript.

WHY THIS EXISTS. The paper describes its statistics -- paired Wilcoxon signed-rank on per-image
scores, Holm-corrected across the family, effect sizes and bootstrap intervals -- and then shows a
table of means and standard deviations. A reviewer is entitled to ask where the tests are. This
computes them and writes macros, so what the paper claims about significance is generated from the
same score tree as everything else rather than asserted.

THE UNIT OF ANALYSIS IS THE IMAGE, NOT THE SEED-IMAGE PAIR. Each record holds `per_image` of length
`len(seeds) * test_per_seed`: the same test images scored under every seed. Testing on that vector
directly treats ten scores of one image as ten independent observations, which inflates n by 10x and
every p-value with it -- the pseudoreplication this project's own fairness audit flagged. Seeds are
therefore averaged per image first, giving one paired sample per test image, which is what the
manuscript says it does.

HOLM, NOT BONFERRONI, and the family is stated: for each claim, the comparisons that claim rests on.
Holm is uniformly more powerful than Bonferroni at the same familywise error rate, so reporting the
weaker correction would understate our own result while sounding more careful.

  ASG_SEM_TREE=results/final10 ASG_SEM_OUT=paper/isbi2027 python scripts/make_stats.py
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy import stats

ROOT = os.environ["ASG_SEM_TREE"]
OUT = os.environ.get("ASG_SEM_OUT", ".")

DATASETS = [("spheroidj", "fg_iou"), ("rozpad", "fg_iou"), ("dsb2018", "fg_iou"),
            ("monuseg", "fg_iou"), ("ctc_u373", "fg_iou"), ("bbbc010", "fg_iou"),
            ("bacteria", "fg_iou"), ("drive", "cldice"), ("hrf", "cldice"),
            ("isbi2012em", "cldice"), ("fisbe", "cldice")]
# ASG_DROP_DS=ctc_u373 (the paper's setting: CTC-U373 is not reported, because its conditions need the CTC
# organisers' permission for public use outside the challenge): the datasets named here are left out of every family.
_DROP = {x for x in os.environ.get("ASG_DROP_DS", "").split(",") if x}
DATASETS = [d for d in DATASETS if d[0] not in _DROP]
# The stripped method (gate and FiLM removed, 2026-07-31). Must match Table 1 and the
# K-scaling figure: a stats table computed against a different arm from the one the table
# reports would compare two methods under one name.
# The Results prose names the dataset Exemplar wins on. That name is an OUTCOME of the test, not
# a fixed label -- if the win moved, the sentence would have to move with it. Emitted as a macro
# for the same reason every number here is.
_DISPLAY = {"spheroidj": "SpheroidJ", "rozpad": "Decay", "dsb2018": "DSB2018",
            "monuseg": "MoNuSeg", "ctc_u373": "CTC-U373", "bbbc010": "BBBC010",
            "bacteria": "Bacteria", "drive": "DRIVE", "hrf": "HRF",
            "isbi2012em": "ISBI2012-EM", "fisbe": "FISBe"}
_NUMWORD = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
            6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven"}
# ASG_OURS_STEM names the arm family (`lean`, the method reported until 2026-10-14, or `exemplar`, the method of
# 14 October), as in Table 1's generator, so the tests and the table cannot describe two methods.
OURS_STEM = os.environ.get("ASG_OURS_STEM", "lean")
OURS = f"{OURS_STEM}_k8"
# ASG_TYCHE_STEM=tyche_bestof: Tyche scored as in its paper, the one of its 16 candidates that best matches the test
# mask by the dataset's metric (scripts/tyche_bench.py). Default `tyche` = the mean of the candidates, the earlier
# column.
TYCHE_STEM = os.environ.get("ASG_TYCHE_STEM", "tyche")
FORWARD_PASS = {"SegGPT": "seggpt_k8", "UniverSeg": "universeg_k8",
                # Tyche is scored here by the MEAN of its candidate maps, which uses no ground
                # truth, because every other column of this table is scored without one and an
                # oracle number sitting among them misleads a reader scanning columns. Its own
                # paper reports best-of-K (min_k L_Dice(y_k, y)); that arm exists as
                # `tyche_bestof_k8` and the Baselines paragraph states its result, which does not
                # change the conclusion -- Exemplar leads on all eleven either way.
                "Tyche": f"{TYCHE_STEM}_k8",
                # A TUPLE would mean a baseline with several documented modes, tested at the
                # per-dataset better one so the headline win count is not earned against a weaker
                # configuration than the table prints. INSID3 no longer needs it: the column is the
                # authors' released implementation in the single configuration their published
                # anchor reproduces from, not our own two-mode read-out.
                "INSID3": "insid3_k8",
                "Matcher": "matcher_k1"}
# ASG_SEM_NO_MATCHER=1: Matcher left Table 1, so it leaves this family too.
if os.environ.get("ASG_SEM_NO_MATCHER", "") == "1":
    del FORWARD_PASS["Matcher"]
NNUNET = "nnunet_k8"
# The same comparison at ONE support mask (family 4). The abstract and the Discussion make a claim
# about K=1, and until 2026-10-09 no test stood behind it. `nnunet_k1` sits beside the tree, not in
# it, which is where the audit script and the K-scaling figure already read it from.
OURS_K1, NNUNET_K1 = f"{OURS_STEM}_k1", "../nnunet_k1"
# The 2x2 inputs ablation: same head, same support-derived rules, differing only in
# which input is zeroed. Both arms are rules-on, so the pair isolates the inputs.
BANK_ONLY = f"{OURS_STEM}_bankonly_k8"
FEATURES_ONLY = f"{OURS_STEM}_nocls_k8"
# The strongest network trained on the support (Table 1's best mean since 2026-10-10): fine-tuned Cellpose-SAM read on
# its foreground map. Family 5, emitted only for the exemplar stem, whose Results compare with it.
CELLPOSE_FT = "cellpose_ft11sem_k8"


def per_image(dirname, ds, metric):
    """Seed-averaged score per test image, or None. One value per image, never per seed-image pair."""
    for f in sorted(glob.glob(os.path.join(ROOT, dirname, f"*__{ds}.json"))):
        j = json.load(open(f))
        if j.get("metric") != metric:
            continue
        pi = np.asarray(j["per_image"], float)
        ns, t = len(j["seeds"]), j["test_per_seed"]
        if ns * t != pi.size:
            raise ValueError(f"{f}: per_image {pi.size} != {ns}*{t}")
        return pi.reshape(ns, t).mean(0)          # <- collapse seeds; unit of analysis is the image
    return None


def record(dirname, ds, metric):
    """The one record of `dirname` for this dataset at this metric, or None."""
    for f in sorted(glob.glob(os.path.join(ROOT, dirname, f"*__{ds}.json"))):
        j = json.load(open(f))
        if j.get("metric") == metric:
            return j
    return None


def require_paired(a_dir, b_dir):
    """Fail unless the two arms hold every dataset on one split, one seed list and one test size.

    `compare` skips a dataset it cannot pair, which is right for a baseline that was never run on
    one. For a family whose counts go into a sentence ("leads on five, trails on six") a skipped
    dataset silently changes both numbers, so that family checks first.
    """
    for ds, metric in DATASETS:
        a, b = record(a_dir, ds, metric), record(b_dir, ds, metric)
        if a is None or b is None:
            raise ValueError(f"{ds}: no {metric} record in {a_dir if a is None else b_dir}")
        for key in ("split_fp", "seeds", "test_per_seed"):
            if a.get(key) != b.get(key):
                raise ValueError(f"{ds}: {a_dir} and {b_dir} differ in {key}; not a paired comparison")


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def arm_scores(arm, ds, metric):
    """Per-image scores for a baseline on one dataset, at its BEST mode when it documents several.

    Selection is on the same quantity the table bolds, so the test and the printed number can never
    describe different configurations of the same baseline.
    """
    if isinstance(arm, str):
        return per_image(arm, ds, metric)
    cands = [v for v in (per_image(d, ds, metric) for d in arm) if v is not None]
    return max(cands, key=lambda v: v.mean()) if cands else None


def compare(arm, ref=None):
    """-> [(dataset, delta, raw p)] for `ref` (default ours) vs `arm`, paired on images.

    `ref` is exposed so an ablation family can pair two ablated arms against each other rather
    than both against the full method: the bank-versus-features claim is about those two inputs,
    not about either one's distance from Exemplar.
    """
    out = []
    for ds, metric in DATASETS:
        a, b = arm_scores(ref or OURS, ds, metric), arm_scores(arm, ds, metric)
        if a is None or b is None or a.shape != b.shape:
            continue
        d = a - b
        p = 1.0 if np.allclose(d, 0) else stats.wilcoxon(a, b).pvalue
        out.append((ds, float(d.mean()), float(p)))
    return out



def table1_significance():
    """{Table 1 column: set of datasets} where that column differs from Exemplar at a Holm-adjusted p < 0.05, in either
    direction, within the families the Results report: the forward-pass methods as ONE family (as \\fpSig counts it),
    nnU-Net, fine-tuned Cellpose-SAM, and Convpaint under ASG_SEM_CONVPAINT=1 (eleven comparisons each). Table 1's
    generator marks these cells under ASG_SEM_SIGMARKS=1. Columns outside these families
    (Prior-bank RF, StarDist, micro-SAM) are not tested and get no mark."""
    sig = {}
    fam, labels = [], []
    for name, arm in FORWARD_PASS.items():
        for ds, _, p in compare(arm):
            fam.append(p); labels.append((name, ds))
    for (name, ds), a in zip(labels, holm(np.asarray(fam))):
        if a < 0.05:
            sig.setdefault(name, set()).add(ds)
    single = [("nnU-Net", NNUNET), ("Cellpose-SAM", CELLPOSE_FT)]
    if os.environ.get("ASG_SEM_CONVPAINT", "") == "1":
        single.append(("Convpaint", "convpaint_k8"))
    for name, arm in single:
        require_paired(OURS, arm)
        rows = compare(arm)
        if len(rows) != len(DATASETS):
            raise ValueError(f"{name} family paired {len(rows)} of {len(DATASETS)} datasets")
        for (ds, _, _), a in zip(rows, holm(np.asarray([p for _, _, p in rows]))):
            if a < 0.05:
                sig.setdefault(name, set()).add(ds)
    return sig


def _list(names):
    """'A', 'A and B', 'A, B and C' -- a dataset list as a sentence writes it."""
    names = list(names)
    return "none" if not names else names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]

def main():
    lines = ["% Generated by scripts/make_stats.py. Paired per-image Wilcoxon, seeds collapsed to one",
             "% score per image before testing, Holm-corrected within each stated family."]

    # FAMILY 1: the headline claim -- ours beats every forward-pass method on every dataset.
    fam, labels = [], []
    for name, arm in FORWARD_PASS.items():
        for ds, d, p in compare(arm):
            fam.append(p); labels.append((name, ds, d))
    adj = holm(np.asarray(fam))
    wins = sum(1 for (_, _, d) in labels if d > 0)
    sig = sum(1 for (_, _, d), a in zip(labels, adj) if d > 0 and a < 0.05)
    print(f"forward-pass family: {len(fam)} comparisons, ours ahead on {wins}, "
          f"{sig} of those Holm-significant at 0.05")
    worst = max(a for (_, _, d), a in zip(labels, adj) if d > 0)
    lines += [f"\\newcommand{{\\fpComparisons}}{{{len(fam)}}}",
              f"\\newcommand{{\\fpWins}}{{{wins}}}",
              f"\\newcommand{{\\fpSig}}{{{sig}}}",
              f"\\newcommand{{\\fpWorstAdjP}}{{{worst:.3g}}}"]

    # FAMILY 2: ours vs nnU-Net, eleven comparisons.
    nn = compare(NNUNET)
    adjn = holm(np.asarray([p for _, _, p in nn]))
    # A dataset whose two means print the same in Table 1 (three decimals) is level: the reader counts the printed
    # cells, and "ahead on eight" beside MoNuSeg 0.671 against 0.671 reads as a miscount (an external review noted it).
    # The tests and their significance lists are unchanged.
    dm = dict(DATASETS)
    nn_tied = [ds for ds, _, _ in nn if round(float(arm_scores(OURS, ds, dm[ds]).mean()), 3)
               == round(float(arm_scores(NNUNET, ds, dm[ds]).mean()), 3)]
    nn_ahead = sum(1 for ds, d, _ in nn if d < 0 and ds not in nn_tied)
    nn_sig = sum(1 for (_, d, _), a in zip(nn, adjn) if d < 0 and a < 0.05)
    us_sig = sum(1 for (_, d, _), a in zip(nn, adjn) if d > 0 and a < 0.05)
    print(f"nnU-Net family: {len(nn)} comparisons, nnU-Net ahead on {nn_ahead} "
          f"({nn_sig} Holm-significant), ours ahead significantly on {us_sig}")
    for ds, d, p in nn:
        a = adjn[[x[0] for x in nn].index(ds)]
        print(f"   {ds:<12} delta {d:+.4f}  raw p {p:.2g}  Holm {a:.2g}")
    # Word form beside \nnSigWord: the sentence spells its other counts, so a bare "9" there
    # reads as a typesetting slip. The numeral form stays available for any table that wants it.
    us_names = [_DISPLAY.get(ds, ds) for (ds, d, _), a in zip(nn, adjn) if d > 0 and a < 0.05]
    nn_names = [_DISPLAY.get(ds, ds) for (ds, d, _), a in zip(nn, adjn) if d < 0 and a < 0.05]
    lines += [f"\\newcommand{{\\nnAheadWord}}{{{_NUMWORD.get(nn_ahead, nn_ahead)}}}",
              f"\\newcommand{{\\oursSigDs}}{{{_list(us_names)}}}",
              f"\\newcommand{{\\nnAhead}}{{{nn_ahead}}}",
              f"\\newcommand{{\\nnSig}}{{{nn_sig}}}",
              # word form: the sentence around it spells its other counts, and "only 6" beside
              # "nine datasets" reads as a typesetting slip rather than a number.
              f"\\newcommand{{\\nnSigWord}}{{{_NUMWORD.get(nn_sig, nn_sig)}}}",
              f"\\newcommand{{\\oursSigVsNN}}{{{us_sig}}}"]
    if OURS_STEM in ("exemplar", "ridge", "ridge2", "ridge3", "ridge4"):                      # the Results name it; an outcome, so a macro
        us_ahead_nn = len(nn) - nn_ahead - len(nn_tied)
        lines += [f"\\newcommand{{\\nnSigDs}}{{{_list(nn_names)}}}",
                  f"\\newcommand{{\\nnOursAheadWord}}{{{_NUMWORD.get(us_ahead_nn, us_ahead_nn)}}}",
                  f"\\newcommand{{\\nnTiedDs}}{{{_list(_DISPLAY.get(ds, ds) for ds in nn_tied)}}}",
                  f"% level as printed (three decimals): {nn_tied}"]

    # FAMILY 3: the two inputs against each other, eleven comparisons. This is the paper's
    # headline measurement, so it gets a declared family of its own rather than riding on
    # FAMILY 1's correction, whose comparisons test a different claim.
    # complete = a record for every dataset of the panel (a count of files would call an arm with a dropped dataset's
    # record incomplete, ASG_DROP_DS)
    if not all(os.path.isdir(os.path.join(ROOT, d))
               and all(glob.glob(os.path.join(ROOT, d, f"*__{ds}.json")) for ds, _ in DATASETS)
               for d in (BANK_ONLY, FEATURES_ONLY)):
        # The two input-ablation arms are still running (C112b for the exemplar stem). The macros print a visible
        # placeholder rather than a number from another method, and audit_paper_numbers refuses it.
        print(f"bank-vs-features family: PENDING ({BANK_ONLY}, {FEATURES_ONLY} incomplete)")
        lines += ["% PENDING: the input ablation arms are incomplete",
                  "\\newcommand{\\bankComparisons}{\\textbf{??}}", "\\newcommand{\\bankAhead}{\\textbf{??}}",
                  "\\newcommand{\\bankSig}{\\textbf{??}}", "\\newcommand{\\featSigVsBank}{\\textbf{??}}"]
        bf = None
    else:
        bf = compare(FEATURES_ONLY, ref=BANK_ONLY)
    if bf is not None:
        adjb = holm(np.asarray([p for _, _, p in bf]))
        bank_ahead = sum(1 for _, d, _ in bf if d > 0)
        bank_sig = sum(1 for (_, d, _), a in zip(bf, adjb) if d > 0 and a < 0.05)
        feat_sig = sum(1 for (_, d, _), a in zip(bf, adjb) if d < 0 and a < 0.05)
        print(f"bank-vs-features family: {len(bf)} comparisons, bank ahead on {bank_ahead} "
              f"({bank_sig} Holm-significant), features ahead significantly on {feat_sig}")
        for ds, d, p in bf:
            a = adjb[[x[0] for x in bf].index(ds)]
            print(f"   {ds:<12} delta {d:+.4f}  raw p {p:.2g}  Holm {a:.2g}")
        lines += [f"\\newcommand{{\\bankComparisons}}{{{len(bf)}}}",
                  f"\\newcommand{{\\bankAhead}}{{{_NUMWORD.get(bank_ahead, bank_ahead)}}}",
                  f"\\newcommand{{\\bankSig}}{{{_NUMWORD.get(bank_sig, bank_sig)}}}",
                  f"\\newcommand{{\\featSigVsBank}}{{{_NUMWORD.get(feat_sig, feat_sig)}}}"]

    # FAMILY 4: ours vs nnU-Net at ONE support mask, eleven comparisons. Its own family because it
    # is its own claim (who leads from a single annotated image), stated as such in the paper.
    require_paired(OURS_K1, NNUNET_K1)
    k1 = compare(NNUNET_K1, ref=OURS_K1)
    if len(k1) != len(DATASETS):
        raise ValueError(f"K=1 family paired {len(k1)} of {len(DATASETS)} datasets")
    adj1 = holm(np.asarray([p for _, _, p in k1]))
    us_ahead = sum(1 for _, d, _ in k1 if d > 0)
    us_sig1 = sum(1 for (_, d, _), a in zip(k1, adj1) if d > 0 and a < 0.05)
    nn_sig1 = sum(1 for (_, d, _), a in zip(k1, adj1) if d < 0 and a < 0.05)
    print(f"nnU-Net family at K=1: {len(k1)} comparisons, ours ahead on {us_ahead} "
          f"({us_sig1} Holm-significant), nnU-Net ahead on {len(k1) - us_ahead} ({nn_sig1})")
    for (ds, d, p), a in zip(k1, adj1):
        print(f"   {ds:<12} delta {d:+.4f}  raw p {p:.2g}  Holm {a:.2g}")
    # The largest single difference, named: the panel mean at K=1 is higher for Exemplar, and a
    # reader should be able to see how much of that one dataset carries.
    top_ds, top_d, _ = max(k1, key=lambda r: abs(r[1]))
    without = np.mean([d for ds, d, _ in k1 if ds != top_ds])
    print(f"   largest: {top_ds} {top_d:+.4f}; panel delta {np.mean([d for _, d, _ in k1]):+.4f}, "
          f"without {top_ds} {without:+.4f}")
    # The same subtraction on the metric group that dataset belongs to. The sentence before this
    # one gives Exemplar's overlap lead over all seven overlap datasets, and a reader holding the
    # largest single difference can work out what is left without it, so the paper states it.
    metric_of = dict(DATASETS)
    group = [d for ds, d, _ in k1 if metric_of[ds] == metric_of[top_ds]]
    rest = [d for ds, d, _ in k1 if metric_of[ds] == metric_of[top_ds] and ds != top_ds]
    rest_mean = float(np.mean(rest))
    print(f"   {metric_of[top_ds]} group: {np.mean(group):+.4f} over {len(group)}, "
          f"{rest_mean:+.4f} over the other {len(rest)} "
          f"(ours ahead on {sum(d > 0 for d in rest)} of them)")
    lines += [f"\\newcommand{{\\kOneAheadWord}}{{{_NUMWORD.get(us_ahead, us_ahead)}}}",
              f"\\newcommand{{\\kOneNNAheadWord}}{{{_NUMWORD.get(len(k1) - us_ahead, len(k1) - us_ahead)}}}",
              f"\\newcommand{{\\kOneSigWord}}{{{_NUMWORD.get(us_sig1, us_sig1)}}}",
              f"\\newcommand{{\\kOneNNSigWord}}{{{_NUMWORD.get(nn_sig1, nn_sig1)}}}",
              f"\\newcommand{{\\kOneTopDs}}{{{_DISPLAY.get(top_ds, top_ds)}}}",
              *([f"\\newcommand{{\\kOneNNSigDs}}{{{_list(_DISPLAY[ds] for (ds, d, _), a in zip(k1, adj1) if d < 0 and a < 0.05)}}}"]
                if OURS_STEM in ("exemplar", "ridge", "ridge2", "ridge3", "ridge4") else []),
              f"\\newcommand{{\\kOneTopDelta}}{{{abs(top_d):.3f}}}",
              f"\\newcommand{{\\kOneRestOverlap}}{{{rest_mean:.3f}}}",
              # which way the largest difference points; the sentence says "a lead for Exemplar"
              f"% largest K=1 difference is {'OURS ahead' if top_d > 0 else 'NNUNET ahead'} on {top_ds}"]
    if top_d <= 0:
        raise ValueError("the largest K=1 difference is no longer an Exemplar lead; the Results "
                         "sentence that calls it 'a lead for Exemplar' must change")
    # "...a lead for Exemplar on SpheroidJ; without it the overlap lead narrows to X": the wording
    # assumes that dataset is scored by overlap, and that what is left is a lead and a smaller one.
    if metric_of[top_ds] != "fg_iou":
        raise ValueError(f"the largest K=1 difference is on {top_ds}, which is not scored by overlap; "
                         f"'without it the overlap lead' in the Results sentence must change")
    if not 0 < rest_mean < np.mean(group):
        raise ValueError(f"without {top_ds} the K=1 overlap difference is {rest_mean:+.4f} against "
                         f"{np.mean(group):+.4f} with it: no longer a lead that narrows")

    # FAMILY 5 (exemplar stem): ours vs fine-tuned Cellpose-SAM at eight masks, eleven comparisons. The Results name
    # the datasets on which each side is significantly ahead, so the names are macros: they are outcomes of the test.
    if OURS_STEM in ("exemplar", "ridge", "ridge2", "ridge3", "ridge4"):
        require_paired(OURS, CELLPOSE_FT)
        cp = compare(CELLPOSE_FT)
        adjc = holm(np.asarray([p for _, _, p in cp]))
        cp_us = [ds for (ds, d, _), a in zip(cp, adjc) if d > 0 and a < 0.05]
        cp_them = [ds for (ds, d, _), a in zip(cp, adjc) if d < 0 and a < 0.05]
        us_ahead_cp = sum(1 for _, d, _ in cp if d > 0)
        print(f"Cellpose-SAM family: {len(cp)} comparisons, ours ahead on {us_ahead_cp} "
              f"({len(cp_us)} Holm-significant: {cp_us}), Cellpose-SAM significantly on {len(cp_them)}: {cp_them}")
        for (ds, d, p), a in zip(cp, adjc):
            print(f"   {ds:<12} delta {d:+.4f}  raw p {p:.2g}  Holm {a:.2g}")
        names = lambda l: ", ".join(_DISPLAY[x] for x in l[:-1]) + (" and " if len(l) > 1 else "") + _DISPLAY[l[-1]] if l else "none"
        lines += [f"\\newcommand{{\\cpOursAheadWord}}{{{_NUMWORD.get(us_ahead_cp, us_ahead_cp)}}}",
                  f"\\newcommand{{\\cpOursSigWord}}{{{_NUMWORD.get(len(cp_us), len(cp_us))}}}",
                  f"\\newcommand{{\\cpOursSigDs}}{{{names(cp_us)}}}",
                  f"\\newcommand{{\\cpSigWord}}{{{_NUMWORD.get(len(cp_them), len(cp_them))}}}",
                  f"\\newcommand{{\\cpSigDs}}{{{names(cp_them)}}}"]
        # FAMILY 5b (ASG_SEM_CONVPAINT=1; Convpaint added to Table 1 as its own test family):
        # ours vs Convpaint (combo_dino_vgg, final10/convpaint_k8) at eight masks, eleven comparisons.
        if os.environ.get("ASG_SEM_CONVPAINT", "") == "1":
            require_paired(OURS, "convpaint_k8")
            cv = compare("convpaint_k8")
            if len(cv) != len(DATASETS):
                raise ValueError(f"Convpaint family paired {len(cv)} of {len(DATASETS)} datasets")
            adjv = holm(np.asarray([p for _, _, p in cv]))
            cv_us = [ds for (ds, d, _), a in zip(cv, adjv) if d > 0 and a < 0.05]
            cv_them = [ds for (ds, d, _), a in zip(cv, adjv) if d < 0 and a < 0.05]
            cv_ahead = sum(1 for _, d, _ in cv if d > 0)
            print(f"Convpaint family: {len(cv)} comparisons, ours ahead on {cv_ahead} ({len(cv_us)} Holm-significant), "
                  f"Convpaint significantly on {len(cv_them)}: {cv_them}")
            for (ds, d, p), a in zip(cv, adjv):
                print(f"   {ds:<12} delta {d:+.4f}  raw p {p:.2g}  Holm {a:.2g}")
            lines += [f"\\newcommand{{\\cvOursAheadWord}}{{{_NUMWORD.get(cv_ahead, cv_ahead)}}}",
                      f"\\newcommand{{\\cvOursSigWord}}{{{_NUMWORD.get(len(cv_us), len(cv_us))}}}",
                      f"\\newcommand{{\\cvSigWord}}{{{_NUMWORD.get(len(cv_them), len(cv_them))}}}",
                      f"\\newcommand{{\\cvSigDs}}{{{names(cv_them)}}}"]
        # FAMILY 6: the same at ONE mask, against the K=1 budget Table 1's generator chooses (the higher mean over
        # the eleven datasets, a selection in Cellpose-SAM's favour), eleven comparisons.
        cands = {d: np.mean([per_image(d, ds, m).mean() for ds, m in DATASETS])
                 for d in ("cellpose_ft11sem_k1_e800", "cellpose_ft11sem_k1_e6400")}
        cp1_dir = max(cands, key=cands.get)
        require_paired(OURS_K1, cp1_dir)
        cp1 = compare(cp1_dir, ref=OURS_K1)
        adj6 = holm(np.asarray([p for _, _, p in cp1]))
        us6 = sum(1 for (_, d, _), a in zip(cp1, adj6) if d > 0 and a < 0.05)
        them6 = sum(1 for (_, d, _), a in zip(cp1, adj6) if d < 0 and a < 0.05)
        print(f"Cellpose-SAM family at K=1 ({cp1_dir}): ours significantly ahead on {us6}, Cellpose-SAM on {them6}")
        lines += [f"\\newcommand{{\\cpOneSigWord}}{{{_NUMWORD.get(them6, them6)}}}",
                  f"\\newcommand{{\\cpOneOursSigWord}}{{{_NUMWORD.get(us6, us6)}}}"]

    path = os.path.join(OUT, "numbers_stats.tex")
    open(path, "w").write("\n".join(lines) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
