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
# The stripped method (gate and FiLM removed, 2026-07-31). Must match Table 1 and the
# K-scaling figure: a stats table computed against a different arm from the one the table
# reports would compare two methods under one name.
# The Results prose names the dataset Exemplar wins on. That name is an OUTCOME of the test, not
# a fixed label -- if the win moved, the sentence would have to move with it. Emitted as a macro
# for the same reason every number here is.
_DISPLAY = {"spheroidj": "SpheroidJ", "rozpad": "Decay", "dsb2018": "DSB2018",
            "monuseg": "MoNuSeg", "ctc_u373": "CTC-U373", "bbbc010": "BBBC010",
            "bacteria": "Bacteria", "drive": "DRIVE", "hrf": "HRF",
            "isbi2012em": "ISBI2012-EM", "fisbe": "FISBE"}
_NUMWORD = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
            6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven"}
OURS = "lean_k8"
FORWARD_PASS = {"SegGPT": "seggpt_k8", "UniverSeg": "universeg_k8",
                # Tyche is scored here by the MEAN of its candidate maps, which uses no ground
                # truth, because every other column of this table is scored without one and an
                # oracle number sitting among them misleads a reader scanning columns. Its own
                # paper reports best-of-K (min_k L_Dice(y_k, y)); that arm exists as
                # `tyche_bestof_k8` and the Baselines paragraph states its result, which does not
                # change the conclusion -- Exemplar leads on all eleven either way.
                "Tyche": "tyche_k8",
                # A TUPLE would mean a baseline with several documented modes, tested at the
                # per-dataset better one so the headline win count is not earned against a weaker
                # configuration than the table prints. INSID3 no longer needs it: the column is the
                # authors' released implementation in the single configuration their published
                # anchor reproduces from (registry C88), not our own two-mode read-out.
                "INSID3": "insid3_k8",
                "Matcher": "matcher_k1"}
NNUNET = "nnunet_k8"
# The 2x2 inputs ablation: same head, same support-derived rules, differing only in
# which input is zeroed. Both arms are rules-on, so the pair isolates the inputs.
BANK_ONLY = "lean_bankonly_k8"
FEATURES_ONLY = "lean_nocls_k8"


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
    nn_ahead = sum(1 for _, d, _ in nn if d < 0)
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
    lines += [f"\\newcommand{{\\nnAheadWord}}{{{_NUMWORD.get(nn_ahead, nn_ahead)}}}",
              f"\\newcommand{{\\oursSigDs}}{{{' and '.join(us_names) if us_names else 'none'}}}",
              f"\\newcommand{{\\nnAhead}}{{{nn_ahead}}}",
              f"\\newcommand{{\\nnSig}}{{{nn_sig}}}",
              # word form: the sentence around it spells its other counts, and "only 6" beside
              # "nine datasets" reads as a typesetting slip rather than a number.
              f"\\newcommand{{\\nnSigWord}}{{{_NUMWORD.get(nn_sig, nn_sig)}}}",
              f"\\newcommand{{\\oursSigVsNN}}{{{us_sig}}}"]

    # FAMILY 3: the two inputs against each other, eleven comparisons. This is the paper's
    # headline measurement, so it gets a declared family of its own rather than riding on
    # FAMILY 1's correction, whose comparisons test a different claim.
    bf = compare(FEATURES_ONLY, ref=BANK_ONLY)
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

    path = os.path.join(OUT, "numbers_stats.tex")
    open(path, "w").write("\n".join(lines) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
