"""Emit the frozen-backbone control's numbers as LaTeX macros for the Results section.

WHY THIS CONTROL HAS ITS OWN GENERATOR. The paper states plainly that we fit a head by gradient
descent where UniverSeg, Tyche, SegGPT and INSID3 consume the support set in a forward pass. That
invites one obvious question -- how much of the margin follows from having trained anything at all --
and the component ablation cannot answer it, because that table was measured on the base head (a wide
3x3 stem, sixty fixed epochs, no regularisation) and mixes the architecture question with the
self-configuration question. This control instead holds the REPORTED training regime fixed and strips
everything else: the same head, trained the same way on the same K masks, over the frozen backbone
alone, with the classical prior bank zeroed and every support-derived rule switched off.

The three arms it compares are the three ways of reading the same eight masks:

  ilastik-RF   classical native-resolution priors, no frozen semantics  (a forest over the bank)
  probe        frozen semantics, no priors and no self-configuration    (this control)
  ours         both                                                     (the reported method)

WHY MACROS RATHER THAN NUMBERS TYPED INTO main.tex. Every figure in this manuscript is regenerated
from the result tree so that a re-run is a mechanical pass rather than a hunt through prose. A number
typed by hand is a number nobody can re-derive, and this project has already been bitten by a stale
one. `\\input{numbers_probe.tex}` in the preamble makes each value a single point of truth.

Fail-loud contract, matching make_semantic_tables.py and make_ablation_table.py: every dataset must
be present in every arm, the metric string must agree between arms for a given dataset (a mismatch is
how a comparison gets silently skipped), the seed count must be uniform, and nothing is written if
any of that fails. A partial file is worse than none, because it still compiles.

  ASG_SEM_TREE=/disk1/prusek/active-segmenter/results \\
  ASG_SEM_OUT=/disk1/prusek/active-segmenter/paper/isbi2027 \\
  python scripts/make_probe_numbers.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.environ["ASG_SEM_TREE"]
OUT = os.environ["ASG_SEM_OUT"]

# The eleven-dataset panel, in the order the paper introduces the morphologies.
PANEL = ["spheroidj", "dsb2018", "monuseg", "rozpad", "ctc_u373", "bbbc010",
         "bacteria", "drive", "hrf", "isbi2012em", "fisbe"]

# (label, directory, method-name prefix). The probe arm's method name is the full token string, so we
# match on the directory and take whatever single record each dataset has.
ARMS = {
    # SWITCHED 2026-08-26 (registry C80) from `probe_k8`, which carried the competitive gate and
    # FiLM on top of the reported head -- the two components the paper measured and rejected -- so
    # the number needed a caveat in the prose. `probev3_k8` is the reported head with the bank
    # zeroed and both support-derived rules off, and nothing added. The value barely moves
    # (0.6723 -> 0.6712), which is itself the finding: those modules do nothing on a bare backbone
    # either.
    "probe": os.path.join(ROOT, "final10", "probev3_k8"),
    # THE 2x2, added 2026-08-27 (C82) and corrected the same day. All three cells must share the
    # REPORTED configuration and differ only in which input is zeroed, or the comparison mixes the
    # 0.009 the support-derived rules are worth into the feature-family contrast. `probe` above is
    # NOT such a cell: it also switches both rules off, which makes it the right control for "what
    # does a bare frozen-feature head do" and the wrong one for "what are the features worth".
    # REPOINTED 2026-09-01 to the LEAN cells. The three arms have to be the same method or the
    # 2x2 measures the method change, not the inputs: while `ours` was `stripv3_k8` and these two
    # were `ablv3_*`, the fused arm carried the support-derived rules the single-source arms also
    # carried, but the paper's reported method no longer has them. All three now read the reported
    # configuration and differ ONLY in which input is zeroed, which is what the prose claims.
    "featonly": os.path.join(ROOT, "final10", "lean_nocls_k8"),      # bank zeroed
    "bankonly": os.path.join(ROOT, "final10", "lean_bankonly_k8"),   # DINO zeroed
    # RE-POINTED 2026-07-31 to the stripped method's own K=8 records, the same directory
    # Table 1 and make_stats.py read. `winner_panel_norm` held the gate+FiLM arm, so leaving
    # it here would make \oursMean disagree with the table on the same page.
    "ours": os.path.join(ROOT, "final10", "lean_k8"),
    "ilastik": os.path.join(ROOT, "ilastik_k8"),
}


def load_arm(path):
    """-> {dataset: (metric, mean_over_images, n_seeds)}; raises rather than returning a partial arm."""
    if not os.path.isdir(path):
        raise SystemExit(f"REFUSING: arm directory does not exist: {path}")
    out = {}
    for f in sorted(glob.glob(os.path.join(path, "*.json"))):
        j = json.load(open(f))
        ds = j["dataset"]
        if ds in out:
            raise SystemExit(f"REFUSING: two records for {ds} in {path}; the arm is ambiguous")
        out[ds] = (j["metric"], float(np.mean(j["per_image"])), len(j["seeds"]))
    return out


def main():
    arms = {k: load_arm(v) for k, v in ARMS.items()}

    # Guard 1: full coverage. A mean over a subset is not the quantity the paper reports, and a
    # silently short panel is exactly how an arm gets to look better than it is.
    for name, a in arms.items():
        missing = [d for d in PANEL if d not in a]
        if missing:
            raise SystemExit(f"REFUSING: arm {name!r} is missing {missing} of the eleven-dataset panel")

    # Guard 2: metric agreement. If our arm recorded fg_iou where a baseline recorded cldice, the
    # comparison is between two different quantities and reads as a difference in method.
    for d in PANEL:
        metrics = {name: a[d][0] for name, a in arms.items()}
        if len(set(metrics.values())) != 1:
            raise SystemExit(f"REFUSING: metric disagreement on {d}: {metrics}")

    # Guard 3: uniform seed count within an arm, so a mean is not weighted by how much each dataset ran.
    for name, a in arms.items():
        seeds = {a[d][2] for d in PANEL}
        if len(seeds) != 1:
            raise SystemExit(f"REFUSING: arm {name!r} has uneven seed counts across datasets: {seeds}")

    means = {name: float(np.mean([a[d][1] for d in PANEL])) for name, a in arms.items()}
    deltas = {d: arms["ours"][d][1] - arms["probe"][d][1] for d in PANEL}
    worst = max(deltas, key=lambda d: deltas[d])
    best = min(deltas, key=lambda d: deltas[d])

    # per-dataset: does the fusion beat max(features, bank)?
    n_beats = sum(1 for d in PANEL
                  if arms["ours"][d][1] > max(arms["featonly"][d][1], arms["bankonly"][d][1]))
    _NUMWORD = {8: "eight", 9: "nine", 10: "ten", 11: "all eleven"}
    lines = [
        "% Generated by scripts/make_probe_numbers.py from the campaign tree. Do not hand-edit.",
        "% The frozen-backbone control: same head, same training regime, same K masks, but the",
        "% classical prior bank zeroed and every support-derived rule switched off.",
        f"\\newcommand{{\\probeMean}}{{{means['probe']:.3f}}}",
        f"\\newcommand{{\\oursMean}}{{{means['ours']:.3f}}}",
        f"\\newcommand{{\\ilastikMean}}{{{means['ilastik']:.3f}}}",
        f"\\newcommand{{\\probeGap}}{{{means['ours'] - means['probe']:.3f}}}",
        f"\\newcommand{{\\ilastikGap}}{{{means['ours'] - means['ilastik']:.3f}}}",
        # The 2x2 read on ONE architecture: what each source adds to the other. Both directions are
        # reported because "complementary" is a two-way word and one direction is not evidence for it.
        f"\\newcommand{{\\featOnlyMean}}{{{means['featonly']:.3f}}}",
        f"\\newcommand{{\\bankOnlyMean}}{{{means['bankonly']:.3f}}}",
        f"\\newcommand{{\\bankOverFeat}}{{{means['bankonly'] - means['featonly']:.3f}}}",
        # UNIFORM POSITIVITY is what "complementary" actually asserts: the fusion must beat the
        # better of its two inputs on every dataset, not merely on the mean, or the claim reduces to
        # "averaging helps". Emitted as a count so the prose cannot drift from it.
        f"\\newcommand{{\\fusedBeatsBoth}}{{{_NUMWORD.get(n_beats, n_beats)}}}",
        # The fitted head's worth OVER the same bank read by a random forest. Both terms are already
        # on this page; the difference was typed by hand and survived a method change that moved both.
        f"\\newcommand{{\\headOverRF}}{{{means['bankonly'] - means['ilastik']:.3f}}}",
        f"\\newcommand{{\\probeWorstDs}}{{{worst}}}",
        f"\\newcommand{{\\probeWorstGap}}{{{deltas[worst]:.3f}}}",
        f"\\newcommand{{\\probeBestDs}}{{{best}}}",
        f"\\newcommand{{\\probeBestGap}}{{{deltas[best]:.3f}}}",
        f"\\newcommand{{\\probeSpread}}{{{deltas[worst] / deltas[best]:.0f}}}",
    ]
    path = os.path.join(OUT, "numbers_probe.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"wrote {path}")
    print(f"  ilastik-RF (bank, no semantics)      {means['ilastik']:.4f}")
    print(f"  probe      (semantics, no bank/rules) {means['probe']:.4f}")
    print(f"  ours       (both)                     {means['ours']:.4f}")
    print(f"  per-dataset gap ours-probe: {best} {deltas[best]:+.4f} .. {worst} {deltas[worst]:+.4f}"
          f"  ({deltas[worst] / deltas[best]:.1f}x spread)")


if __name__ == "__main__":
    main()
