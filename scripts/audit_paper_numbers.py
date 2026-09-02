#!/usr/bin/env python3
"""Recompute EVERY number the ISBI results prose states inline, from the score tree.

WHY THIS EXISTS. Tables and macros in this paper are script-generated, but the results
paragraphs still quote roughly forty numbers in running text: per-dataset accuracies, the
overlap/centreline split means, the nnU-Net deltas at K=1 and K=8, and the held-out versus
design-time split. When the reported arm changed from `oursv3n_k8` (gate + FiLM) to
`stripv3_k8` (both removed) on 2026-07-31, every one of those moved by a little. Hand-editing
forty numbers against a diff is exactly how a paper ends up with two that were missed, and a
reviewer who finds one stops trusting the rest.

This prints each number with the sentence fragment it belongs to, so applying the edit is a
mechanical pass rather than a hunt. It reads the same records the tables read, so a number here
can never disagree with the table it sits beside.

Usage:  ASG_SEM_TREE=/disk1/prusek/active-segmenter/results/final10 python3 audit_paper_numbers.py
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.environ.get("ASG_SEM_TREE", "results/final10")

# The eleven datasets and the metric each is scored by, in the paper's own order. OVERLAP is the
# seven scored by foreground IoU, CENTRELINE the four scored by centreline Dice; the paper reports
# a mean over each because a single panel mean hides that the nnU-Net gap is entirely topological.
DATASETS = [("spheroidj", "fg_iou"), ("rozpad", "fg_iou"), ("dsb2018", "fg_iou"),
            ("monuseg", "fg_iou"), ("ctc_u373", "fg_iou"), ("bbbc010", "fg_iou"),
            ("bacteria", "fg_iou"), ("drive", "cldice"), ("hrf", "cldice"),
            ("isbi2012em", "cldice"), ("fisbe", "cldice")]
OVERLAP = [d for d, m in DATASETS if m == "fg_iou"]
CENTRELINE = [d for d, m in DATASETS if m == "cldice"]

# Four datasets were prepared AFTER the method was frozen and took no part in any design decision.
# The paper reports the nnU-Net gap separately on them because the panel figure flatters us
# otherwise, and that honesty is load-bearing: it is the difference between a measured margin and
# one that design-time choices bought.
HELD_OUT = ["bacteria", "isbi2012em", "fisbe", "bbbc010"]
DESIGN_TIME = [d for d, _ in DATASETS if d not in HELD_OUT]


def seed_means(d, ds):
    """Per-seed mean score for one dataset, or None if the cell is missing.

    Returns one number per seed rather than per image, because every panel-level statement in the
    paper is seed-paired: datasets are averaged within a seed and the deviation is taken across
    those aggregates. Averaging images across datasets instead would weight a 117-image dataset
    like DRIVE thirty times a 4-image one, which is not the quantity the prose claims.
    """
    fs = glob.glob(os.path.join(ROOT, d, f"*__{ds}.json"))
    if not fs:
        return None
    if len(fs) > 1:
        sys.exit(f"{d}/{ds}: {len(fs)} records match; a directory must resolve to exactly one")
    j = json.load(open(fs[0]))
    pi = np.asarray(j["per_image"], float)
    return pi.reshape(len(j["seeds"]), j["test_per_seed"]).mean(1)


def panel(d, subset=None):
    """Seed-paired mean over a subset of datasets, and the count that actually resolved."""
    names = [ds for ds, _ in DATASETS] if subset is None else subset
    got = [(ds, seed_means(d, ds)) for ds in names]
    have = [v for _, v in got if v is not None]
    missing = [ds for ds, v in got if v is None]
    if not have:
        return None, names
    return np.mean(have, axis=0), missing


def fmt(x):
    return "  --  " if x is None else f"{np.mean(x):.3f}"


def line(label, value, missing=()):
    tail = f"   [MISSING {sorted(missing)}]" if len(missing) else ""
    print(f"  {label:<62} {fmt(value)}{tail}")


OURS = {k: f"lean_k{k}" for k in (1, 4, 8, 16)}
NN = {1: "../nnunet_k1", 8: "nnunet_k8", 4: "nnunet_k4", 16: "nnunet_k16"}


def require_reported_arm():
    """Refuse to print an audit in which the reported arm resolved to nothing.

    Without this the script is worse than useless: pointed at a tree that lacks OURS[8] it prints
    '--' into every cell of every section and still exits 0, so a run that verified nothing is
    indistinguishable at the shell from one that verified everything. That is exactly how the local
    checkout -- whose final10 predates the switch to stripv3 -- silently 'confirmed' the paper on
    2026-08-01. The sibling make_ablation_table.py already refuses on a missing record; this brings
    the auditor up to the same standard, because a verifier that can pass vacuously is a liability.
    """
    resolved = {k: seed_means(d, "spheroidj") is not None for k, d in OURS.items()}
    if not resolved[8]:
        sys.exit(f"REFUSING to audit: the reported arm {OURS[8]!r} has no records under {ROOT}. "
                 f"Every 'ours' number below would print as '--' and the run would exit 0, "
                 f"certifying nothing. Point ASG_SEM_TREE at the tree that holds it "
                 f"(the campaign tree on the compute host, not a stale local copy).")
    thin = [k for k, ok in resolved.items() if not ok]
    if thin:
        print(f"WARNING: no records for K={thin} ({[OURS[k] for k in thin]}); "
              f"sections that need them will be incomplete.\n")


print(f"tree: {ROOT}\n")
require_reported_arm()
print("=" * 78)
print("PER-DATASET, K=8 -- the values quoted in the forward-pass and specialist paragraphs")
print("=" * 78)
for ds, metric in DATASETS:
    v = seed_means(OURS[8], ds)
    print(f"  {ds:<14} {metric:<8} {fmt(v)}")

print()
print("=" * 78)
print("PANEL MEANS AT K=8 -- 'we average X on overlap ... Y on centreline'")
print("=" * 78)
for tag, sub in (("all eleven", None), ("overlap (7)", OVERLAP), ("centreline (4)", CENTRELINE)):
    o, mo = panel(OURS[8], sub)
    n, mn = panel(NN[8], sub)
    line(f"ours, {tag}", o, mo)
    line(f"nnU-Net, {tag}", n, mn)
    if o is not None and n is not None:
        print(f"  {'-> delta (ours - nnU-Net)':<62} {np.mean(o) - np.mean(n):+.3f}")

print()
print("=" * 78)
print("THE HELD-OUT SPLIT -- 'X on the four prepared after the method was frozen against Y'")
print("=" * 78)
for tag, sub in (("held out (4)", HELD_OUT), ("design time (7)", DESIGN_TIME)):
    o, _ = panel(OURS[8], sub)
    n, _ = panel(NN[8], sub)
    if o is not None and n is not None:
        print(f"  {tag:<62} {np.mean(o) - np.mean(n):+.3f}")

print()
print("=" * 78)
print("K=1 -- the abstract's headline and the 'same split at one mask' sentence")
print("=" * 78)
for tag, sub in (("all eleven", None), ("overlap (7)", OVERLAP), ("centreline (4)", CENTRELINE)):
    o, mo = panel(OURS[1], sub)
    n, mn = panel(NN[1], sub)
    line(f"ours K=1, {tag}", o, mo)
    if o is not None and n is not None:
        print(f"  {'-> delta vs nnU-Net':<62} {np.mean(o) - np.mean(n):+.3f}")
for ds in ("spheroidj", "fisbe"):
    o, n = seed_means(OURS[1], ds), seed_means(NN[1], ds)
    if o is not None and n is not None:
        print(f"  K=1 {ds:<58} {np.mean(o):.3f} vs nnU-Net {np.mean(n):.3f} "
              f"({np.mean(o) - np.mean(n):+.3f}, ours spread {np.std(o):.3f} vs {np.std(n):.3f})")

print()
print("=" * 78)
print("THE K CURVE -- '89% of what we ourselves reach at sixteen'")
print("=" * 78)
ref = None
for k in (1, 4, 8, 16):
    v, miss = panel(OURS[k])
    line(f"ours K={k}", v, miss)
    if k == 16:
        ref = v
k1, _ = panel(OURS[1])
if ref is not None and k1 is not None:
    print(f"  {'-> K=1 as a fraction of K=16':<62} {100 * np.mean(k1) / np.mean(ref):.0f}%")
else:
    print("  -> K=16 incomplete: the '88%' claim CANNOT be restated yet")
