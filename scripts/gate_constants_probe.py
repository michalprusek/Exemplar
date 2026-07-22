#!/usr/bin/env python
"""Are the self-configuring gates decided by constants fitted to this panel, or by the data?

Three of the gates compare a support-derived descriptor against a hardcoded number:

    thin_max_side = 1500    # native classical bank ON only below this image size
    thin_ref      = 0.4     # ... and only above this mean tubularity
    color_margin  = 1.05    # switch colour channel only if it beats gray by this factor

`thin_max_side` is the worst of them: the registry records that it was placed between HRF (3504 px)
and the rest of the panel because HRF fell 0.607 -> 0.469. A threshold positioned to separate one
dataset from the others is per-dataset tuning no matter which descriptor it is routed through, and
it carries no promise whatsoever about the eleventh dataset.

This probe evaluates PRINCIPLED replacements against the fitted ones on every registry dataset, and
reports where they agree and where they diverge. It trains nothing and needs no GPU: the gates are
decided from the K support masks at fit time, so the decision can be reproduced directly.

    REPLACEMENT 1 (size). The comment explaining thin_max_side says the failure is a distribution
    shift between a NATIVE-resolution classical bank and a head trained on features capped at
    `max_side` (1536). What actually matters is therefore whether that cap downscales the image at
    all -- not an absolute pixel count. So the rule becomes `mean_side <= self.max_side`, which
    introduces no new number: below the cap, "native" and "capped" are the same image and the shift
    is zero BY CONSTRUCTION. That the fitted 1500 landed within 2.4% of the architectural 1536 is
    evidence the constant was approximating exactly this quantity.

    REPLACEMENT 2 (colour). `best > gray * 1.05` asks whether an improvement clears an invented
    margin. The separability is already measured per support image, so we have a sample, and the
    honest question is whether the improvement is larger than its own sampling noise. The rule
    becomes: switch only if the channel wins on a majority of support images AND the median gain
    exceeds the standard error across them. No constant, and it adapts to K on its own -- more
    support makes it more willing to switch, which is the correct behaviour.

    Unlike the size rule, this one CANNOT be assumed behaviour-neutral, and the probe reports it
    across K for exactly that reason. A standard error needs at least two support images, so at K=1
    there is no noise estimate to compare against and the rule must abstain. The reported campaign
    includes K=1, so a divergence there is a real change of results, not a change of justification.
    Measure first; only then decide whether the change is worth re-running for.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--support", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max_side", type=int, default=1536, help="the head's own training cap")
    args = ap.parse_args()

    from active_segmenter.eval.registry import PANEL, load_dataset
    # thin_gate is IMPORTED, never re-expressed here. This script is the evidence behind the paper's
    # "no per-dataset tuning" claim, so a local copy of the rule could certify something the
    # segmenter does not actually run.
    from active_segmenter.segment.head_fusion_backend import _tubularity, thin_gate

    POOL = {"ctc_u373": 15, "isbi2012em": 16, "fisbe": 16}
    print(f"{'dataset':<12}{'mean_side':>10}{'tubul':>8}   {'fitted':>8} {'principled':>11}   verdict")
    disagree = []
    for name, spec in PANEL.items():
        try:
            pool, _ = load_dataset(spec, POOL.get(name, 20), 4, seed=0)
        except Exception as exc:                       # a missing dataset must name itself
            print(f"{name:<12} LOAD FAILED: {type(exc).__name__}: {exc}")
            continue
        sides, tubs = [], []
        for seed in range(args.seeds):
            k = min(args.support, len(pool))
            idx = np.random.default_rng(seed).choice(len(pool), k, replace=False)
            shots = [pool[i] for i in idx]
            sides.append(float(np.mean([max(np.asarray(im).shape[:2]) for im, _ in shots])))
            tubs.append(float(np.mean([_tubularity(np.asarray(lb)) for _, lb in shots])))
        ms, mt = float(np.mean(sides)), float(np.mean(tubs))

        fitted = (mt > 0.4) and (ms < 1500)            # the OLD rule, kept verbatim as the comparison
        principled = thin_gate(mt, ms, 0.4, args.max_side)   # what the segmenter now runs
        mark = "same" if fitted == principled else "DIVERGES"
        if fitted != principled:
            disagree.append(name)
        print(f"{name:<12}{ms:>10.0f}{mt:>8.3f}   {str(fitted):>8} {str(principled):>11}   {mark}")

    print(f"\nsize-gate divergences vs the fitted constant: {disagree or 'none'}")
    print("A principled rule that reproduces every fitted decision on the panel it was fitted to,\n"
          "while being defined by the model's own training cap rather than by a dataset, is strictly\n"
          "the better claim: it says what happens on the eleventh dataset, and 1500 does not.")

    _probe_colour(args, PANEL, load_dataset, POOL)
    return 0


class _Shot:
    """load_dataset yields (image, label) pairs; channel_separability reads .image/.label_map."""

    __slots__ = ("image", "label_map")

    def __init__(self, image, label_map):
        self.image, self.label_map = image, label_map


def _colour_rules(scores, n_used, margin):
    """Return ``(fitted_choice, principled_choice)`` for one support draw.

    FITTED: the winner must beat gray by a factor of ``margin`` (1.05), a number nobody measured.
    PRINCIPLED: the winner must beat gray on a MAJORITY of support images AND its median per-image
    gain must exceed the standard error of that gain across them -- i.e. the improvement has to be
    larger than its own sampling noise. With fewer than two usable images there is no noise estimate,
    so the rule abstains and keeps gray, which is why K=1 is reported separately below.
    """
    from active_segmenter.segment.head_fusion_backend import colour_gate

    # IMPORTED, never re-expressed: the fitted rule must be the one the segmenter runs, or this
    # probe certifies a comparison against something no code path takes.
    fitted, med = colour_gate(scores, n_used, margin)
    eligible = {k: v for k, v in scores.items() if len(v) == n_used}
    if not eligible:
        return "gray", "gray"
    best = max(med, key=med.get)

    if best == "gray" or "gray" not in eligible or n_used < 2:
        return fitted, "gray"
    d = np.asarray(eligible[best], float) - np.asarray(eligible["gray"], float)
    wins = int((d > 0).sum()) > n_used / 2
    se = float(d.std(ddof=1)) / np.sqrt(n_used)
    principled = best if (wins and float(np.median(d)) > se) else "gray"
    return fitted, principled


def _probe_colour(args, PANEL, load_dataset, POOL):
    """Does the noise-based colour rule reproduce the fitted 1.05 margin, and at which K?"""
    from active_segmenter.segment.head_fusion_backend import _color_channels, channel_separability

    ks = [1, 4, 8, 16]
    print(f"\n\n{'dataset':<12}{'K':>4}  {'fitted(1.05)':>13} {'principled':>12}   verdict")
    diverge = []
    for name, spec in PANEL.items():
        try:
            pool, _ = load_dataset(spec, POOL.get(name, 20), 4, seed=0)
        except Exception as exc:
            print(f"{name:<12} LOAD FAILED: {type(exc).__name__}: {exc}")
            continue
        for k in ks:
            kk = min(k, len(pool))
            fit_c, pri_c = [], []
            for seed in range(args.seeds):
                idx = np.random.default_rng(seed).choice(len(pool), kk, replace=False)
                shots = [_Shot(pool[i][0], pool[i][1]) for i in idx]
                # mirror the backend's monochrome short-circuit exactly: a mono dataset never
                # reaches either rule, so counting it as agreement would pad the result
                if sum(1 for s in shots if len(_color_channels(s.image)) > 1) <= len(shots) / 2:
                    fit_c.append("gray"); pri_c.append("gray"); continue
                sc, n_used = channel_separability(shots)
                f, p = _colour_rules(sc, n_used, 1.05)
                fit_c.append(f); pri_c.append(p)
            f_s, p_s = "/".join(sorted(set(fit_c))), "/".join(sorted(set(pri_c)))
            same = f_s == p_s
            if not same:
                diverge.append(f"{name}@K{k}")
            print(f"{name:<12}{kk:>4}  {f_s:>13} {p_s:>12}   {'same' if same else 'DIVERGES'}")

    print(f"\ncolour-gate divergences vs the fitted margin: {diverge or 'none'}")
    if diverge:
        print("NOT behaviour-neutral. Every divergence listed is a dataset/K whose reported numbers\n"
              "would change, so this replacement must NOT be applied to a finished campaign without\n"
              "re-running those cells. Decide with the cost of the re-run in hand.")
    else:
        print("Behaviour-neutral on the panel: the fitted 1.05 margin can be replaced by the sample's\n"
              "own noise with no change to any reported number, removing a second invented constant.")


if __name__ == "__main__":
    sys.exit(main())
