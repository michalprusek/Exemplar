"""Install the semantic-fix K=1/K=4 reruns into the score tree, or refuse and say why.

WHY A SCRIPT. Replacing reported cells by hand is how a tree ends up holding two run environments.
Every check that has to pass is written down here and run every time, so the swap is auditable and
so a failure names itself instead of being noticed three steps later in a paper number.

The reruns exist because `_detect_n_classes` built a MULTI-CLASS head on instance-labelled support
at K=1 (CTC-U373 10/10 seeds, FISBE 9/10) and at K=4 (CTC-U373 3/10). Exemplar is a semantic
segmenter, so that is a defect, and the whole K=1 and K=4 panels were re-run under the fix rather
than the three affected cells being spliced in beside eight older ones.

GATES, all of which must pass before anything is written:
  1. every rerun record carries the metric that dataset is reported under;
  2. every rerun record's split fingerprint matches the campaign's -- same test images, not merely
     the same count;
  3. all eleven datasets are present for each K;
  4. the control datasets AS A GROUP show no systematic shift. This is the gate that matters, and
     per-dataset thresholds are the wrong instrument for it. Run-to-run noise here is real and it
     GROWS as K falls, because a K=1 fit sees one image and a K=8 fit averages over eight; measured
     on cells where the fix provably cannot apply, mean |delta| is 0.0004 at K=8 (ctc_u373 and
     isbi2012em), 0.0035 at K=4 and 0.0061 at K=1, and three same-config repeats of fisbe at K=1
     gave 0.6138 / 0.6240 / 0.6152. So an individual control moving 0.01 at K=1 proves nothing.
     What WOULD prove a broken environment is a shift with a SIGN: if the reruns were measuring
     different code or different data, the controls' mean delta would be displaced rather than
     centred. So the gate is on the group mean, with a generous per-dataset bound only to catch a
     single catastrophic cell.

    python scripts/install_semfix.py --src /tmp/semfix        # check only
    python scripts/install_semfix.py --src /tmp/semfix --write
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

METRIC = {"spheroidj": "fg_iou", "rozpad": "fg_iou", "dsb2018": "fg_iou", "monuseg": "fg_iou",
          "ctc_u373": "fg_iou", "bbbc010": "fg_iou", "bacteria": "fg_iou", "drive": "cldice",
          "hrf": "cldice", "isbi2012em": "cldice", "fisbe": "cldice"}
FIXED_CELLS = {1: {"ctc_u373", "fisbe"}, 4: {"ctc_u373"}}   # where the multi-class head actually fired
# Calibrated from the K=8 controls and the fisbe repeats above, not guessed. Per-dataset bounds are
# deliberately loose: they exist to catch one broken cell, not to adjudicate noise.
TOL_ABS = {1: 0.030, 4: 0.020}      # a single control may not move more than this
TOL_GROUP_MEAN = 0.005              # the controls' MEAN delta must stay centred

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="dir holding k1/ and k4/ rerun records")
ap.add_argument("--tree", default="results/final10")
ap.add_argument("--write", action="store_true", help="actually install (default: check only)")
args = ap.parse_args()


def stats(path, metric):
    d = json.load(open(path))
    if d.get("metric") != metric:
        return None, f"metric {d.get('metric')!r}, expected {metric!r}"
    ns, t = len(d["seeds"]), d["test_per_seed"]
    pi = np.asarray(d["per_image"], float).reshape(ns, t)
    return (pi.mean(), pi.mean(1).std(), d.get("split_fp")), None


fatal = []
for K in (1, 4):
    src = sorted(glob.glob(os.path.join(args.src, f"k{K}", "*.json")))
    have = {f.split("__")[-1].replace(".json", "") for f in src}
    missing = set(METRIC) - have
    if missing:
        fatal.append(f"K={K}: missing {sorted(missing)}")
    print(f"\n=== K={K}  ({len(src)}/11) ===")
    deltas = []
    print(f"  {'dataset':<12}{'campaign':>10}{'rerun':>9}{'delta':>9}{'own sd':>9}  verdict")
    for f in src:
        ds = f.split("__")[-1].replace(".json", "")
        met = METRIC[ds]
        new, err = stats(f, met)
        if err:
            fatal.append(f"K={K} {ds}: {err}"); print(f"  {ds:<12} {err}"); continue
        old = None
        for g in glob.glob(os.path.join(args.tree, f"stripv3_k{K}", f"*__{ds}.json")):
            o, e = stats(g, met)
            if o: old = o
        if old is None:
            fatal.append(f"K={K} {ds}: no campaign record"); continue
        if old[2] != new[2]:
            fatal.append(f"K={K} {ds}: split fingerprint {old[2]} vs {new[2]} -- DIFFERENT TEST IMAGES")
            verdict = "SPLIT MISMATCH"
        elif ds in FIXED_CELLS[K]:
            verdict = "fixed cell"
        elif abs(new[0] - old[0]) <= TOL_ABS[K]:
            verdict = "control ok"
            deltas.append(new[0] - old[0])
        else:
            fatal.append(f"K={K} {ds}: control moved {new[0]-old[0]:+.4f}, over the {TOL_ABS[K]} bound")
            verdict = "CONTROL DRIFT"
        print(f"  {ds:<12}{old[0]:10.4f}{new[0]:9.4f}{new[0]-old[0]:+9.4f}{old[1]:9.4f}  {verdict}")

    if deltas:
        gm = float(np.mean(deltas))
        ok = abs(gm) <= TOL_GROUP_MEAN
        print(f"  {len(deltas)} controls: mean delta {gm:+.4f}, mean |delta| {float(np.mean(np.abs(deltas))):.4f}"
              f"  -> {'centred, no systematic shift' if ok else 'SYSTEMATIC SHIFT'}")
        if not ok:
            fatal.append(f"K={K}: controls' mean delta {gm:+.4f} exceeds {TOL_GROUP_MEAN} -- the reruns "
                         f"are measuring something systematically different, not noise")

if fatal:
    print("\nREFUSING TO INSTALL:")
    for m in fatal:
        print("  " + m)
    sys.exit(1)
print("\nall gates pass")
if not args.write:
    print("check-only; pass --write to install")
    sys.exit(0)
for K in (1, 4):
    dst = os.path.join(args.tree, f"stripv3_k{K}")
    for f in sorted(glob.glob(os.path.join(args.src, f"k{K}", "*.json"))):
        shutil.copy2(f, os.path.join(dst, os.path.basename(f)))
    print(f"installed K={K} into {dst}")
