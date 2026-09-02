"""Aggregate the multi-run ablation (P0-1 + P0-3): control per-run cudnn nondeterminism by averaging over
R independent training runs, and report the incremental ablation with run-level significance.

Layout: results/scores_mr/<tag>_run<r>/<method>__<dataset>.json  (tag ∈ A0..A3, r ∈ 1..R).
Each run's per-image scores → one "run mean" (mean over all 6-seed × test images). We then report
mean ± std OVER RUNS per (tag, dataset), so the reported spread IS the nondeterminism floor, and a
delta only counts (*) if it clears that floor (|Δ| > 2× the LARGER group's run-std, 1e-6 floor) AND both
tags have >=2 runs — a single-run tag has std 0, so without the >=2 guard any delta would spuriously star.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

MR = "results/scores_mr"
TAGS = ["A0", "A1", "A2", "A3"]
LABEL = {"A0": "uni_adapt", "A1": "+adapt-loss", "A2": "+superres", "A3": "+native(FULL)"}
DATASETS = ["spheroid", "spheroidj", "dsb2018", "rozpad", "kvasir", "hrf", "microtubules",
            "drive", "isbi2012em", "monuseg", "ctc_u373"]


def load():
    data = defaultdict(lambda: defaultdict(list))   # tag -> dataset -> [run means]
    for tag in TAGS:
        for rundir in sorted(glob.glob(os.path.join(MR, f"{tag}_run*"))):
            for fp in glob.glob(os.path.join(rundir, "*.json")):
                d = json.load(open(fp))
                data[tag][d["dataset"]].append(float(np.mean(d["per_image"])))
    return data


def main():
    data = load()
    print("\n===== MULTI-RUN (mean over runs ± std over runs; run = 6-seed mean over 144 imgs) =====")
    print(f"{'dataset':13s}" + "".join(f"{LABEL[t]:>17s}" for t in TAGS) + "   (#runs)")
    for ds in DATASETS:
        row = f"{ds:13s}"
        counts = []
        for tag in TAGS:
            v = data[tag].get(ds, [])
            if v:
                counts.append(len(v))
            row += f"{np.mean(v):>11.3f}±{np.std(v):<5.3f}" if v else f"{'—':>17s}"
        # show MIN..MAX run count so a single-run tag (unreliable std) is visible, not hidden by the max
        rng = f"{min(counts)}" if counts and min(counts) == max(counts) else \
              (f"{min(counts)}-{max(counts)}" if counts else "0")
        print(row + f"   (runs {rng})")

    print("\n===== INCREMENTAL ABLATION (Δ mean-over-runs; * = |Δ| > 2·run-std = clears noise floor) =====")
    pairs = [("A0", "A1", "+adapt-loss"), ("A1", "A2", "+superres"),
             ("A2", "A3", "+native"), ("A0", "A3", "TOTAL")]
    print(f"{'dataset':13s}" + "".join(f"{lab:>16s}" for _, _, lab in pairs))
    for ds in DATASETS:
        row = f"{ds:13s}"
        for lo, hi, _ in pairs:
            a, b = data[hi].get(ds, []), data[lo].get(ds, [])
            if a and b:
                dlt = np.mean(a) - np.mean(b)
                noise = max(np.std(a), np.std(b), 1e-6)
                star = len(a) >= 2 and len(b) >= 2 and abs(dlt) > 2 * noise  # no star from single-run std=0
                row += f"{dlt:>+15.3f}{'*' if star else ' '}"
            else:
                row += f"{'—':>16s}"
        print(row)
    print("\nMR_AGG_DONE")


if __name__ == "__main__":
    main()
