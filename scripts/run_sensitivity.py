"""Loss-weight SENSITIVITY scan: how flat is the landscape around the self-config heuristic?

Runs best_v2 at several global multipliers ASG_LOSS_SCALE on the adaptive auxiliary loss weights
(clDice/boundary/Tversky relative to the region anchor; 1.0 = the heuristic). If the query score is
flat across a wide multiplier range, the exact weights are not critical and the closed-form rules are
defensible ("no per-dataset tuning"); if sharp, tuning matters. One decisive cheap experiment.

Runs on the datasets where the adaptive terms are actually active (thin vessels, crowded nuclei) plus
a blob control. Writes results/sensitivity/scale_{lam}/ per (scale, dataset); analyse the spread offline.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.environ.get("ASG_REPO_ROOT", "/disk1/prusek/active-segmenter")
PY = os.environ.get("ASG_PY", os.path.expanduser("~/dinov3_env/bin/python"))
CACHE = os.environ.get("ASG_FEAT_CACHE", "/disk1/prusek/cache_sens")
METHOD = "head_fusion_best_cgate_film_nobank"
SCALES = [float(x) for x in os.environ.get("ASG_SCALES", "0.5,1.0,2.0").split(",")]
SEEDS = os.environ.get("ASG_SENS_SEEDS", "5")   # sota_final --seeds is a COUNT (0..N-1), not a list
# dataset -> semantic metric (adaptive terms are most active on thin/crowded; dsb2018 = blob control)
DATASETS = {"drive": "cldice", "hrf": "cldice", "monuseg": "fg_iou", "dsb2018": "fg_iou"}
POOL = 20


def main() -> None:
    jobs = [(s, ds, m) for s in SCALES for ds, m in DATASETS.items()]
    print(f"{len(jobs)} sensitivity jobs ({len(SCALES)} scales x {len(DATASETS)} datasets, seeds={SEEDS})",
          flush=True)
    failures = []
    for i, (scale, ds, metric) in enumerate(jobs, 1):
        sd = f"{ROOT}/results/sensitivity/scale_{scale}"
        # PYTHONPATH is REQUIRED: sota_final.py imports active_segmenter, so without it every job dies
        # on ImportError, each prints one FAILED line, and the scan would still report DONE.
        cmd = (f"ASG_LOSS_SCALE={scale} PYTHONPATH={ROOT} {PY} {ROOT}/scripts/sota_final.py run "
               f"--method {METHOD} --datasets {ds} --support 8 --pool {POOL} --test 10000 "
               f"--seeds {SEEDS} --score_dir {sd} --res 672 --metric_override {metric} --cache {CACHE}")
        print(f"[{i}/{len(jobs)}] scale={scale} {ds} ({metric})", flush=True)
        r = subprocess.run(cmd, shell=True)
        if r.returncode != 0:
            failures.append(f"scale={scale}/{ds}")
            print(f"  ! FAILED scale={scale} {ds} (rc={r.returncode})", flush=True)
    # FAIL LOUD: a partial scan must never exit 0 -- the landscape would be read as complete while the
    # arm that would have shown the sensitivity is simply absent.
    if failures:
        raise SystemExit(f"SCAN_INCOMPLETE — {len(failures)}/{len(jobs)} job(s) FAILED: {failures}")
    print("DONE sensitivity scan", flush=True)


if __name__ == "__main__":
    main()
