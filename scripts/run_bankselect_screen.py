"""bankselect fast-screen (CLAUDE.md protocol): current-best vs +bankselect on 2 TARGET + 2 CONTROL
datasets, 3 seeds (directional). GO = improves >=1 target by >~+0.01 AND regresses NO control beyond
~-0.005 (bar is DON'T HURT — the lever is a simplification, one Fisher rule for input+bank).

SEPARATE cache (cache_bankselect) so there is NO write race with a concurrently running campaign on
cache_final10 (the documented race that produced irreproducible numbers). Jobs run SEQUENTIALLY (one at a
time), so the two arms share cache_bankselect without an intra-screen write race.
"""
import os
import subprocess
import sys

ROOT = os.environ.get("ASG_REPO_ROOT", "/disk1/prusek/active-segmenter")
PY = os.environ.get("ASG_PY", os.path.expanduser("~/dinov3_env/bin/python"))
DEV = os.environ.get("ASG_SCREEN_DEV", "0")             # A100 has the headroom while the regen winds down
CACHE = "/disk1/prusek/cache_bankselect"                # SEPARATE from cache_final10 (no race with the regen)
BASE = "head_fusion_best_cgate_film_nobank"
METHODS = [BASE, BASE + "_bankselect"]                  # current best vs +bankselect (default-off lever)
# 2 TARGET (bank-heavy: vessels Frangi + dense H&E) + 2 CONTROL (must not regress: semantic blob + instance)
DATASETS = {"drive": "cldice", "monuseg": "fg_iou", "spheroidj": "fg_iou", "dsb2018": "fg_iou"}
SEEDS = os.environ.get("ASG_SCREEN_SEEDS", "3")
POOL = 20


def main() -> None:
    jobs = [(m, ds, met) for m in METHODS for ds, met in DATASETS.items()]
    print(f"{len(jobs)} bankselect-screen jobs (2 methods x 4 datasets, seeds={SEEDS}, dev={DEV})", flush=True)
    failures = []
    for i, (m, ds, met) in enumerate(jobs, 1):
        sd = f"{ROOT}/results/bankselect_screen/{m}"
        cmd = (f"CUDA_VISIBLE_DEVICES={DEV} PYTHONPATH={ROOT} {PY} {ROOT}/scripts/sota_final.py run "
               f"--method {m} --datasets {ds} --support 8 --pool {POOL} --test 10000 --seeds {SEEDS} "
               f"--res 672 --metric_override {met} --cache {CACHE} --score_dir {sd}")
        print(f"[{i}/{len(jobs)}] {m} {ds} ({met})", flush=True)
        r = subprocess.run(cmd, shell=True)
        if r.returncode != 0:
            failures.append(f"{m}/{ds}")
            print(f"  ! FAILED {m} {ds} (rc={r.returncode})", flush=True)
    # FAIL LOUD: the GO/NO-GO gate is "regresses NO CONTROL". If a control arm crashed, no score file
    # exists for it, the A/B is computed over the datasets that survived, and the screen would be
    # structurally incapable of failing the lever -- while printing DONE and exiting 0.
    if failures:
        raise SystemExit(f"SCREEN_INCOMPLETE — {len(failures)}/{len(jobs)} job(s) FAILED: {failures}; "
                         f"the GO/NO-GO verdict is NOT valid on a partial screen")
    print("DONE bankselect screen", flush=True)


if __name__ == "__main__":
    main()
