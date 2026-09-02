"""scaleconf fast-screen (CLAUDE.md protocol): current-best vs +scaleconf. TARGET = scale-relevant thin/
small morphologies where matching the bank scale to the object should help (drive vessels, bacteria thin
rods, monuseg dense small nuclei); CONTROL = large/blob morphologies that must NOT regress (spheroidj,
dsb2018). Bar this time is IMPROVE a target (Δ>~+0.01) with NO control regression (Δ>~-0.005) -- the lever
adds capability (right-scaled filters), not just simplification.

SEPARATE cache (reuses cache_bankselect's DINO features -- scaleconf changes only the classical bank, not
DINO) so there is NO write race with the campaign on cache_final10. Jobs run SEQUENTIALLY.
"""
import os
import subprocess
import sys

ROOT = os.environ.get("ASG_REPO_ROOT", "/disk1/prusek/active-segmenter")
PY = os.environ.get("ASG_PY", os.path.expanduser("~/dinov3_env/bin/python"))
DEV = os.environ.get("ASG_SCREEN_DEV", "1")             # A5000 is free while the regen finishes on the A100
CACHE = "/disk1/prusek/cache_bankselect"                # reuse (DINO feats shared); separate from cache_final10
BASE = "head_fusion_best_cgate_film_nobank"
METHODS = [BASE, BASE + "_scaleconf"]
# TARGET (scale-relevant): drive vessels, bacteria thin rods, monuseg dense small nuclei. CONTROL: blobs.
# INTERLEAVED (baseline then scaleconf per dataset) so each dataset's A/B lands as it finishes; bacteria
# LAST because it is the slow pole (dense, ~104 instances/img) and must not block the fast A/B points.
DATASETS = [("drive", "cldice"), ("monuseg", "fg_iou"), ("spheroidj", "fg_iou"),
            ("dsb2018", "fg_iou"), ("bacteria", "fg_iou")]
SEEDS = os.environ.get("ASG_SCREEN_SEEDS", "3")
POOL = 20


def main() -> None:
    jobs = [(m, ds, met) for ds, met in DATASETS for m in METHODS]
    print(f"{len(jobs)} scaleconf-screen jobs (2 methods x {len(DATASETS)} datasets, seeds={SEEDS}, dev={DEV})",
          flush=True)
    failures = []
    for i, (m, ds, met) in enumerate(jobs, 1):
        sd = f"{ROOT}/results/scaleconf_screen/{m}"
        cmd = (f"CUDA_VISIBLE_DEVICES={DEV} PYTHONPATH={ROOT} {PY} {ROOT}/scripts/sota_final.py run "
               f"--method {m} --datasets {ds} --support 8 --pool {POOL} --test 10000 --seeds {SEEDS} "
               f"--res 672 --metric_override {met} --cache {CACHE} --score_dir {sd}")
        print(f"[{i}/{len(jobs)}] {m} {ds} ({met})", flush=True)
        r = subprocess.run(cmd, shell=True)
        if r.returncode != 0:
            failures.append(f"{m}/{ds}")
            print(f"  ! FAILED {m} {ds} (rc={r.returncode})", flush=True)
    # FAIL LOUD: the GO/NO-GO gate is "regresses NO CONTROL". A crashed control arm leaves no score
    # file, the A/B is then computed over the survivors, and the screen could not have failed the lever.
    if failures:
        raise SystemExit(f"SCREEN_INCOMPLETE — {len(failures)}/{len(jobs)} job(s) FAILED: {failures}; "
                         f"the GO/NO-GO verdict is NOT valid on a partial screen")
    print("DONE scaleconf screen", flush=True)


if __name__ == "__main__":
    main()
