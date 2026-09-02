#!/bin/bash
# GPU1 (A5000): Tyche + UniverSeg re-scored on foreground-IoU for the instance datasets (dsb, monuseg).
cd /disk1/prusek/active-segmenter
export PYTHONPATH=/disk1/prusek/active-segmenter
export CUDA_VISIBLE_DEVICES=1
PY=~/dinov3_env/bin/python
CACHE=/disk1/prusek/asg_cache_fgk_base

# FAIL-LOUD run helper: clean the score_dir first, and on failure remove the partial dir + print a marker,
# so a failed run can never be silently consumed as a valid result downstream.
run() {
  local dir="$1"; shift
  rm -rf "$dir"
  if ! $PY scripts/sota_final.py run "$@" --score_dir "$dir"; then
    rm -rf "$dir"
    echo "!!!!!!!! FAILED run -> removed $dir (no partial/stale data served) !!!!!!!!"
  fi
}

for M in tyche universeg; do
  for K in 1 4 8 16; do
    echo "### $M K=$K $(date)"
    run results/scores_fgk/"${M}"_k"$K" --method "$M" --datasets dsb2018,monuseg \
      --metric_override fg_iou --seeds 6 --pool 20 --test 24 --support "$K" --cache "$CACHE"
  done
done
echo "### GPU1 ALL DONE $(date)"
