#!/bin/bash
# GPU0 (A100): DINOv3-using methods (ours, insid3) + Matcher. Serial => safe shared feature cache.
cd /disk1/prusek/active-segmenter
export PYTHONPATH=/disk1/prusek/active-segmenter
export CUDA_VISIBLE_DEVICES=0
PY=~/dinov3_env/bin/python
CACHE=/disk1/prusek/asg_cache_fgk_dino
D5=spheroidj,dsb2018,monuseg,drive,hrf

# FAIL-LOUD run helper: clean the score_dir first (no stale data can survive), and on failure remove the
# partial dir + print a prominent marker so a failed run can NEVER be silently consumed as a valid result.
run() {
  local dir="$1"; shift
  rm -rf "$dir"
  if ! $PY scripts/sota_final.py run "$@" --score_dir "$dir"; then
    rm -rf "$dir"
    echo "!!!!!!!! FAILED run -> removed $dir (no partial/stale data served) !!!!!!!!"
  fi
}

for K in 1 4 8 16; do
  echo "### ours K=$K $(date)"
  run results/scores_fgk/ours_k"$K" --method head_fusion_best_cgate_film_nobank --datasets dsb2018,monuseg \
    --metric_override fg_iou --seeds 6 --pool 20 --test 24 --support "$K" --res 672 --cache "$CACHE"
done
for K in 1 4 8 16; do
  echo "### insid3 K=$K $(date)"
  run results/scores_fgk/insid3_k"$K" --method insid3 --datasets dsb2018,monuseg \
    --metric_override fg_iou --seeds 6 --pool 20 --test 24 --support "$K" --cache "$CACHE"
done
# Matcher must be scored on the SAME foreground metric as everything else in the K-scaling figure.
# Running all of $D5 in one call left dsb2018/monuseg on their native instance-AP, and
# make_final_kscale.py then DISCARDED those files (its metric filter only accepts the foreground
# metric) — which silently reduced Matcher to a single K=1 point rather than a curve. Split the run
# exactly like ours: override to fg-IoU on the instance datasets, native metric elsewhere
# (spheroidj -> fg_iou, drive/hrf -> cldice).
for K in 4 8 16; do
  echo "### matcher K=$K $(date)"
  run results/scores_fgk/matcher_k"$K"_fg --method matcher --datasets dsb2018,monuseg \
    --metric_override fg_iou --seeds 6 --pool 20 --test 24 --support "$K" --cache "$CACHE"
  run results/scores_fgk/matcher_k"$K" --method matcher --datasets spheroidj,drive,hrf \
    --seeds 6 --pool 20 --test 24 --support "$K" --cache "$CACHE"
done
echo "### GPU0 ALL DONE $(date)"
