#!/bin/bash
# Regenerate every generated file of the ISBI 2027 paper from the score records shipped in results/.
#
#   paper/make_paper.sh [out dir]          (default: paper/out)
#
# Writes into the output directory exactly the files the manuscript inputs:
#   tab_fewshot.tex        Table 1   (make_semantic_tables.py; its last row from numbers_fitinf.tex)
#   tab_ablation.tex       Table 2   (make_ablation_table.py)
#   numbers_stats.tex      paired Wilcoxon tests and win counts of the Results (make_stats.py)
#   numbers_ablation.tex   Table 2's numbers in the prose (make_ablation_table.py)
#   numbers_fitinf.tex     fit and inference time of every method, Table 1's last row (c119/timing_table.py)
#   numbers_timing.tex     fit / refit times and the cost ratios of the prose (make_timing_numbers.py)
#   fig_kscale.tex         Fig. 2, accuracy against the number of masks (make_final_kscale.py; also figures/kscale.pdf
#                          and figures/kscale.png)
#   figures/fig1_*.png     Fig. 1's two thumbnails (make_fig1_thumbs.py)
#   figures/qual3/*.png    Fig. 3's eight panels (render_qual_grid.py)
# and timing_all_a5000.json, the summary timing_table.py writes beside its macros.
#
# The paper reports ten datasets: CTC-U373 is left out of every table, statistic, median and figure (ASG_DROP_DS below),
# because its conditions require the permission of the Cell Tracking Challenge organisers for public use outside the
# challenge. Fig. 2 never used it (its pool of fifteen images cannot supply K=16).
#
# The generators are the paper's own (copied from the research repository; only their default paths are adapted, see
# tools/sync_paper.py) and keep their guards: each refuses an incomplete or inconsistent arm rather than writing a
# table with a hole in it. The configuration below is the paper's (the timing
# macros; the fit and inference times with --nnunet_full --ilastik; the tables with Convpaint, ilastik, the fit-time
# row and the forest rows; all without CTC-U373). Run from anywhere; needs numpy,
# scipy, matplotlib and Pillow. With the versions of the reference run (Python 3.12, numpy 2.5.3, scipy 1.18.1,
# matplotlib 3.11.2, Pillow 12.3.0) every file is byte-identical to the paper's, except the creation date matplotlib
# writes into figures/kscale.pdf (set SOURCE_DATE_EPOCH to fix it); tests/test_paper.py checks this against
# paper/expected_outputs.sha256. In another output directory than paper/out, the comment lines of numbers_timing.tex
# name the records by another relative path.
#
# Environment (all optional): PYTHON (interpreter, default python3), RUN (command prefix for the generators, default
# $PYTHON; tools/sync_paper.py runs them under an audit hook this way), EXEMPLAR_REPO (the root that holds results/,
# default this repository), EXEMPLAR_SCRIPTS (the generators' directory, default paper/).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${EXEMPLAR_REPO:-$HERE/..}" && pwd)"
S="$(cd "${EXEMPLAR_SCRIPTS:-$HERE}" && pwd)"
PYTHON="${PYTHON:-python3}"
RUN="${RUN:-$PYTHON}"
OUT="${1:-$REPO/paper/out}"
mkdir -p "$OUT/figures/qual3"
OUT="$(cd "$OUT" && pwd)"
cd "$REPO"
[ -d results/final10 ] || { echo "no results/final10 under $REPO" >&2; exit 2; }
# the records seen from the output directory (../../results for paper/out: numbers_timing.tex names its sources so)
REL="$("$PYTHON" -c 'import os, sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$REPO/results" "$OUT")"
say () { echo "== $*" >&2; }
export ASG_DROP_DS=ctc_u373          # the paper's ten datasets (see the header)

# 1. fit and inference time per method on the RTX A5000 (docs/BASELINES.md, "Timing"), medians over the ten datasets;
#    nnU-Net's column from its measured 100-epoch runs (--nnunet_full), ilastik's from its CPU runs (--ilastik)
say "numbers_fitinf.tex"
$RUN "$S/c119/timing_table.py" results/c119/timing_all_records "$OUT/timing_all_a5000.json" "$OUT/numbers_fitinf.tex" \
  --nnunet_full --ilastik

# 2. the prose's fit / refit times and cost ratios (STEP=timing; no Convpaint source, as in the paper)
say "numbers_timing.tex"
(cd "$OUT" && ASG_TIMING="$REL/timing_a5000.json" ASG_TIMING_OURS="$REL/c119/timing_ridge4_a5000.json" \
   ASG_TIMING_CP="$REL/timing_cellpose_ft_a5000.json" ASG_TIMING_SPEC="$REL/c114/timing_specialists_ft_a5000.json" \
   ASG_SEM_OUT=. $RUN "$S/make_timing_numbers.py" > /dev/null)

# 3. tables, statistics, figures (STEP=tables, FITINF=1 CONVPAINT=1 ILASTIK=1 FOREST=views,priors)
export ASG_SEM_TREE=results/final10 ASG_SEM_OUT="$OUT" ASG_OURS_STEM=ridge4 ASG_TYCHE_STEM=tyche_bestof \
       ASG_SEM_NO_MATCHER=1 ASG_SEM_NO_K1=1 ASG_SEM_NO_TYCHE_MARK=1 ASG_SEM_CONVPAINT=1 ASG_SEM_FIT_ROW=1 ASG_SEM_FITINF=1 \
       ASG_SEM_ILASTIK=1
say "tab_fewshot.tex"
$RUN "$S/make_semantic_tables.py"
say "numbers_stats.tex"
$RUN "$S/make_stats.py"
say "tab_ablation.tex, numbers_ablation.tex"
ASG_ABL_SET=ridge4 ASG_ABL_FOREST=views,priors $RUN "$S/make_ablation_table.py"
say "fig_kscale.tex, figures/kscale.pdf"
ASG_KSCALE_HEIGHT=66 ASG_RESULTS_ROOT="$REPO/results" ASG_REPO_ROOT="$REPO" ASG_KSCALE_OUT="$OUT/figures/kscale.pdf" \
  $RUN "$S/make_final_kscale.py"
# Fig. 1's thumbnails and Fig. 3's panels are drawn from test images of the datasets, which this repository does not
# redistribute. Make the dumps with your own copy of the data (docs/REPRODUCE.md, section 1, "Figure 3") into
# results/c119/qual_v4/, or point QUAL_DIR at them; without them these two steps are skipped.
QUAL_DIR=${QUAL_DIR:-results/c119/qual_v4}
if [ -f "$QUAL_DIR/rozpad_0.npz" ]; then
  say "figures/fig1_*.png"
  $RUN "$S/make_fig1_thumbs.py" "$QUAL_DIR/rozpad_0.npz" "$OUT/figures" 715,633,700
  say "figures/qual3/*.png"
  $RUN "$S/render_qual_grid.py" "$QUAL_DIR" "$OUT/figures/qual3"
else
  say "figures/fig1_*.png, figures/qual3/*.png: skipped (no qualitative dumps in $QUAL_DIR; they hold dataset images)"
fi
say "done: $OUT"
