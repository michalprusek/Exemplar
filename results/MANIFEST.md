# Score records behind the paper

Every number, table and figure of the ISBI 2027 paper is regenerated from these files by `paper/make_paper.sh`
(byte-identical to the paper's files, version of 18 October 2026). The set is exactly what the paper's generators
open, determined by running them under an audit hook (`tools/sync_paper.py`), and nothing else: 676 files, 17.7 MB, in
61 directories (18 October 2026). Paths are those of the research repository, so the generators read them unchanged.
The records were copied with `tools/sync_paper.py`'s record scrub: fields that describe the machine a run happened on
(host, process ID, environment variables, interpreter and code paths, the GPU's process list) are removed, and
absolute paths in the other fields are shortened to `<path>/<last component>`. No generator reads these fields.

Record format (`final10/`, `nnunet_k1/`): one JSON file `<method>__<dataset>.json` per arm and dataset, with
`per_image` (the score of every test image, seed-major: 10 seeds x the full test split), `seeds`, `test_per_seed`,
`metric` (foreground IoU, or centreline Dice on DRIVE, HRF, ISBI2012-EM and FISBe), `split_fp` (a digest of the test
split, which the generators require to match across arms) and `protocol`. Exemplar's records (`ridge4_*`, method
`ridge`) also hold `alt`, the other readouts of the same fits per test image. `benchmark/run.py` writes records in this
format. The paper reports ten datasets. CTC-U373, evaluated in earlier versions, is not reported, because its
conditions require the permission of the Cell Tracking Challenge organisers for public use outside the challenge;
`paper/make_paper.sh` leaves it out (`ASG_DROP_DS=ctc_u373`) and its records are not shipped. Every arm below therefore
holds one record per dataset for these ten datasets (the two `cellpose` example arms excepted).

Abbreviations: T1 = Table 1 (`tab_fewshot.tex`), T2 = Table 2 (`tab_ablation.tex`), F1/F2/F3 = Figures 1-3,
stats = `numbers_stats.tex`, abl = `numbers_ablation.tex`, fitinf = `numbers_fitinf.tex`, timing = `numbers_timing.tex`.

## Exemplar (v4, the reported method; method `ridge`)

- `final10/ridge4_k8/` — K = 8. T1 Exemplar column; T2 "Exemplar" row and the two readout rows (`alt["plain_mse"]` = plain readout at 1/2, `alt["bal_mse"]` = balanced readout at 1/2); F2 at K = 8; stats.
- `final10/ridge4_k1/` — K = 1. F2; stats (one-mask comparisons).
- `final10/ridge4_k4/` — K = 4. F2.
- `final10/ridge4_k16/` — K = 16. F2.
- `final10/ridge4_nocls_k8/` — K = 8, `--blocks coarse,fine` (coarse + fine view, no priors). T2 "Coarse + fine view only"; abl; stats.
- `final10/ridge4_bankonly_k8/` — K = 8, `--blocks bank` (priors only). T2 "Priors only"; abl; stats.
- `final10/ridge4_coarseonly_k8/` — K = 8, `--blocks coarse,bank`. T2 "Coarse view + priors"; abl.
- `final10/ridge4_fineonly_k8/` — K = 8, `--blocks fine,bank`. T2 "Fine view + priors"; abl.
- `final10/ridge4_nolf12_k8/` — K = 8, `--last_layer_only` (feature string without `_lf12`). T2 "Last layer only"; abl.
- `final10/ridge4_nozoom_k8/` — K = 8, `--zoom_fixed 1`. T2 "Fine view not magnified"; abl.
- `c119/qual_v4/` — NOT shipped: the seed 0, K = 8 qualitative dumps (`--dump_qual`) hold test images of the datasets, which this repository does not redistribute. Make them with your own copy of the data (docs/REPRODUCE.md); `paper/make_paper.sh` then also draws Fig. 1's thumbnails and Fig. 3's panels, and skips those two steps without them.

## Fitted without gradient descent (Table 1) and forest controls (Table 2)

- `final10/ilastik142_k8/` — the ilastik software, version 1.4.2 (pixel classification: its default random forest of 100 trees on all of its filter features, trained on pixels sampled from the support masks), K = 8. T1 "ilastik" column.
- `final10/ilastik_k8/` — a random forest in the style of the ilastik workflow over Exemplar's 35 priors (method `ilastik_rf`, not the ilastik software), K = 8. T2 "Forest on the priors".
- `final10/views_rf3_k8/` — the same forest on Exemplar's coarse and fine views, no priors (method `views_rf`), K = 8. T2 "Forest on the views".
- `final10/convpaint_k8/` — Convpaint (a classifier fitted to pretrained features of the support pixels), K = 8. T1 column; stats.

## Forward-pass in-context methods

- `final10/seggpt_k8/` — SegGPT, K = 8. T1; stats; F2.
- `final10/seggpt_k1/` — SegGPT, K = 1. F2.
- `final10/seggpt_k4/` — SegGPT, K = 4. F2.
- `final10/seggpt_k16/` — SegGPT, K = 16. F2.
- `final10/universeg_k8/` — UniverSeg, K = 8. T1; stats; F2.
- `final10/universeg_k1/` — UniverSeg, K = 1. F2.
- `final10/universeg_k4/` — UniverSeg, K = 4. F2.
- `final10/universeg_k16/` — UniverSeg, K = 16. F2.
- `final10/insid3_k8/` — INSID3 (the authors' code, CRF on), K = 8. T1; stats; F2.
- `final10/insid3_k1/` — INSID3, K = 1. F2.
- `final10/insid3_k4/` — INSID3, K = 4. F2.
- `final10/insid3_k16/` — INSID3, K = 16. F2.
- `final10/tyche_bestof_k8/` — Tyche scored by its best of 16 candidates, K = 8. T1; stats; F2.
- `final10/tyche_bestof_k1/` — Tyche, K = 1. F2.
- `final10/tyche_bestof_k4/` — Tyche, K = 4. F2.
- `final10/tyche_bestof_k16/` — Tyche, K = 16. F2.

## Networks trained or fine-tuned on the support masks

Fig. 2 draws, per specialist and K, the better of two training lengths (the `_e<epochs>` arms).

- `final10/nnunet_k8/` — nnU-Net trained from scratch on the 8 support masks. T1; stats; F2.
- `nnunet_k1/` — nnU-Net on 1 mask (outside `final10/` in the research repository too). F2; stats.
- `final10/nnunet_k4/` — nnU-Net on 4 masks. F2.
- `final10/nnunet_k16/` — nnU-Net on 16 masks. F2.
- `final10/cellpose_ft11sem_k8/` — Cellpose-SAM fine-tuned on the 8 support masks (800 epochs, semantic read-out). T1; stats; F2.
- `final10/stardist_ft11sem_k8/` — StarDist fine-tuned on the 8 support masks (semantic read-out). T1; F2.
- `final10/microsam_ft11sem_k8/` — micro-SAM fine-tuned on the 8 support masks (semantic read-out). T1; F2.
- `final10/cellpose_ft11sem_k1_e800/` — Cellpose-SAM fine-tuned on 1 mask, 800 epochs. F2 (the better of the two lengths per K); stats.
- `final10/cellpose_ft11sem_k1_e6400/` — Cellpose-SAM fine-tuned on 1 mask, 6400 epochs. F2; stats.
- `final10/cellpose_ft11sem_k4_e800/` — Cellpose-SAM fine-tuned on 4 masks, 800 epochs. F2.
- `final10/cellpose_ft11sem_k4_e1600/` — Cellpose-SAM fine-tuned on 4 masks, 1600 epochs. F2.
- `final10/cellpose_ft11sem_k16_e400/` — Cellpose-SAM fine-tuned on 16 masks, 400 epochs. F2.
- `final10/cellpose_ft11sem_k16_e800/` — Cellpose-SAM fine-tuned on 16 masks, 800 epochs. F2.
- `final10/stardist_ft11sem_k1_e1600/` — StarDist fine-tuned on 1 mask, 1600 epochs. F2.
- `final10/stardist_ft11sem_k1_e12800/` — StarDist fine-tuned on 1 mask, 12800 epochs. F2.
- `final10/stardist_ft11sem_k4_e1600/` — StarDist fine-tuned on 4 masks, 1600 epochs. F2.
- `final10/stardist_ft11sem_k4_e3200/` — StarDist fine-tuned on 4 masks, 3200 epochs. F2.
- `final10/stardist_ft11sem_k16_e800/` — StarDist fine-tuned on 16 masks, 800 epochs. F2.
- `final10/stardist_ft11sem_k16_e1600/` — StarDist fine-tuned on 16 masks, 1600 epochs. F2.
- `final10/microsam_ft11sem_k1_e50/` — micro-SAM fine-tuned on 1 mask, 50 epochs. F2.
- `final10/microsam_ft11sem_k1_e400/` — micro-SAM fine-tuned on 1 mask, 400 epochs. F2.
- `final10/microsam_ft11sem_k4_e50/` — micro-SAM fine-tuned on 4 masks, 50 epochs. F2.
- `final10/microsam_ft11sem_k4_e100/` — micro-SAM fine-tuned on 4 masks, 100 epochs. F2.
- `final10/microsam_ft11sem_k16_e25/` — micro-SAM fine-tuned on 16 masks, 25 epochs. F2.
- `final10/microsam_ft11sem_k16_e50/` — micro-SAM fine-tuned on 16 masks, 50 epochs. F2.
- `final10/cellpose_sam/` — Cellpose-SAM off the shelf, MoNuSeg only (one file): read by `make_semantic_tables.py` for the Table 1 caption's off-the-shelf against fine-tuned example.
- `final10/cellpose_ft_k8/` — the July fine-tune of Cellpose-SAM on three datasets, MoNuSeg only (1 of 3 files): the other half of that example.

## Timing (one RTX A5000)

- `c119/timing_all_records/` — fit and inference time of every Table 1 method (the timing protocol of docs/BASELINES.md: seed-0 draw, K = 8, first five test images), one record per method and dataset on the ten datasets: SegGPT, UniverSeg, INSID3, Tyche, Exemplar, Convpaint, Cellpose-SAM, StarDist, micro-SAM, the ilastik software (`ilastik__*`, on the CPU of the same host, the card idle) and nnU-Net's measured 100-epoch runs (`check-nnU-Net-full-run__*`), plus the forest on the priors (`Prior-bank_RF__*`, which the generator reads but no table prints). fitinf (T1's last row, the prose's fit and refit ranges). nnU-Net's 5-epoch extrapolations (`nnU-Net__*`, replaced by the measured runs under `--nnunet_full`) and the raw `refit__*` / `predict__*` outputs are read by no generator and not shipped.
- `c119/` — `timing_ridge4_a5000.json`: Exemplar's fit and refit (the 8th mask added) on DSB2018, DRIVE, MoNuSeg and HRF. timing (`\fitSmall`, `\refitLarge`, ...).
- `timing_a5000.json` — nnU-Net's fit on DSB2018 and HRF and the coarse view's encoding time. timing (`\nnSmallMin`, `\costRatioLo`, `\encHi`, ...).
- `timing_cellpose_ft_a5000.json` — Cellpose-SAM's fine-tune times. timing (`\cpFitSmall`, `\cpRefitSmall`, ...).
- `c114/` — `timing_specialists_ft_a5000.json`: StarDist and micro-SAM fine-tune times. timing (`\sdFit*`, `\msFit*`).

## Not read by the paper

- **Cellpose-SAM fine-tuned for a limited time** (4, 16, 64 and 200 epochs) and its fit times: measured after this
  version; no table or figure of the paper reads them, so they are not shipped.
