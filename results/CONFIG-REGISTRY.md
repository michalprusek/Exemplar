# CONFIG REGISTRY — retrievable index of every tested experiment config

Purpose: a **config-lookup index** so any run is reproducible later. Each entry pins the exact
setup (method + all flags, K/pool/test/seeds, res, datasets, cache dir, score_dir, launcher, result
numbers, date). This is NOT the narrative — the story, dead-ends, and verdicts live in
`results/EXPERIMENT-LOG.md`; the polished conclusions in `results/*-FINDINGS.md`. Use this file to
answer "what was the exact config that produced number X, and how do I re-run it?"

Numbers here are cross-checked against `EXPERIMENT-LOG.md` and the matching `*-FINDINGS.md`; any
disagreement is flagged in the entry rather than silently reconciled.

## Shared protocol (applies to EVERY entry unless overridden)

- **Multi-draw fixed-pool**: `load_dataset(spec, pool, test, seed=0)` loaded ONCE; K subsampled per
  seed. Never `load_dataset(spec, 8, 24, seed=seed)` (shifts the test slice on download datasets).
- **Stats**: paired per-image Wilcoxon. On small datasets (n≈12: hrf/microtubules) trust
  reproduced-across-runs effects, not single paired p-values (per-run cudnn nondeterminism shifts
  all test images together, ±0.01).
- **Encoder**: frozen DINOv3-ViT-L, `--res 672`.
- **Env**: `HF_HOME=/disk1/prusek/.cache/huggingface`; Python env `~/dinov3_env`.
- **Compute**: remote **tulen** — A100 80GB = GPU0, RTX A5000 24GB = GPU1.
- **Repo on tulen**: `/disk1/prusek/active-segmenter`. Launcher `.sh` scripts + one-off eval scripts
  named below live there (not mirrored to the local mac, which has no GPU env).
- **Reproduce (all entries)**: ssh tulen → `cd /disk1/prusek/active-segmenter` → run the named
  launcher; caches/score_dirs are as listed. Never let concurrent processes share a WRITABLE cache.

---

## Session 2026-07-16 — encoder/upsampler, colour bank, instance decoder, refine, bank-unfreeze

### C1 — Feature-upsampler A/B (superres vs AnyUp)

| field | value |
|---|---|
| method | `head_fusion_adaptive` (fixed across all 4 arms) |
| arms / flags | (a) `--superres 1` [baseline] · (b) `--superres 2` [sr2] · (c) `--feat_upsampler anyup --upsampler_factor 2` [anyup2] · (d) `--feat_upsampler anyup --upsampler_factor 4` [anyup4] |
| K (support) | 8 |
| pool / test | 20 / 24 |
| seeds | 6 |
| res | 672 |
| datasets | microtubules, drive, hrf |
| cache | `/disk1/prusek/asg_cache_ab` |
| score_dir | `results/scores_ab/{baseline,sr2,anyup2,anyup4}` |
| launcher | `run_ab_upsampler.sh` (tulen); comparator `ab_upsampler_stats.py` |
| notes | AnyUp `wimmerth/anyup` pinned commit `351807a`, `use_natten=False`, RAW-feature input; arms vary ONLY coarse-grid densification, all UNGATED. Code `active_segmenter/encoder/feat_upsample.py`. |
| date | 2026-07-16 |

Results — clDice:

| arm | microtubules | drive | hrf |
|---|---|---|---|
| baseline (`--superres 1`) | 0.760 | 0.643 | 0.599 |
| sr2 (`--superres 2`) | 0.801 | 0.673 | 0.603 |
| anyup2 (`anyup ×2`) | 0.766 | 0.643 | 0.597 |
| anyup4 (`anyup ×4`) | 0.818 | 0.647 | 0.597 |

Key deltas: anyup4−sr2 microtubules **+0.017 (p=0.002)**; anyup4−sr2 drive −0.026; anyup4 vs hrf
≈ tie. Verdict: don't swap the encoder; real shift-merge superres is the robust general lever;
AnyUp ×4 = optional targeted lever for finest near-patch filaments only.
Cross-check: matches `EXPERIMENT-LOG.md` (2026-07-16 section) and `ENCODER-UPSAMPLER-FINDINGS.md`
(which also carries ±std: baseline 0.760±0.113 / sr2 0.801±0.094 / anyup4 0.818±0.080). No discrepancy.

### C2 — Colour/stain HyperBank channel A/B

| field | value |
|---|---|
| method | `head_fusion_adaptive` (grayscale bank) vs `head_fusion_adaptive_color` (support-selected colour/stain channel) |
| flags | method alias only (colour channel self-selected at fit) |
| K (support) | 8 |
| pool / test | 20 / 24 |
| seeds | 6 |
| res | 672 |
| datasets | monuseg, drive, hrf |
| cache | `/disk1/prusek/asg_cache_ab` |
| score_dir | `results/scores_color/{adaptive,color}` |
| launcher | `run_color_ab.sh` (tulen) |
| notes | Channel picks (per-dataset, support-derived): drive→green, hrf→eosin, monuseg→R. Paired p<0.001 all 3. |
| date | 2026-07-16 |

Results:

| dataset | metric | adaptive (gray) | color | Δ |
|---|---|---|---|---|
| monuseg | instance-AP | 0.135 | 0.156 | +0.021 |
| drive | clDice | 0.644 | 0.671 | +0.026 |
| hrf | clDice | 0.598 | 0.650 | +0.052 |

Verdict: colour wins all 3; keep. Cross-check: matches `EXPERIMENT-LOG.md` (2026-07-16 colour entry).
No dedicated `COLOR-FINDINGS.md` file on disk (narrative is in EXPERIMENT-LOG). No discrepancy.

### C3 — Instance-decoder A/B (blob-marker vs affinity-watershed, both SAM-free)

| field | value |
|---|---|
| method | `head_fusion_adaptive` (`instance_mode=blob`) vs `head_fusion_adaptive_affinity` (`instance_mode=affinity`, SAM-free) |
| flags | `--refine none` |
| K (support) | 8 |
| pool / test | 20 / 24 |
| seeds | 6 |
| res | 672 |
| datasets | dsb2018, monuseg, ctc_u373 |
| cache | `/disk1/prusek/asg_cache_inst` |
| score_dir | `results/scores_inst/{blob,affinity}` |
| launcher | `run_inst_ab.sh` (tulen) |
| notes | Affinity = DT markers + Frangi-ridge watershed + DINO-affinity merge-veto; r* + merge-cos-floor calibrated from K support (no learned dense field). Paired p<0.001 all 3. |
| date | 2026-07-16 |

Results — instance-AP:

| dataset | blob | affinity | Δ |
|---|---|---|---|
| dsb2018 | 0.450 | 0.478 | +0.028 |
| monuseg | 0.134 | 0.171 | +0.037 |
| ctc_u373 | 0.097 | 0.430 | +0.334 (4.4×) |

Verdict: affinity wins all 3; SAM-free (the goal). Cross-check: matches `EXPERIMENT-LOG.md`
(2026-07-16 affinity entry) and `INSTANCE-DECODER-FINDINGS.md`. No discrepancy.

### C4 — Amodal-SAM refine arm (3-way instance comparison)

| field | value |
|---|---|
| method | `head_fusion_adaptive` `--refine amodal` |
| flags | `--refine amodal` |
| K (support) | 8 |
| pool / test | 20 / 24 |
| seeds | 6 |
| res | 672 |
| datasets | dsb2018, ctc_u373 (GPU0) + monuseg (GPU1, 2h timeout cap) |
| cache | `asg_cache_am0` (GPU0) / `asg_cache_am1` (GPU1) — under `/disk1/prusek/` |
| score_dir | `results/scores_amodal` |
| launcher | `run_amodal.sh` (tulen) |
| notes | Compared 3-way vs the two SAM-free arms of C3 (blob / affinity). |
| date | 2026-07-16 |

Results — instance-AP (amodal-SAM arm; blob/affinity from C3 for context):

| dataset | blob | affinity | amodal-SAM | status |
|---|---|---|---|---|
| dsb2018 | 0.450 | 0.478 | 0.530 | complete |
| ctc_u373 | 0.097 | 0.430 | 0.186 | complete (SAM per-nucleus prompting FAILS on large phase-contrast cells) |
| monuseg | 0.134 | 0.171 | — | **INCOMPLETE / pending** — 432 nuclei/img, likely hit the 2h cap |

Verdict: SAM leads on dsb's small clustered nuclei but the SAM-free affinity decoder BEATS SAM 2.3×
on ctc_u373's large cells → strengthens the "drop SAM" argument. Cross-check: matches
`EXPERIMENT-LOG.md` (2026-07-16 amodal-SAM 3-way entry). No discrepancy.

### C5 — Bank-unfreeze A/B (frozen vs trainable classical bank) at K=1 and K=8

| field | value |
|---|---|
| method | `head_fusion_adaptive` (frozen bank) vs `head_fusion_adaptive_tc` (trainable classical bank) |
| flags | `--support 1` AND `--support 8` (two separate K values) |
| K (support) | 1 and 8 |
| pool / test | 20 / 24 |
| seeds | 6 |
| res | 672 |
| datasets | spheroidj, microtubules, drive |
| cache | `asg_cache_g0` (frozen) / `asg_cache_g1` (tc) — under `/disk1/prusek/` |
| score_dir | `results/scores_bankfreeze/{frozen_k1,frozen_k8,tc_k1,tc_k8}` |
| launcher | `run_bankfreeze.sh` (tulen; dual-GPU parallel) |
| notes | Morphology-gated lever: helps thin filaments (adapt Frangi sensitivity), neutral-to-negative on blobs/vessels. |
| date | 2026-07-16 |

Results:

| dataset | metric | frozen K1 | tc K1 | frozen K8 | tc K8 |
|---|---|---|---|---|---|
| microtubules | clDice | 0.435 | 0.457 (+0.022) | 0.774 | 0.805 (+0.031) |
| spheroidj | fg-IoU | 0.823 | 0.819 | 0.884 | 0.883 |
| drive | clDice | 0.629 | 0.623 (−0.006) | 0.643 | 0.640 (−0.003) |

Verdict: unfreezing the ~dozen bank params helps THIN FILAMENTS (scales with K, not overfitting),
tie on blobs, slightly negative on vessels → gate on tubularity. Cross-check: matches
`EXPERIMENT-LOG.md` (2026-07-16 bank-unfreeze entry) — the LOG records the microtubules/spheroidj
absolutes and the drive DELTAS (−0.006 K1, −0.003 K8). The drive ABSOLUTES (0.629/0.643 frozen,
0.623/0.640 tc) are from this session's config data; their deltas are consistent with the LOG. No
discrepancy.

### C6 — Oracle-FG diagnostic (affinity decoder on GT foreground)

| field | value |
|---|---|
| method | affinity instance decoder run on GT foreground (calibrate-only, no head fit) |
| flags | oracle foreground (GT fg), calibrate-only |
| K (support) | 8 |
| pool / test | (per shared protocol) |
| seeds | 6 |
| res | 672 |
| datasets | dsb2018, ctc_u373 (GPU0) + monuseg (GPU1) |
| cache | `asg_cache_inst` (READ-ONLY) — under `/disk1/prusek/` |
| script | `oracle_fg_diag.py` (tulen) |
| notes | Diagnostic upper-bound: isolates the instance-separation stage from foreground error. |
| date | 2026-07-16 |

Results:

| dataset | ORACLE-fg AP (perfect fg) | real-fg AP (ours) | specialist |
|---|---|---|---|
| dsb2018 | 0.782±0.164 | 0.478 | ~0.64 |
| ctc_u373 | 0.884±0.129 | 0.430 | ~0.65–0.73 |
| monuseg | 0.886±0.049 | 0.171 | ~0.39 |

VERDICT: the instance gap to specialists is FOREGROUND-limited, NOT separation-limited — with perfect fg
the affinity decoder BEATS specialists on all 3. All roadmap energy → foreground. (Script bug fixed:
`instance_ap` wants raw `.mask`, not `InstanceMask`.)

---

### C7 — Superres fg lever on the affinity decoder (roadmap #2 quick test)

| field | value |
|---|---|
| method | `head_fusion_adaptive_affinity --superres 2` vs the affinity arm from C3 (no superres) |
| K / pool / test / seeds / res | 8 / 20 / 24 / 6 / 672 |
| datasets | dsb2018, ctc_u373 (GPU0) + monuseg (GPU1) |
| cache | `/disk1/prusek/asg_cache_fgsr0` / `asg_cache_fgsr1` |
| score_dir | `results/scores_fgsr` |
| launcher | `run_fgsr.sh` (tulen) |
| notes | tests whether superres sharpens fg → instance-AP (fg is the bottleneck per C6). CAVEAT: ran while other jobs shared the GPUs (separate caches → results valid, just slow). |
| date | 2026-07-16 |

Results (instance-AP, vs C3 affinity baseline): ctc_u373 0.430→**0.488** (+0.058); dsb2018 0.478→0.471
(−0.007); monuseg 0.171→0.166 (−0.005). → superres is a MORPHOLOGY-MIXED fg lever: helps LARGE/sparse cells
(ctc), mild hurt on DENSE/small nuclei (dsb, monuseg). Fold in behind a size/density gate (ON for ctc-like),
NOT universal. Consistent with C6 (fg is the bottleneck but no single lever fixes all morphologies).

---

### C8 — Universal DINO layer-set A/B (roadmap #7)

| field | value |
|---|---|
| method | `head_fusion_adaptive`; arms `--layers ""` [single/-1 baseline] vs `--layers "-1,12"` [last+mid] vs `--layers "-1,12,6"` [last+mid+early] |
| K / pool / test / seeds / res | 8 / 20 / 24 / 6 / 672 |
| datasets | spheroidj, dsb2018, microtubules (GPU0) + hrf, monuseg, ctc_u373 (GPU1) |
| cache | `/disk1/prusek/asg_cache_lf0` / `asg_cache_lf1` |
| score_dir | `results/scores_lf/{single,lf12,lf1226}` |
| launcher | `run_layerfusion.sh` (tulen); comparator `lf_stats.py` |
| notes | LAYER-FUSION: concat blocks, L2 per-layer then unit, head learns weighting. FIX applied pre-launch (review): fusion `-1` maps to POST-norm `last_hidden_state` (parity with the single-layer baseline; `hidden_states[-1]` is pre-LayerNorm). Finds the best UNIVERSAL layer set. |
| date | 2026-07-16 |

Results: RUNNING — single baseline sits (spheroidj 0.884, dsb 0.449, microtubules 0.798); fusion arms
pending. Update when `LF_DONE`.

---

### C9 — head_fusion_best fold-in A/B (unified: color ⊕ affinity ⊕ tubularity-gated bank-unfreeze)

| field | value |
|---|---|
| method | `head_fusion_best` vs `head_fusion_adaptive` (baseline) |
| K / support / pool / test / seeds / res | 8 / 8 / 20 / 24 / 6 / 672 |
| datasets | ALL 7: spheroidj, dsb2018, monuseg, ctc_u373, drive, hrf, microtubules |
| cache | `/disk1/prusek/asg_cache_best0` (instance, GPU0) / `asg_cache_best1` (semantic, GPU1) |
| score_dir | `results/scores_best/{best,adaptive}` |
| launcher | `reorchestrate.sh` (tulen) — 5 parallel lanes, 1 dataset/lane (distinct cache keys → race-free); re-run: `CUDA_VISIBLE_DEVICES=<g> python scripts/sota_final.py run --method head_fusion_best --datasets <d> --seeds 6 --pool 20 --test 24 --support 8 --res 672 --cache <c> --score_dir results/scores_best/best` |
| notes | fold-in of C2(color)+C3(affinity)+C5(bank-unfreeze). TWO bug fixes made this run valid (see EXPERIMENT-LOG 2026-07-16): (1) affinity `_calibrate_instance` RAISE→SKIP on binary/semantic masks; (2) `_tubularity` bbox-crop (monuseg 27min stall → 0.78s). Both independently reviewed. |
| date | 2026-07-16 |

Results (COMPLETE 7/7, mean per_image, paired per-image Wilcoxon best vs adaptive):

| dataset | metric | adaptive | best | Δ | wilcox_p |
|---|---|---|---|---|---|
| spheroidj | fg_iou | 0.884 | 0.884 | +0.000 | 0.573 (tie) |
| dsb2018 | AP | 0.449 | 0.478 | +0.029 | <1e-4 |
| monuseg | AP | 0.135 | 0.198 | +0.063 | <1e-4 |
| ctc_u373 | AP | 0.096 | 0.427 | **+0.330** | <1e-4 |
| drive | clDice | 0.647 | 0.668 | +0.021 | <1e-4 |
| hrf | clDice | 0.616 | 0.652 | +0.036 | <1e-4 |
| microtubules | clDice | 0.793 | 0.773 | **−0.021** | 2e-4 |

VERDICT: fold-in wins 5/7 significantly (ctc +0.330 = affinity decoder on large phase-contrast cells; monuseg
+0.063; hrf +0.036; dsb +0.029; drive +0.021), neutral spheroidj, ONE significant REGRESSION microtubules
−0.021 (thin filaments; a lever interacts badly in combination — the factorial's leave-one-out will find the
culprit). cgate (C10) RECOVERS microtubules to 0.823. Caveat: microtubules is small (per-run cudnn shifts all its
images together per CLAUDE.md) so treat the −0.021 as directional; reproduce + factorial to confirm.

---

### C10 — Competitive-gate fast-screen (cgate vs best)

| field | value |
|---|---|
| method | `head_fusion_best_cgate` (softmax group-gate, zero-init parity) vs `head_fusion_best` (C9) |
| K / support / pool / test / seeds / res | 8 / 8 / 20 / 24 / 6 / 672 |
| datasets (SCREEN set) | TARGET: dsb2018 (instance), microtubules (thin) · CONTROL: spheroidj (blob), monuseg (dense-instance) |
| cache | `asg_cache_best0` / `asg_cache_best1` (shared with C9, distinct keys) |
| score_dir | `results/scores_best/cgate` |
| launcher | `reorchestrate.sh` (tulen); re-run: same as C9 with `--method head_fusion_best_cgate --score_dir results/scores_best/cgate` |
| notes | FAST-SCREEN (per protocol): GO iff ≥1 target Δ>+0.01 AND no control Δ<−0.005 AND no crash → then full panel. Competitive gate is zero-init → parity at init, so any delta is a real trained effect. |
| date | 2026-07-16 |

Results (COMPLETE 4/4, paired vs C9 best, same cache): TARGETS ↑ — dsb2018 AP 0.478→**0.514** (+0.036),
microtubules clDice 0.773→**0.823** (+0.050, recovers best's −0.020 regression vs adaptive). CONTROLS not hurt —
spheroidj fg_iou 0.884→0.884 (~0), monuseg AP 0.198→**0.218** (+0.020, control improved too). **VERDICT: GO**
(both targets Δ>+0.01, no control Δ<−0.005). cgate enters the factorial as a strong lever + fixes the
microtubules fold-in regression. Full 7-dataset panel to run in the factorial phase.

---

### C11 — Phase-2 lever fast-screens (FiLM / corr / bank vs best)

| field | value |
|---|---|
| method | `head_fusion_best_{film,corr,bank}` vs `head_fusion_best` (C9) |
| screen datasets | dsb2018, monuseg, microtubules, spheroidj (6 seeds, paired) |
| cache | `asg_cache_best0` (read-only, features prebuilt) |
| score_dir | `results/scores_p2/{film,corr,bank}` |
| launcher | `phase2_screen.sh` (tulen) |
| notes | Levers built + independently reviewed 2026-07-16 (all clean). DEPLOYMENT bug caught here: a dir-target rsync flattened paths → tulen ran stale code → film "unknown backend"; fixed by explicit-path resync + on-target alias smoke-test (now standard). |
| date | 2026-07-17 |

Results (Δ vs best, paired Wilcoxon p):
- **FiLM: GO (strong)** — dsb +0.019 (p.002), monuseg +0.005 (.060), microtubules **+0.043** (p<1e-3),
  spheroidj +0.008 (p<1e-3). No regression → strong GO; also fixes best's microtubules regression.
- **corr: GO (marginal)** — dsb +0.002, monuseg +0.001, microtubules +0.015 (.206), spheroidj −0.001 (.004).
  Tiny gains, no real regression → weak GO (fold-in already has strong fg via affinity+color).
- **bank: NO-GO / DROP** — dsb **−0.016** (p.039), monuseg +0.008, microtubules +0.017, spheroidj +0.010.
  Regresses dsb → DROP (the recorded-negative "more classical features hurt some morphologies" pattern).

---

### C12 — FACTORIAL 2^3 over {cgate, film, corr} → head_fusion_best_v2

| field | value |
|---|---|
| method | 8 configs: `head_fusion_best[_cgate][_film][_corr]` (composable lever suffixes; bank dropped at C11) |
| K/pool/test/seeds/res | 8 / 20 / 24 / 6 / 672; datasets = full 7-panel |
| cache | `asg_cache_best0` (hrf,drive prebuilt in → all 7 read-only, no race) |
| score_dir | `results/scores_fact/<config>` |
| launcher | `factorial.sh` (tulen); reuses best/cgate/film/corr cells, runs the 4 combos + single-lever gaps; re-run a cell: `python scripts/sota_final.py run --method head_fusion_best_cgate_film --datasets <d> --seeds 6 --pool 20 --test 24 --support 8 --res 672 --cache asg_cache_best0 --score_dir results/scores_fact/cgate_film` |
| notes | al_testbed token-parser (name→lever flags) added + on-target smoke-tested. Goal: find the best lever subset accounting for interactions (cgate & film each fix the microtubules fold-in regression — do they compound?). Winner = `head_fusion_best_v2`. |
| date | 2026-07-17 |

Results (FACTORIAL_DONE 06:28; hrf for the 3 cgate-combos re-running after an over-packed-GPU OOM — 6/7 each,
finalizing). meanΔ = mean per-dataset gain vs `best` (22:00 fold-in); #reg = datasets regressing >0.005:

| config | meanΔ vs best | #reg | note |
|---|---|---|---|
| best (22:00) | +0.000 | 0 | baseline |
| cgate | +0.019 | 1 (drive) | |
| film | +0.010 | 1 (drive −0.035) | |
| corr | +0.000 | 1 | no gain |
| **cgate_film** | **+0.021** | **0** | ✅ **WINNER = head_fusion_best_v2** |
| cgate_corr | +0.015 | 1 | |
| film_corr | +0.009 | 1 | |
| cgate_film_corr | +0.021 | 0 | = cgate_film (corr adds nothing) |

**VERDICT: `head_fusion_best_v2 = head_fusion_best_cgate_film`.** cgate + FiLM COMPOUND: highest mean gain
(+0.021 vs 22:00-best) AND remove ALL regressions (each lever alone still regressed drive; together drive→+0.009).
Per-dataset vs best (6/7, hrf finalizing): dsb +0.016, monuseg +0.026, ctc +0.017, drive +0.009, microtubules
**+0.062** (fixes the fold-in regression), spheroidj ~0. corr redundant on top → dropped; bank dropped at C11.
Lesson logged: A100 @ res672+superres uses ~14–17 GB/proc → cap ~4 concurrent lanes (5 OOM'd hrf). Next:
promote to CLAUDE.md best-so-far once hrf confirms 0-reg; then K-scaling K=1,4,8,16 best_v2 vs 22:00-best.

---

### C13 — DROP bank-unfreeze (latency + quality A/B) → new best_v2 = head_fusion_best_cgate_film_nobank

| field | value |
|---|---|
| method | `head_fusion_best_cgate_film_nobank` (bank-unfreeze OFF) vs `head_fusion_best_cgate_film` (C12 best_v2, bank-unfreeze ON) |
| K/pool/test/seeds/res | 8 / 20 / 24 / 6 / 672; datasets = thin/tubular (where bank-unfreeze fires): drive, hrf, microtubules |
| cache | `asg_cache_best0`; score_dir `results/scores_nobank` |
| launcher | `python scripts/sota_final.py run --method head_fusion_best_cgate_film_nobank --datasets drive,microtubules,hrf --support 8 --pool 20 --test 24 --seeds 6 --res 672 --cache asg_cache_best0 --score_dir results/scores_nobank` |
| notes | al_testbed `nobank` token added (disables `bank_unfreeze_adaptive`). Motivated by a LATENCY pathology: with bank-unfreeze the training loop recomputes the FULL native-res classical bank (`_classical(image, grad=True)`) every iteration → hrf cells hung >1–2.6h (blocked the K-scaling; deployment-killer for the interactive tool). |
| date | 2026-07-17 |

Results (nobank vs with-bank, clDice @K8): drive 0.677 → **0.696** (+0.019, bank-unfreeze HURT), microtubules
0.835 → 0.828 (−0.007, noise), hrf ∞(hung) → **0.679** (now runs in ~7 min). **VERDICT: DROP bank-unfreeze** —
net-negative-to-neutral on quality AND a catastrophic latency liability; the C5 "microtubules +0.031" was not
reproducible. **New best_v2 = `head_fusion_best_cgate_film_nobank`** (fast, deployable). Strong ablation result
for the paper: a lever rigorously pruned for not earning its cost. K-curve being re-filled with nobank on the
thin datasets (`results/scores_kscale/k*_v2nb`).

---

## Session 2026-07-17 — external baseline integration: Tyche (few-shot in-context)

### B1 — Tyche baseline K-scaling (Rakic et al., CVPR'24, MIT CSAIL)

| field | value |
|---|---|
| method | `tyche` (in-harness backend `active_segmenter/segment/tyche_backend.py`; wired in `al_testbed.make_backend`) |
| K (support) | 1, 4, 8, 16 |
| pool / test | 20 / 24 |
| seeds | 6 |
| res | 672 (harness res; Tyche itself is fixed 128×128 grayscale I/O — resize in/out, like our UniverSeg backend) |
| datasets | spheroidj, dsb2018, monuseg, ctc_u373, drive, hrf, microtubules |
| cache | `/disk1/prusek/asg_cache_tyche` (GPU1, SEPARATE writable cache → no race vs the GPU0 INSID3/UniverSeg K-scaling) |
| score_dir | `results/scores_basekscale/k{K}_tyche` |
| launcher | `tyche_kscale.sh` (tulen, GPU1 = `CUDA_VISIBLE_DEVICES=1`); needs `TYCHE_SRC=/disk1/prusek/tyche_src` |
| weights | pretrained CVPR `tyche_v1_model_weights_CVPR.pt` (auto-download via torch.hub → `TORCH_HOME=/disk1/prusek/.cache/torch`); 1.77 M params |
| notes | Stochastic set-predictor. FAIRNESS: single output mask = MEAN over `n_pred=16` candidate prob maps (marginal/expected), threshold 0.5; **NO GT used to pick candidates** (no oracle). Model built once/`__init__` → uniform noise across seeds. Semantic-only → CC instances on instance-AP datasets (same handicap as UniverSeg). Runs in `~/dinov3_env` (torch 2.5.1 / pydantic 2.13.4) despite its torch==1.13.1 pin — verified. Setup detail in `results/BASELINE-SETUP.md`. |
| date | 2026-07-17 |

Results (mean per_image ± std-over-seeds; each dataset's designated metric). K=16 skips ctc_u373/microtubules
(datasets too small: pool+test > available images — expected dataset-size limit):

| dataset | metric | K=1 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|
| spheroidj | fg_iou | 0.534±0.098 | 0.718±0.049 | 0.788±0.031 | 0.823±0.004 |
| dsb2018 | ap | 0.116±0.028 | 0.230±0.019 | 0.254±0.034 | 0.260±0.016 |
| monuseg | ap | 0.009±0.001 | 0.012±0.001 | 0.012±0.000 | 0.013±0.000 |
| ctc_u373 | ap | 0.002±0.001 | 0.020±0.010 | 0.023±0.004 | skip |
| drive | cldice | 0.365±0.023 | 0.463±0.006 | 0.481±0.003 | 0.493±0.003 |
| hrf | cldice | 0.109±0.022 | 0.173±0.024 | 0.187±0.015 | 0.189±0.003 |
| microtubules | cldice | 0.047±0.104 | 0.360±0.111 | 0.407±0.001 | skip |

Run complete (ALL_DONE 17:05, 2026-07-17). Clean monotonic K-scaling on every dataset. K=16 skipped
ctc_u373 + microtubules (`2/7 FAILED` = expected dataset-size limit: pool+test > available images); the other
5 datasets' K=16 JSONs were still written before the run exited non-zero. Sanity cross-check (K=1, same
panel/protocol) — Tyche ≥ UniverSeg on 4/5 shared datasets, consistent with Tyche being the improved
same-lab successor: spheroidj 0.534 vs 0.384, dsb2018 ap 0.116 vs 0.024, drive clDice 0.365 vs 0.181,
monuseg ap 0.009 vs 0.001, ctc_u373 0.002 vs 0.005. Semantic-only + CC keeps the instance-AP numbers
(monuseg/ctc_u373) very low — its honest handicap on touching objects, same as UniverSeg; the fg/vessel
metrics (spheroidj/drive/hrf/microtubules) are where it is competitive.

---

### C14 — Adaptive-resolution lever (self-config res from support object scale)

| field | value |
|---|---|
| method | `head_fusion_best_cgate_film_nobank --adaptive_res 200` vs fixed `--res 672` |
| K/pool/test/seeds/res | 8 / 20 / 24 / 6 / base 672, res_max 1536 |
| datasets | 7-panel; score_dir `results/scores_adres`; cache `asg_cache_adres` |
| launcher | `sota_final.py run --method head_fusion_best_cgate_film_nobank --adaptive_res 200 ...` |
| notes | Rule (sota_final `adaptive_res_decision`): signal = median support fg_frac (NOT CC-count); eff_obj_px=sqrt(median_fg)*base_res; fires when eff_obj_px<min_obj_px AND S>base_res (downscale headroom); target=clip(round16(min_obj/sqrt(fg)),base,min(res_max,native)); forces superres=1 when raised. Support-only decision (no test peek). MOTIVATION: res-confound showed PerSeg candle 4%@672→92%@1024, mIoU 84.9→90.2. |
| date | 2026-07-17 |

Results (adaptive_res vs fixed-672): NO REGRESSION on biomed panel (spheroidj 0.883→0.884 @res976; dsb/monuseg/
drive/ctc ±0.001; hrf/microtubules pending). Fires on small-object-in-large-image (spheroidj fg0.042→976, hrf
fg0.076/S3504→720, ctc→704), keeps 672 on normal-scale (dsb fg0.15, monuseg fg0.22, drive fg0.087). VERDICT: GO —
safe (no biomed regression) + recovers PerSeg small-object catastrophes automatically (end-to-end by composition:
rule→~1104, res-confound→1024 gives candle 92%). Robustness lever for the Nature Methods tool + a self-configuring
novelty layer. Minor over-fire on spheroidj (quality no-op, extra compute) — threshold tunable but not required.

---

### C15 — Multi-prototype correspondence (`mproto`) TRAINING-FREE PRE-SCREEN → NO-GO / DROP

| field | value |
|---|---|
| method | `mproto` lever: replace the single mean fg/bg corr prototype with k-means centroids + max-pooled cosine `max_k cos(feat,fg_k) − max_j cos(feat,bg_j)` (composable token `head_fusion_best_cgate_film_nobank_mproto`, n_proto=4). Motivated by C6 fg bottleneck + research (ASGNet/AENet/SPROUT) + in-code audit (corr dropped on panel-mean, never isolated on monuseg). |
| screen | TRAINING-FREE pre-screen (`scripts/mproto_prescreen.py`, no GPU training): per-support LOO fg/bg separability (AUROC + best-thr fg-IoU at grid res) of single vs kmeans vs kNN corr; TARGET monuseg, CONTROL spheroidj |
| K/res/seeds | 8 / 672 / seeds {0,1}, k-sweep {2,4,8}; cache `/disk1/prusek/asg_cache_mproto_prescreen` (tulen A100) |
| launcher | `PYTHONPATH=. python scripts/mproto_prescreen.py --k <k> --seed <s>`; log `/disk1/prusek/mproto_prescreen*.log` |
| date | 2026-07-18 |

Results (monuseg AUROC, single vs kmeans vs kNN):

| config | single | kmeans | kNN | verdict |
|---|---|---|---|---|
| k=2 seed0 | **0.844** | 0.832 | 0.794 | NO-GO |
| k=4 seed0 | **0.844** | 0.813 | 0.794 | NO-GO |
| k=8 seed0 | **0.844** | 0.774 | 0.794 | NO-GO |
| k=4 seed1 | **0.879** | 0.862 | 0.814 | NO-GO |

VERDICT: **NO-GO → DROP.** On monuseg the SINGLE prototype beats every multi-prototype variant, monotonically
(more centroids → worse: k2>k4>k8), robust across both seeds; spheroidj control unaffected (≈0.999). Refutes the
multi-prototype hypothesis (ASGNet/AENet/SPROUT did NOT transfer to monuseg's grid-scale corr channel — dense
nuclei ≈patch-scale → fg patches are noisy/mixed; k-means splits the noise and max-pool optimistically matches bg
to a noisy centroid → MORE false positives). **Reframe:** coarse fg/bg is ALREADY separable at res 672 (single
AUROC 0.84) → the monuseg fg-IoU-0.6 bottleneck is DOWNSTREAM (resolution / boundary precision / decoding), NOT
coarse correspondence. Code kept default-OFF behind the `mproto` token (recorded negative); pre-screen saved a
multi-hour A/B. → prioritise Lever 2 (Boundary DoU, crisp fg boundaries); Lever 3 FAPM (feature-discrimination
adapter) now LESS likely to help (discrimination isn't the gap).

---

### C16 — Boundary DoU (`bdou`) loss term — fast fg-IoU screen + diagnostic → NO-GO / DROP

| field | value |
|---|---|
| method | `head_fusion_best_cgate_film_nobank_bdou` — Boundary DoU (Sun MICCAI'23, ref-verified α=1−2C/S≤0.8) added to the adaptive menu, gated by inst_density × (1−thinness) (dense COMPACT instances only). Motivated by the monuseg fg diagnostic (precision 0.68 vs recall 0.89 = over-prediction). |
| screen | FAST fg-IoU screen (`--metric_override fg_iou`, dual-GPU sep caches) monuseg+dsb (target) + spheroidj/drive/microtubules (control); + a monuseg precision/recall re-diagnostic (`monuseg_fg_diag.py`) |
| K/res/seeds | 8 / 672 / 3; caches `asg_cache_bdfg{0,1}`; date 2026-07-18 |

Results (fg-IoU, bdou vs best_v2): monuseg 0.629→**0.630** (Δ+0.002), dsb2018 0.845→**0.847** (Δ+0.002) — both TIED
(< +0.01 GO bar). Diagnostic (bdou vs base): precision 0.683→**0.678** (UNCHANGED), recall 0.885→0.888, boundary-band
err 0.521→0.501 (marginally better), detection 0.970→0.979.

VERDICT: **NO-GO → DROP.** Boundary DoU does not reduce the over-prediction (precision stays 0.68). The monuseg
bleed is **diffuse INTERIOR** (stroma painted fg away from boundaries; only ~½ errors boundary-local), and Boundary
DoU's α down-weights the interior → wrong tool. Code kept default-OFF behind `bdou` (recorded negative).
**LESSON:** fg-IoU is the WRONG screen metric for over-prediction levers — it is blind to the bled pixels that MERGE
touching nuclei (the instance-AP killer per C6 oracle-fg). Next levers targeting precision MUST be screened on
instance-AP. → Next: a PRECISION-favoring term (penalize interior FP) for dense-compact instances, AP-evaluated.

---

### C17 — Precision-Tversky (`prec`, α>β) loss term — monuseg diagnostic → NO-GO / DROP

| field | value |
|---|---|
| method | `head_fusion_best_cgate_film_nobank_prec` — Tversky α=0.7/β=0.3 (penalise FALSE POSITIVES), gated dense-compact (inst_density × (1−thinness)). Direct response to C16's diffuse-interior over-prediction (precision 0.68). |
| screen | fast monuseg fg diagnostic (fit best_v2_prec 1 seed + precision/recall decomposition) vs base; date 2026-07-18 |

Results (prec vs base, monuseg): precision 0.683→**0.687** (STUCK), recall 0.885→0.890, fg-IoU 0.620→0.627,
boundary-band 0.521→0.528. A precision-favouring loss barely moved the operating point.

VERDICT: **NO-GO → DROP.** Code kept default-OFF behind `prec` (recorded negative). **STRATEGIC CONCLUSION:** BOTH
a boundary loss (C16 bdou) AND a precision loss (C17 prec) leave monuseg precision stuck at ~0.68 — and the fg
threshold is FIXED at prob 0.5 (`foreground_from_score`, not adaptive), so the loss *should* have moved it. Combined
with the mproto finding (coarse fg/bg is already separable, AUROC 0.84), this proves the monuseg over-prediction is
**FEATURE / ROC-limited, NOT loss-limited.** Loss levers are exhausted for this gap. → Next direction = FEATURES /
FUSION / operating-point: (a) self-configuring fg THRESHOLD from support-LOO (cheap, moves along the ROC — precision
for excess recall → fewer merges → AP↑); (b) repulsive classical-ridge prior INTO the fg head (suppress fg at cell
membranes/stroma → sharper nucleus boundary); (c) FAPM/high-fidelity feature adapter (raises the ROC, higher ceiling
but fully-supervised in source → riskier at K≈8). See [[foreground-is-the-bottleneck]].

---

### C18 — Frozen feature upsampler (AnyUp) on monuseg fg — de-risk proxy for JAFAR → NEGATIVE

| field | value |
|---|---|
| method | best_v2 with `--feat_upsampler anyup --upsampler_factor {2,4}` (frozen DINOv3-feature upsampler densifying the coarse grid) vs base best_v2. monuseg fg diagnostic (1 seed). Ran as the CHEAP proxy for the JAFAR direction (JAFAR isn't pip-installable + research flags it aligns to low-level H&E texture). |
| date | 2026-07-18 |

Results (monuseg, vs base precision 0.683 / recall 0.885 / fg-IoU 0.620): anyup2 → prec **0.666** / rec 0.897 /
fg-IoU **0.613**; anyup4 → prec **0.670** / rec 0.892 / fg-IoU **0.614**. Both WORSE (lower precision = MORE
over-prediction).

VERDICT: **NEGATIVE.** A frozen feature upsampler HURTS dense-H&E fg — it aligns DINOv3 features to low-level
chromatin/stroma texture → increases the stroma bleed (precision ↓). JAFAR is the same class (aligns to low-level
edges even harder) → the de-risk proxy predicts JAFAR would also hurt; NOT building the messy JAFAR integration.
**Consistent with [[encoder-upsampler-verdict]]** (AnyUp helps only the finest filaments, never dense blobs).
**Six probes now agree** the monuseg fg gap is a hard FEATURE CEILING that cheap/safe single-mechanism levers cannot
crack (C15 correspondence, C16 boundary loss, C17 precision loss, threshold=modest, upsampler=NEGATIVE) — likely the
K≈8-vs-thousands data-asymmetry wall the specialists don't face. Achievable = uniform ≥ best_v2 (+ a safe
support-calibrated threshold); beating the monuseg specialist outright at K=8 appears fundamentally gated.

---

### C19 — Foreground-consistent K-scaling + K=8 hrf fill (paper "Exemplar" Fig 3 + Table 1)

| field | value |
|---|---|
| method | `head_fusion_best_cgate_film_nobank` (best_v2 = Exemplar) vs UniverSeg / Tyche / INSID3 (steelman = guided-filter on thin) / Matcher (one-shot, K=1 only) |
| K / protocol | K=1,4,8,16; pool 20, test 24, seeds 6, res 672 (ours); multi-draw fixed-pool |
| metric | FOREGROUND-consistent (fair to semantic-only baselines; matches Table 1): fg-IoU for blobs/nuclei (spheroidj/dsb/monuseg/ctc), clDice for vessels/filaments (drive/hrf/microtubules). Instance datasets re-scored `--metric_override fg_iou` |
| datasets | Table 1: all 7 @K=8. Fig 3: fixed common-5 {spheroidj,dsb,monuseg,drive,hrf} present at ALL K (ctc/microtubules cap K≤8) |
| score_dir | ours dsb/monuseg fg-IoU `scores_fgk/ours_k{1,4,8,16}`; ctc `scores_fg_inst/k{1,4,8}_*`; hrf@K8 `scores_hrfk8`; baselines `scores_fgk/*`, `scores_basekscale/k*_{tyche,universeg,insid3,insid3guided}` |
| cache | `asg_cache_fgk_dino` (GPU0 ours+insid3), `asg_cache_fgk_base` (GPU1 tyche+universeg) — SEPARATE writable caches (no race) |
| launcher | `scripts/driver_gpu0.sh` / `driver_gpu1.sh` (sota_final.py run … --metric_override fg_iou); `make_final_kscale.py` (common-5 aggregate + 95% CI figure); `make_table_data.py` (per-dataset mean±std) |
| date | 2026-07-18 |

Results (Table 1, K=8, fg metric, mean±std — **ours wins ALL 7**): spheroidj 0.883±0.026, dsb 0.826±0.026, monuseg 0.626±0.014,
ctc 0.743±0.013, drive 0.677±0.021, hrf 0.679±0.004, microtubules 0.835±0.007.
K-scaling (mean fg over common-5, 95% CI, n=6): ours 0.692→0.722→0.738→0.755 (K=1→16); Tyche 0.344→0.463→0.489→0.501;
UniverSeg 0.203→0.366→0.423→0.454; INSID3 (steelman) 0.350→0.379→0.425→0.428; Matcher (one-shot) 0.313.

VERDICT: **Exemplar dominates every dataset at every K** (even K=1 beats all few-shot baselines on all 7). Paper reframed on
all-K dominance (NOT K=8-specific), per user. Fig 3 = disclosed mean-of-foreground-metrics over fixed common-5, 95% CI bands,
legend OUTSIDE curves, Matcher as one-shot marker (K>1 dropped — extrapolation beyond its documented ~5-shot config + slow).
INSID3 unified to steelman (guided on thin) in BOTH figure and Table 1.

---

### C20 — DINO-only classical-bank ablation (`nocls` token) — bank is the largest contributor

| field | value |
|---|---|
| method | `head_fusion_best_cgate_film_nobank_nocls` — NEW `nocls` token zeros the classical prior bank (shape-preserving: `_classical` returns `zeros_like(feats)`; head trained+tested on zeros = pure DINO head). Decoder ridge (image-Frangi) untouched. vs full best_v2 |
| K / protocol | K=8, pool 20, test 24, seeds 6, res 672 |
| datasets | all 7 (native primary metric) |
| score_dir | `scores_nocls` · cache `asg_cache_nocls` |
| code | `dino_only` flag in `HeadFusionBackend.__init__` + `nocls` token in `make_backend` (default-OFF, parity-safe) |
| date | 2026-07-18 |

Results (full → DINO-only, bank_gain): spheroidj fg-IoU 0.883→0.882 (**+0.002**); ctc AP 0.444→0.375 (+0.069);
dsb AP 0.494→0.307 (+0.187); drive clDice 0.677→0.465 (+0.211); monuseg AP 0.224→**0.025** (+0.199). (hrf/microtubules were finishing.)

VERDICT: **Classical bank is the LARGEST single contributor** — negligible on well-separated blobs (+0.002) but ESSENTIAL for
every crowded/thin morphology (monuseg AP collapses to 0.025 without it). Justifies the hybrid frozen-DINOv3 ⊕ classical design;
ties to wall analysis (bank lifts monuseg 0.025→0.224, but specialist 0.405 → bank helps, does NOT fully close the
frozen-representation gap). In paper as the "backbone only" ablation sentence. See [[hyperbank-fusion]], [[foreground-is-the-bottleneck]].

---

### C21 — DSB2018 specialists on our split + INSID3 K=16 steelman (Table 3 + Fig 3 completion)

| field | value |
|---|---|
| method | StarDist-fluo (2D_versatile_fluo) + Cellpose-SAM (cpsam, cellpose 4.2.1.1) on dsb2018, instance-AP, OUR test split (K=0 off-the-shelf, deterministic) |
| protocol | seeds 0–5, test 24, same `score_prediction(instance_ap)` as ours |
| envs | `/disk2/prusek/stardist_env` (TF), `/disk2/prusek/cellpose4_env` (cellpose≥4) — CPU (`CUDA_VISIBLE_DEVICES=""`) to avoid GPU contention |
| out | `base_full/cs_stardist_fluo_dsb.json`, `base_full/cs_cellpose_cpsam_dsb.json`; `scores_basekscale/k16_insid3guided` (ASG_CRF=guided) |
| launcher | `cellpose_stardist_bench.py --backend {stardist_fluo,cellpose_cpsam} --datasets dsb2018 --seeds 0 1 2 3 4 5` |
| date | 2026-07-18 |

Results (dsb2018 instance-AP): ours(K=8) 0.494±0.033, StarDist-fluo **0.622**, Cellpose-SAM **0.645**, micro-SAM (not run on dsb).
Reproduces published anchors. INSID3 steelman completed @K=16 (guided, drive/hrf) → fixes Fig 3 INSID3 K16 dip (0.388→0.428, monotonic).

VERDICT: Table 3 (specialist gap) extended to 3 datasets (dsb/monuseg/ctc). Gap = 1.3× (dsb) to 2× (monuseg) — honest few-shot
cost on standard cells; NOT beating specialists on home ground (per [[method-goal-beat-specialists]]).

### C22 — Encoder/backbone + SegGPT baseline study (paper reviewer #3/#4 defence + encoder ablation)

| field | value |
|---|---|
| protocol | all runs `--seeds 6 --pool 20 --test 24 --support 8 --res 672`, same metrics as Table 1 (fg-IoU blobs/nuclei, clDice vessels/filaments, instance-AP for ablation MoNuSeg) |
| SegGPT | `sota_final.py run --method seggpt --datasets <7> --score_dir results/scores_seggpt` (BAAI/seggpt-vit-large, native K-shot feature-ensemble) |
| UNI2-h histo | `--method head_fusion_best_cgate_film_nobank --model hf-hub:MahmoodLab/UNI2-h --datasets monuseg,dsb2018,drive --score_dir results/scores_histo_uni2h` |
| OpenPhenom | `--method head_fusion_best_cgate_film_nobank --model openphenom:recursionpharma/OpenPhenom --datasets dsb2018 --score_dir results/scores_openphenom_smoke` (1-seed smoke) |
| ConvNeXt (aniso) | `--method head_fusion_best_cgate_film_nobank_coarseonly --model facebook/dinov3-convnext-large-pretrain-lvd1689m --datasets spheroidj,monuseg,drive,microtubules --score_dir results/scores_convnext` |
| early-layers | `--method head_fusion_best_cgate_film_nobank --layers=-1,6 --score_dir results/scores_ablation/layers16` |
| coarse/fine/pd | `_coarseonly` / `_fineonly` / `_pd64` / `_pd128` tokens, `--score_dir results/scores_ablation/{tag}` |
| cache | separate writable caches per encoder (`cache_convnext`, etc.) — no shared-cache race |
| date | 2026-07-19 |

Results vs best_v2 DINOv3 (ours):
- **SegGPT** (Table-1 metric): spheroidj **0.900** (>ours 0.883, paired p=0.0016), ctc **0.756** (>0.743); loses dsb 0.771/0.826,
  monuseg 0.124/0.626, drive 0.400/0.677, hrf 0.234/0.679, micro 0.641/0.835. Variance 5–10× ours. Ours leads 5/7, never <0.63.
- **UNI2-h** (H&E histo ViT-H): monuseg fg-IoU 0.640 (+0.014), dsb 0.818 (−0.008), drive 0.711 (+0.034). Domain backbone barely dents dense-nuclei wall.
- **OpenPhenom** (microscopy MAE ViT-S): dsb 0.825 ≈ DINOv3 0.826.
- **ConvNeXt** (stride8 aniso, 6-seed): spheroidj 0.791 (−0.09), monuseg-AP 0.255 (+0.031), drive 0.739 (+0.062), micro 0.788 (−0.047) → MIXED, not a broad win.
- **early-layers** (−1,6 fusion): spheroidj 0.886, monuseg-AP 0.221, drive 0.695, micro 0.830 → neutral, no gain.
- **coarse/fine**: both(best_v2) micro 0.835 > coarse 0.822 / fine 0.817 (scale-fusion helps finest filaments; neutral/slightly-neg on drive).
- **proj_dim**: 32 (drive 0.677/micro 0.835) vs 64 (0.692/0.825) vs 128 (0.693/0.812) → 32 sufficient, no consistent gain.

VERDICT: no off-the-shelf encoder (H&E histo / microscopy-MAE / anisotropic ConvNeXt / early-layers) broadly beats frozen DINOv3-L;
even a domain-matched histo backbone lifts dense-nuclei fg only +0.014 → the wall is frozen-features-at-K-labels, closable only by
feature fine-tuning (specialists), not a backbone swap. Integrated into paper: SegGPT → Table 1 (ours leads 5/7 + never-collapse framing);
UNI2-h → §4 turns speculation into tested evidence (defuses reviewer #4); coarse/fine/proj_dim → ablation sentence. See [[isbi-exemplar-paper]].

### C23 — Benchmark-fairness / leakage audit + harness fixes (NO new lever; INVALIDATES prior p-values)

| field | value |
|---|---|
| trigger | user question: "is the comparison fair, is there data leakage — it is strange that it dominates everything" |
| scope | audit of the whole eval path (splits, self-config data flow, baseline protocols, stats), then code fixes |
| leakage verdict | **NONE FOUND.** Test GT never reaches the model: `fit(support)` / `foreground(image, feat_grid)` / `predict(...)` take no label argument. All self-config levers read support only; affinity thresholds are per-test-IMAGE (transductive, legitimate), never per-test-MASK. Normalisation is per-image. No early stopping / epoch selection. `load_fewshot`/`load_dsb2018`/`_load_instance_split` take first-N from separate `test/` dirs; CTC-U373 = seq01 train / seq02 test (disjoint movies); MoNuSeg = official HF split. Tables 1+2 were split-fair as run. |
| fix 1 | `reset_support_state()` (head_fusion_backend) + `reset_backend_for_new_support()` (segment/base), wired into `sota_final.py` + `panel_benchmark.py`. Gates latch on `is None` and were never reset while the harness only nulled `head`, so seeds 1..N reused **seed 0's** colour channel / CLAHE / thin gate / affinity calibration / FiLM prototypes. Also restores `tile_classical`/`fine_scales`/`trainable_classical` and drops `_bank`/`_n_classical`/`_ccache` (`_fcache` is encoder-only, kept). Demonstrated: thin draw then blobby draw kept `_thin_active=True` (tubularity 0.910 carried into a 0.017 draw) and never re-logged the gate. |
| fix 2 | `stats` pseudoreplication. Test split is loaded ONCE (seed=0) so all 6 seeds score the SAME 24 images; `wilcoxon` ran on 6x24=144 "independent" pairs. Now `_per_image_mean` averages over seeds first (n = test_per_seed). Synthetic check: p 1.5e-22 (inflated) -> 1.2e-07 (correct), ~15 orders. **Every previously reported paired p-value is inflated and must be recomputed.** |
| fix 3 | `split_fingerprint()` (sha256 of test image+mask bytes) + `protocol` block written into every score json; `_pairing_problem()` refuses to pair records differing in metric / test_per_seed / split_fp, and the summary table flags MIXED METRICS and MIXED TEST SPLITS rows. The old guard checked length + test_per_seed only — shape, never identity. |
| fix 4 | baseline split protocol. `persam_bench` / `microsam_bench` / `cellpose_stardist_bench` called `load_dataset(spec, support, test, seed=seed)` vs harness `(pool=20, test=24, seed=0)`. `load_flat_fewshot` slices `permutation(seed)[support:support+test]`, so on **download-kind** sets (kvasir/hrf/drive/isbi2012em) they scored different images. Measured overlap with ours: kvasir 12/24 @seed0 and 0-1/24 @seeds1-5; drive 9-13/24; hrf 11-14/24. Latent only (those baselines ran on dsb/instance-kind sets, whose loaders ignore seed+support) — now aligned to the harness protocol + `--pool`. |
| fix 5 | `driver_gpu0.sh` ran Matcher on `$D5` without `--metric_override fg_iou`, so dsb2018/monuseg were AP-scored and `make_final_kscale.py`'s metric filter silently DROPPED them, collapsing Matcher to a K=1-only point. Split into `_fg` (override) + native runs; `make_final_kscale` DIRS updated. The paper's "Matcher is a one-shot method" framing needs re-checking against the re-run. |
| fix 6 | `_calibrate_instance` docstring claimed the two support-derived scalars are the decoder's only fitted state while `merge_ridge=0.25` / `w_dt` / `w_ridge` / `min_area` stay at fixed defaults (`predict` passes only `r_star`/`merge_cos`). Corrected: support-CALIBRATED, not fully self-configured. |
| tests | `tests/test_support_state_reset.py` (3 tests: fresh-state invariant, gate re-derivation across draws, cache policy). Suite 187 passed; 3 pre-existing failures are a missing local `pydensecrf`. |
| NOT fixed (methodological, needs a decision) | no validation split exists anywhere — every lever C1..C22 was accepted/rejected on the SAME 7 reported test datasets at a +0.01 screen gate, i.e. the selection threshold is the size of the reported fold-in gain (+0.021). Panel-fitted constants: `thin_max_side=1500` (placed between hrf 3504px and the rest because hrf fell 0.607->0.469), adaptive-loss ramps "measured on the panel", adaptive_superres gated off for instance+huge-downscale after observed regressions. Recommend holding out >=2 datasets used in NO lever decision. |
| date | 2026-07-19 |

Re-run required (numbers change): every `sota_final.py run` for ours AND in-harness baselines (reset changes per-seed
self-config -> means and especially **std**), then `stats` for all p-values. Expect our reported std to GROW: the paper's
"five to ten times our per-image variance" claim and HRF's +-0.004 leaned on the frozen seed-0 configuration.

### C24 — PerSAM baseline VERIFIED against its published anchor (fairness protocol, no lever)

| field | value |
|---|---|
| trigger | user: "ověř ten persam" — campaign produced fg-IoU 0.050 (PerSAM) / 0.202 (PerSAM-F) on dsb2018, low enough to look like a broken baseline |
| question | is the low number our integration failing, or the method's ceiling? |
| anchor run | upstream `persam.py` on PerSeg (40 objects), scored by upstream `eval_miou.py` |
| exact command | `cd /disk2/prusek/Personalize-SAM && /disk2/prusek/persam_env/bin/python persam.py --data "/disk1/prusek/incontext/PerSeg_data/data 3" --ckpt /disk2/prusek/sam_vit_h_4b8939.pth --outdir /disk1/prusek/persam_anchor_out` then `eval_miou.py --pred_path /disk1/prusek/persam_anchor_out --gt_path "/disk1/prusek/incontext/PerSeg_data/data 3/Annotations"`. NOTE both scripts prepend `./outputs/` to the path given, so masks land in `Personalize-SAM/outputs/<abs path>/`. |
| **RESULT** | **mIoU 89.32, mAcc 92.19 vs 89.3 published** (Zhang et al., ICLR'24, Tab. 1). Environment, SAM checkpoint and the `per_segment_anything` fork are validated; `persam_bench.py` reuses all three. |
| behavioural cross-check | spheroidj (1.0 obj/img) 0.837 vs dsb2018 (55.1 obj/img) 0.050 — a broken integration fails on BOTH, this fails only where the method is built to |
| quantitative cross-check | on dsb2018 a PERFECT segmentation of one median instance against the whole foreground is IoU **0.0406** (measured over 20 test images: fg fraction 0.130, 55.1 instances/image). PerSAM measures 0.0498–0.0564, i.e. slightly ABOVE its own ceiling because SAM sometimes grabs a cluster. |
| verdict | the campaign's PerSAM numbers are CORRECT. Report them WITH the ceiling, never bare: "PerSAM 0.05" reads as a baseline we broke; "PerSAM is at the 0.04 structural ceiling of a one-object personaliser" is the true and more informative claim. |
| doc fixed | `persam_bench.py` docstring previously said no published number had been reproduced; that is now measured and recorded there. |
| **PerSAM-F anchor** | run the same way (`persam_f.py`, same data/ckpt): **mIoU 95.18, mAcc 95.57 vs 95.3 published**. The two-scalar fit therefore works correctly in this environment. |
| consequence for PerSAM-F | PerSAM-F scoring BELOW PerSAM on spheroidj (0.783 vs 0.837) is NOT a broken fit, and NOT a small-K artefact either: PerSeg also uses a single reference and there the fit GAINS ~6 points. It is a domain property of microscopy. Report it as measured; do not explain it away. |
| STILL OPEN | our K=8 multi-shot prototype scores below K=1 on spheroidj (0.764 vs 0.837). The multi-shot form is OUR adaptation, not PerSAM's method, so the K=8 column must be labelled as such or PerSAM is understated at its own best operating point. |
| date | 2026-07-19 |

### C25 — Classical bank crashed on images smaller than its largest filter (method bug, not a lever)

| field | value |
|---|---|
| trigger | `ours_k1_bacteria` / `_k4_` / `_k8_` FAILED during the kajman `ours` run |
| error | `RuntimeError: Padding size should be less than the corresponding input dimension, but got: padding (64, 64) at dimension 3 of input [1, 1, 66, 58]` |
| cause | `hyperbank_bank.gaussian_blur` reflect-pads by `ksize//2`. The bank's largest Frangi scale is sigma=16 → half-width 64, so ANY image narrower than 65 px raises and kills the whole dataset job. |
| **measured blast radius** | `bacteria`: 168 images, smallest side 58–2038 px, **median 596**; exactly **2** images below 65 px — (66, 58) and (62, 113). Two images in 168 destroyed the dataset. `bbbc010` (all 520 px) and `fisbe` (≥680 px) are unaffected; no other panel set has an image under 65 px. |
| correction | an earlier commit message said "bacteria is 66x58". Wrong: bacteria is a HETEROGENEOUS set that merely CONTAINS two tiny images. The distinction matters — the failure mode is a mixed-size folder, which is the normal deployment case, not a uniformly tiny dataset. |
| fix | clamp the Gaussian kernel to `min(ksize//2, H-1, W-1)` and re-normalise. A scale wider than the image cannot describe structure in it, so clamping degrades that scale to the widest the image can express. |
| safety argument | provably a no-op wherever the kernel already fits: `test_the_clamp_is_a_no_op_when_the_kernel_already_fits` asserts bit-for-bit equality against the unclamped path at every sigma the bank uses (1,2,4,8,16) on a 512×512 image. So this can only turn a crash into a result, never move an existing number — which is why it was safe to deploy mid-campaign. |
| tests | `tests/test_hyperbank_small_images.py` (8 tests). Mutation-verified: removing the clamp restores the crash; dropping the re-normalisation rescales image brightness (a constant image no longer blurs to itself), which would read as a contrast change rather than an error. |
| deployment relevance | the goal is a tool a biologist points at their own folder. A segmenter that raises RuntimeError on one small crop is not deployable, and the "self-configures across morphologies" claim does not survive a 58-pixel image. |
| date | 2026-07-19 |

### C26 — Training-free probe: is the MoNuSeg foreground ceiling linear-specific or fundamental?

| field | value |
|---|---|
| trigger | overnight research mandate: find what could break the dense-H&E-nuclei wall (fg IoU ~0.62 at K=8) |
| method | `scripts/monuseg_ceiling_probe.py` — extract frozen DINOv3 features, sample fg vs LOCAL-RING bg cells, compare a linear probe vs cosine-kNN vs a 1-hidden MLP vs local-background-subtracted-linear, scored as grid fg-IoU on the local ring (nucleus-vs-adjacent-tissue, the wall), evaluated on HELD-OUT QUERY images. No training loop; feature extraction only. 8 support, 15 query, 3 seeds. |
| result (monuseg) | linear 0.332±0.011, knn 0.325±0.004, **mlp 0.396±0.003 (+0.064)**, local_sub 0.331±0.022 |
| result (dsb2018, control) | linear 0.637±0.002, knn 0.666±0.003, **mlp 0.689±0.003 (+0.052)**, local_sub 0.637±0.001 |
| finding 1 | Nonlinearity (MLP) beats a linear rule on RAW features, tightly reproduced (std 0.003). So the info IS in the frozen features but is NOT linearly separable. kNN and local-background-subtraction do NOT help. |
| finding 2 (the honest caveat) | The MLP gain is NOT wall-specific: +0.064 on monuseg vs +0.052 on the dsb2018 control. It is a general "linear-on-raw-features leaves performance on the table" effect, not a dense-nuclei breakthrough. And +0.064 grid-IoU is far smaller than the gap to the oracle (0.62 vs 0.886). |
| **critical re-interpretation** | `DINOHeadFusion` ALREADY has a 2-layer 3x3-conv GroupNorm+GELU body (hidden=256), i.e. it is NOT linear on the raw features — it already supplies the nonlinearity this probe found. So the probe does NOT identify an unexploited lever; it confirms the existing body is doing its job and the residual 0.62->0.886 gap is beyond what more of the SAME nonlinearity buys. Leans toward "ceiling is fundamental at K=8 with a frozen backbone + light head", consistent with the recorded negatives (matching/losses/threshold/upsampler/self-training/backbone-swap each <=0.01). |
| next | deep-research (running) to check for a 2023-2026 technique that adds NEW INFORMATION (not just more nonlinearity) — test-time adaptation, feature-affinity refinement, a principled few-shot nonlinear classifier resistant to K=8 overfitting. If research surfaces nothing that plausibly beats the existing nonlinear head, the paper's "needs backbone fine-tuning" limitation claim is CONFIRMED, which is itself the useful finding. |
| date | 2026-07-19 |

### C27 — Fine-tuned specialist independently confirms the wall is the FEATURES (cross-check of C26)

| field | value |
|---|---|
| trigger | FT phase 1 cellpose_ft results landed while running the overnight mandate |
| monuseg (dense H&E, the wall) | cellpose_ft: fg_iou **0.716**, instance_ap **0.419** (10 seeds, e200). Ours: fg ~0.62, AP 0.224. Off-the-shelf cellpose AP 0.386. |
| dsb2018 (fluorescence, no wall) | cellpose_ft: fg 0.872, AP 0.647. Off-the-shelf 0.645 -> fine-tuning barely moved it. |
| finding | On H&E, fine-tuning Cellpose on the SAME 8 support masks LIFTS it (0.386->0.419 AP), unlike dsb2018 where it does nothing. So the support labels ARE useful on dense nuclei -- but only to a model that fine-tunes its FEATURES on the stain. |
| **cross-confirmation of C26** | The fine-tuned specialist, which tunes features on the stain, reaches foreground 0.716 where our FROZEN features plateau at ~0.62. That +0.10 foreground gap is exactly what feature fine-tuning buys, and it is exactly the gap our frozen-feature ceiling cannot close. This is independent evidence, from the specialist side, that the wall is the frozen features and the fix is stain-specific feature tuning -- the paper's limitation claim, confirmed empirically rather than asserted. |
| paper use | Report honestly: on dense nuclei a specialist trained on our exact 8 labels beats us ~2x on instance AP; this is the honest few-shot cost, and it localises the cause to the features (0.716 vs 0.62 foreground), motivating the Nature-Methods direction (lightweight backbone fine-tuning on the target stain). |
| date | 2026-07-19 |

### C28 — Deep-research verdict on the MoNuSeg wall (103 agents, adversarially verified)

| field | value |
|---|---|
| bottom line | The K≈8 H&E-nucleus ceiling is probably NOT a fundamental feature-GENERATION limit but a READOUT / feature-SELECTION gap. Backbone fine-tuning at K≈8 is the WRONG fix (overfits below zero-shot; every "PEFT/LoRA/full-FT beats a frozen head" claim was refuted 0-3 or 1-2). |
| RANK 1 (cheapest) | Nonlinear metric-space readout: FROST nonparametric KDE density-ratio, TPM tied-prototype (sigma_F<sigma_B inside/outside). Training-free, support-configured. CAVEAT: cousin of the already-tried multi-prototype correspondence (≤0.01); may collapse if the failure is feature SEPARABILITY not boundary shape. |
| RANK 2 | RePRI transductive TTA with a foreground-PROPORTION KL constraint (CE on support + entropy on query + KL to target fg proportion). Oracle proportion gives +11-14% mIoU, paralleling our oracle-foreground finding. The proportion constraint is the ONE signal our earlier self-training lacked. CAVEAT: RePRI's head is linear (pair with RANK 1); proportion is a scalar, and RePRI base-trains in-domain (DINOv3 does not). |
| RANK 3 | Light nonlinear attention decoder on frozen features (AMFormer/DEAP/HDMNet). Highest ceiling, needs a training loop, episodically meta-trained (K≈8 is extrapolation). The specific "detail-mining forces clean boundaries against bleed" claim was REFUTED 0-3. |
| **strongest confirmed finding (3-0)** | LAYER-SELECTION gap: an oracle per-episode DINOv3 layer selection reaches +7-13 mIoU over the last-layer baseline, and current heuristic selectors fall BELOW last-layer — substantial headroom hides in frozen features that last-layer readout does not access. Our probe (C26) used ONLY the last layer, so this is untested here. |
| tension with C27 | C27 showed a nucleus-pretrained specialist (Cellpose) reaches foreground 0.716 by tuning features on the stain. This is NOT a contradiction: the research says do not FT DINOv3 at K=8, but a model whose features were ALREADY nucleus-pretrained (Cellpose) is the specialist advantage, consistent with "needs stain-tuned features" for the deployed tool while readout/selection is the K=8 lever for the frozen path. |
| honest caveats (from the research) | NO cited method is demonstrated on H&E dense nuclei at K≈8; the cheapest levers were validated on remote sensing / abdominal MRI; several are cousins of already-tried levers; the selection-gap oracle is unrealizable (proves headroom exists, not that a support-LOO selector reaches it); two "frozen SOTA" methods (FS-DINO, Mimic) covertly use SAM. |
| next | Extend the probe with (a) a DINOv3 LAYER SWEEP on monuseg (does a non-last layer separate nucleus-vs-tissue better?) and (b) a FROST-style KDE readout. Cheapest test of the two best-confirmed levers before building anything. Run when a GPU frees (all busy now: SegGPT g1, reruns g0, campaign+FT tulen). |
| date | 2026-07-20 |

### C29 — Layer-sweep probe: DINOv3 layer 12 >> last layer for foreground separation (GO for a smoke test)

| field | value |
|---|---|
| trigger | testing the deep-research (C28) strongest-confirmed lever, the layer-selection gap |
| method | `scripts/monuseg_ceiling_probe.py --layers=-1,18,12,6` — same training-free probe, per DINOv3 layer |
| monuseg | layer -1: linear 0.332 / mlp 0.397. layer 12: linear **0.398** / knn **0.441** / mlp 0.423. Best = layer 12 / knn = 0.441, **+0.109** over last-layer linear. |
| dsb2018 (control) | layer -1: linear 0.637 / mlp 0.688. layer 12: linear 0.674 / knn **0.778**. Best = layer 12 / knn = 0.778, **+0.141**. |
| finding | The LAST layer (which best_v2 uses) is the WORST for fg separation; a MID layer (12) is dramatically better on BOTH datasets. Even a LINEAR readout on layer 12 beats an MLP on the last layer. knn is the best readout on layer 12 (mid-layer features are more locally clustered). FROST-style KDE readout did NOT help (collapsed to linear, exactly the research's caution). |
| honest caveats | (1) This is a PROBE metric (grid-resolution, local-ring, oracle-threshold) -- a proxy, not the real task; the true smoke test on the designated metric is the arbiter. (2) The gain is NOT wall-specific -- it is LARGER on the dsb2018 control (+0.141) than on monuseg (+0.109), so this is a general "wrong layer" finding, potentially a whole-panel lever rather than a MoNuSeg breakthrough. (3) best_v2 uses the last layer deliberately (semantic content for the affinity instance decoder's merge-veto and cross-morphology robustness); a fixed switch to layer 12 risks regressing those. |
| lever to test | LAYER FUSION `--layers=-1,12` (head learns the per-layer weighting from the K masks; no-regression by design, captures the layer-12 benefit where it helps) AND single `--layer 12` (upper bound of the benefit). Both are supported flags, not new code. |
| screen | best_v2 baseline vs +L12 vs +fusion, on monuseg + dsb2018 (targets) + drive + spheroidj (controls), 3 seeds, designated metric. GO gate: improve a target by a clear margin, no control regression, no crash. Only on GO -> integrate + full re-run (user-authorised). |
| date | 2026-07-20 |

### C30 — Layer-fusion smoke screen: promising lead, NOT a clean win, does not solve MoNuSeg

| field | value |
|---|---|
| trigger | GO-gate arbiter for the C29 layer-12 finding |
| protocol | best_v2 baseline vs `--layer 12` vs `--layers=-1,12` (fusion), monuseg+dsb2018 (targets) + drive+spheroidj (controls), 3 seeds, designated metric. base reproduced the campaign (monuseg 0.620 vs 0.622, dsb 0.853 vs 0.846), so the comparison is trustworthy. |
| result | monuseg: base 0.620 / L12 0.657 (+0.036) / fuse 0.639 (**+0.019**). dsb2018: 0.853 / 0.856 / 0.853 (+0.001). drive: 0.693 / 0.728 (+0.034) / 0.712 (**+0.018**). spheroidj: 0.897 / 0.851 (**-0.046**) / 0.891 (**-0.006**). |
| single-L12 verdict | FAILS the GO gate: regresses the spheroidj blob control by -0.046. This cleanly explains why best_v2 uses the last layer -- mid-layer 12 helps dense/thin structures (nuclei, vessels) but hurts compact blobs, which need last-layer semantics. The layer tradeoff is real. |
| fusion verdict | The head learned the per-layer weighting from K masks and cut the spheroidj regression from -0.046 to -0.006, while keeping ~half the L12 gain (monuseg +0.019, drive +0.018). But -0.006 is JUST past the -0.005 control-regression threshold, so fusion MARGINALLY fails the gate. On 3 seeds and high-variance spheroidj, -0.006 is plausibly noise. |
| decision | PROMISING LEAD, not a clean win, and it does NOT solve MoNuSeg (+0.019 does not close the 0.62->0.72 specialist gap). Per the user's bar ("if it SOLVES monuseg -> fold into main + re-run whole panel") the auto-fold condition is NOT met, and the borderline spheroidj needs the post-campaign rigorous A/B to resolve. NOT folded into main; recorded and presented for the user's decision. |
| if pursued later | run fusion vs base at 6-10 seeds on the full panel to resolve spheroidj and confirm the monuseg/drive gains reproduce; it is a candidate WHOLE-PANEL lever (helps nuclei AND vessels) if the blob regression proves to be noise. The clean scientific finding either way: the last DINOv3 layer is suboptimal for dense/thin structures and a support-learned layer fusion recovers most of the benefit with little blob cost. |
| date | 2026-07-20 |

### C31 — Real Table 1 with SegGPT (K=8, 10 seeds, fixed harness): SegGPT beats us NOWHERE significantly

| field | value |
|---|---|
| trigger | SegGPT complete on all 6 reported + common-5 datasets (only held-out bacteria K8/K16 still running) |
| source | kajman tree, `stats --support 8 --ours ours`; ours vs seggpt, Holm-adjusted paired Wilcoxon |
| ours vs SegGPT | monuseg +0.495 (p_holm 4e-3*), hrf +0.445 (4e-6*), drive +0.293 (1e-4*), isbi2012em +0.262 (4e-3*), fisbe +0.171 (6e-3*), bbbc010 +0.153 (1e-9*), dsb2018 +0.069 (4e-13*), spheroidj +0.006 (0.43 ns), ctc_u373 -0.019 (0.26 ns) |
| verdict | Ours SIGNIFICANTLY beats SegGPT on 7 of 9 datasets, TIES on 2 (spheroidj, ctc_u373), loses significantly on ZERO. Even SegGPT's one nominal win (ctc_u373) is NOT significant after Holm. |
| vs smoke test | The 2-seed smoke test had hinted SegGPT beat us on spheroidj AND ctc_u373; at 10 seeds on the fixed harness, spheroidj flips to a near-tie (ours +0.006) and ctc_u373's SegGPT lead is not significant. The full protocol is kinder to us than the smoke test, not harsher. |
| paper impact | The "most accurate few-shot method at every K" claim is SAFE against SegGPT, the strongest in-context competitor. Report honestly: ties on ctc_u373 (phase-contrast) and spheroidj (compact blobs); dominates everywhere else, especially the wall datasets (monuseg +0.495, hrf +0.445). |
| still pending | Matcher K1 + PerSAM (tulen, finishing), off-the-shelf specialists, held-out bacteria SegGPT. Then merge tulen+kajman and regenerate the full tables/figure. |
| date | 2026-07-20 |

### C32 — Foreground-PROPORTION prior (C28 RANK 2 / RePRI) — training-free pre-screen: NO-GO

| field | value |
|---|---|
| trigger | C28 RANK 2 named a foreground-proportion KL constraint "the ONE signal our earlier self-training lacked" (RePRI: oracle proportion = +11-14 mIoU). C16 measured our MoNuSeg failure as OVER-prediction (precision 0.683 / recall 0.885 / detection 0.97) => predicted fg area = R/P = **1.296x** the true area. A proportion constraint attacks exactly that — IF the proportion can be estimated without an oracle. |
| pre-screen question | Can the K=8 support masks estimate the QUERY fg proportion well enough to constrain on? Pure GT-mask statistics: no GPU, no model, no training. |
| script | `scripts/fgprop_prescreen.py` (uploaded to tulen) |
| command | `cd /disk1/prusek/active-segmenter && PYTHONPATH=. ~/dinov3_env/bin/python scripts/fgprop_prescreen.py` |
| protocol | `load_dataset(spec, pool=20, test=24, seed=0)` once, K=8 subsampled per seed, 10 seeds. Targets monuseg (the wall) + dsb2018/ctc_u373 (same metric); controls spheroidj (blobs) + drive (vessels). |

Results — relative error of the support-derived proportion estimate (all fractions of the true dataset fg fraction):

| dataset | true test_fg | per-img spread | support est | bias | dataset-level MAE | per-image MAE |
|---|---|---|---|---|---|---|
| monuseg | 0.2157 | 0.187 | 0.2535 | **+0.175** | 0.179 | 0.239 |
| dsb2018 | 0.1173 | 0.676 | 0.1633 | **+0.392** | 0.392 | 0.817 |
| ctc_u373 | 0.0506 | 0.162 | 0.0822 | **+0.623** | 0.623 | 0.623 |
| spheroidj | 0.1233 | 0.889 | 0.0885 | **-0.282** | 0.282 | 0.806 |
| drive | 0.0861 | 0.084 | 0.0872 | **+0.013** | 0.033 | 0.091 |

| field | value |
|---|---|
| **verdict** | **NO-GO.** On monuseg the support prior is itself biased **+17.5% in the SAME direction as our error** — constraining to it would move over-prediction only 30% -> 17.5%, not close the 0.62 -> 0.716 fg gap. On three of five datasets (dsb2018 +39%, ctc_u373 +62%, spheroidj -28%) the prior is large AND wrong-signed, so an unconditional constraint would REGRESS the controls: it fails the no-regression gate by construction. |
| why the prior fails | MoNuSeg train/test are different organs (genuine density shift), and dsb2018/spheroidj have huge per-image fg heterogeneity (per-img spread 0.68 / 0.89) — no single global proportion can describe them, oracle or not. |
| the one real signal | **drive** (vessels): bias +1.3%, dataset MAE 0.033, per-image MAE 0.091 — vessel fg fraction is near-constant, so the prior IS accurate there. A self-gated variant (constrain only when the support's own fg-fraction dispersion is low — measurable from the support alone, so it stays self-configuring) is defensible, but drive is not the bottleneck (0.677) so the expected value is small. Not pursued now. |
| cost saved | Minutes of CPU killed a lever that would have needed a full transductive-TTA implementation plus a multi-hour panel A/B. |
| date | 2026-07-20 |

### C8-STATUS — recovered 2026-07-20: the C8 layer-fusion A/B produced NO comparison (only the baseline arm ran)

| field | value |
|---|---|
| why this entry | C8 has read "Results: RUNNING ... Update when `LF_DONE`" since 2026-07-16, and a lever audit flagged it as a possibly-lost 6-seed/7-dataset A/B that would supersede the C30 smoke screen. Recovered from tulen rather than left ambiguous. |
| what is actually on disk | `/disk1/prusek/active-segmenter/results/scores_lf/` contains **one** subdir, `single/` (the `--layers ""` BASELINE arm), with 3 jsons (spheroidj, dsb2018, hrf; the microtubules json the log reports was written was later deleted when microtubules left the panel). **`lf12/` and `lf1226/` do not exist.** |
| launcher logs | `results/lf_g0.log`: `[g0 single]` spheroidj fg_iou 0.884±0.020, dsb2018 ap 0.449±0.028, microtubules cldice 0.798±0.033, then `SOTA_RUN_DONE`. `results/lf_g1.log`: `[g1 single]` hrf cldice 0.617±0.017 — and nothing after. `results/layerfusion.log` is 0 bytes. |
| **verdict** | The run died after the BASELINE arm. The two layer-fusion arms (`-1,12` and `-1,12,6`) **never executed**, so C8 never compared anything. It does NOT supersede C30 and there is nothing to recover. The C30 follow-up (fusion vs base, 6-10 seeds, full panel) is still the outstanding experiment. |
| secondary note | C8's baseline arm used `head_fusion_adaptive`, not `best_v2`, so even its baseline numbers are not a usable reference for the current method. |
| date | 2026-07-20 |

### C33 — Operating-point (threshold) sweep scored on INSTANCE-AP — the screen C16 mandated and nobody ran

| field | value |
|---|---|
| trigger | C16 recorded verbatim: "fg-IoU is the WRONG screen metric for over-prediction levers ... Next levers targeting precision MUST be screened on instance-AP." Every probe after it (C17 prec, C18 anyup, C22 backbones, C26/C29 readout, C30 layer fusion) was nevertheless screened on fg-IoU or the fg diagnostic. C17 named the untested consequence as its next direction (a): the fg threshold is HARD-WIRED at prob 0.5 (`foreground_from_score(logits, hw, thresh=0.0)`) while monuseg sits at precision 0.683 / recall 0.885 / detection 0.970 — excess recall available to trade. The experiment log summarises this probe as "≤0.01" with **no numbers logged anywhere**, so it was re-run properly. |
| script | `scripts/thresh_ap_sweep.py` |
| command | `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. ASG_THR_CACHE=/disk1/prusek/asg_cache_thrap ~/dinov3_env/bin/python scripts/thresh_ap_sweep.py` (separate cache — GPU1 held `fg_sensitivity.py` concurrently; no shared writable cache) |
| protocol | best_v2, K=8, pool 20 / test 24, res 672, **2 seeds**, τ ∈ {0.30 … 0.80}, BOTH fg-IoU and instance-AP recorded per τ |

| dataset | τ=0.50 fg-IoU / AP (current) | best τ (TEST-chosen) | AP at best τ | ΔAP | Δfg-IoU |
|---|---|---|---|---|---|
| monuseg | 0.616 / 0.218 | **0.70** | 0.236 | **+0.018** | +0.022 |
| dsb2018 | 0.843 / 0.518 | 0.40 | 0.522 | +0.004 | −0.000 |
| ctc_u373 | 0.724 / 0.411 | 0.55 | 0.415 | +0.004 | +0.001 |

MoNuSeg AP is **monotone in τ** over 0.30→0.70 (0.179 / 0.201 / 0.218 / 0.225 / 0.229 / 0.234 / 0.236) then falls at 0.80.

| field | value |
|---|---|
| **the hypothesis this REFUTES** | The motivating hypothesis was that AP hides headroom fg-IoU cannot see. It does **not**: on monuseg the two curves move *together* (+0.018 AP alongside +0.022 fg-IoU), and on dsb2018/ctc_u373 both are flat. For the threshold lever specifically, fg-IoU was NOT a blind screen. The C16 lesson stands as a caution but did not conceal a large effect here. |
| **what is nonetheless real** | +0.022 fg-IoU on monuseg clears the project's +0.01 GO bar, needs **zero new machinery**, and the log's blanket "threshold ≤0.01" dismissal is simply wrong. Because dsb2018/ctc_u373 are flat, a single global τ≈0.65 is non-regressive on both (dsb +0.003 AP, ctc +0.001 AP) while gaining monuseg ~+0.016 AP. |
| **honest limits** | 2 seeds; τ chosen ON TEST, so this is an upper bound, not a method. A global τ=0.65 would also be exactly the kind of panel-fitted constant this project has been burned by four times. The principled version is a support-LOO-selected τ, which is cheap and untested. Controls (spheroidj, drive) were NOT swept — a semantic dataset could behave differently and must be included before adoption. |
| **scale** | +0.018 AP against a −0.199 MoNuSeg gap to Cellpose-FT ≈ 9% of the gap. Worth harvesting; nowhere near sufficient to beat a fine-tuned specialist. |
| date | 2026-07-20 |

### C34 — fg-quality → instance-AP SENSITIVITY: it is the TOPOLOGY of the foreground error, not its size

| field | value |
|---|---|
| trigger | The project had exactly TWO points on the fg→AP map (our real fg, and the C6 oracle). The fixed-harness campaign made the gap urgent: on dsb2018 a **0.026** foreground deficit vs cellpose_ft accompanies a **0.146** AP deficit, which no linear reading of "foreground is the bottleneck" explains. |
| script | `scripts/fg_sensitivity.py` (GPU1, cache `asg_cache_oracle`; the threshold sweep ran concurrently on GPU0 with a separate cache) |
| design | Degrade the GT foreground two ways to the SAME fg-IoU levels and push both through best_v2's unchanged affinity decoder. **DILATE** = grow every object (models the measured diffuse bleed; BRIDGES neighbours). **SPECKLE** = add the identical false-positive AREA as disks at random background locations kept away from objects (same fg-IoU, same over-prediction, does NOT bridge). 2 seeds, K=8, pool 20 / test 24, res 672. |

MoNuSeg:

| dilation r | dil fg-IoU | dil AP | spk fg-IoU | spk AP | AP cost of bridging |
|---|---|---|---|---|---|
| 0 | 1.000 | 0.888 | 1.000 | 0.888 | — |
| 1 | 0.861 | 0.612 | 0.863 | **0.888** | **0.276 (speckle costs ZERO)** |
| 2 | 0.754 | 0.396 | 0.764 | 0.730 | 0.334 |
| 3 | 0.659 | 0.214 | 0.680 | 0.411 | 0.197 |
| 4 | 0.594 | 0.114 | 0.628 | 0.258 | 0.144 |
| 6 | 0.488 | 0.022 | 0.557 | 0.341 | — |
| 8 | 0.421 | 0.006 | 0.537 | 0.449 | — |

| field | value |
|---|---|
| **headline** | At **equal fg-IoU**, false foreground that BRIDGES touching nuclei costs **2–3× the AP** of the same amount placed apart. At r=1, throwing away 13.7% of foreground IoU as non-bridging false positives costs **exactly zero AP** (0.888 → 0.888). Instance AP is therefore not a function of foreground *quality*; it is a function of foreground *topology*. |
| **where we actually sit** | Our real MoNuSeg point (fg-IoU 0.616, AP 0.218) lies BETWEEN the two curves — the dilation curve interpolates to ≈0.145 at that fg-IoU and the speckle curve to ≈0.26. So our errors are partly, not wholly, bridging. Making our existing errors fully non-bridging at unchanged fg-IoU is worth roughly **+0.04 AP**; that alone does not reach cellpose_ft's 0.419. |
| **why cellpose_ft wins** | cellpose_ft (fg-IoU 0.716, AP 0.419) sits essentially ON our speckle curve (0.680 → 0.411). Its foreground errors are close to non-bridging. At its foreground quality a bridging-error method interpolates to AP ≈0.32 and a non-bridging one to ≈0.53. **Both axes matter and are comparable in size**; neither alone closes the gap, together they would overshoot it. |
| **consequence for the research programme** | Every one of the "ten probes" was screened on fg-IoU (or precision/recall), a metric this table shows can move 0.14 with zero AP consequence and can hold still while AP halves. The conclusion "0.62 fg-IoU is a frozen-feature ceiling, unbreakable by any downstream mechanism" was reached with an instrument partly blind to the quantity that matters. It should be treated as unproven, not as settled. |
| honest limits | 2 seeds; dilation/speckle are models of the error, not our actual error; the interpolations above are read off a 7-point curve. dsb2018 and ctc_u373 rows were still running when this was logged — add them. |
| date | 2026-07-20 |

### C35 — FIRST diagnostic ever run on ctc_u373 (the LARGEST specialist gap) — finds an uncalibrated decoder

| field | value |
|---|---|
| trigger | ctc_u373 carries the largest absolute specialist deficit (instance AP 0.428 vs cellpose_ft 0.716 = **−0.288**), yet every fg probe C15–C30 hardcoded monuseg as the target. ctc_u373 had never been diagnosed. |
| command | `PYTHONPATH=. python -c "from scripts.monuseg_fg_diag import diag; diag('ctc_u373', 672, 0, 'head_fusion_best_cgate_film_nobank', ...)"` (cache `asg_cache_ctcdiag`) |
| result (seed 0, 19 test images) | fg-IoU **0.721** \| precision **0.775** \| recall **0.918** \| boundary-band(3px) error fraction **0.419** \| per-object detection **0.992** \| large-object recall 0.915 |
| reading | The SAME signature as monuseg: near-perfect detection (0.992), excess recall over precision (0.918 vs 0.775 ⇒ predicted area ≈ 1.18× true area), and under half the wrong pixels boundary-local ⇒ diffuse over-prediction, not missed objects and not boundary imprecision. The failure mode generalises beyond H&E. |
| **the new defect** | The fit log reports `[affinity] WARNING: <5 adjacent support pairs → merge-veto UNCALIBRATED, using conservative merge_cos=0.97` and `calibrated r*=33.5px merge_cos=0.970 (51 instances, **0 adjacent pairs**, ~6/img)`. ctc_u373 cells are large and sparse, so the K=8 support contains **zero touching pairs** — the merge-veto threshold, one of the decoder's two calibrated parameters, silently falls back to a **hard-coded 0.97** on the dataset with the biggest gap. This is exactly the "fitted constant that fails on the eleventh dataset, silently" pattern, and it has never been screened. |
| next | Sweep `merge_cos` on ctc_u373 against instance-AP to size what the uncalibrated fallback costs, and replace the fallback with a rule derived from the support's own feature statistics rather than a constant. Cheap, and it targets the largest single deficit in the table. |
| date | 2026-07-20 |

### C34b — sensitivity, dsb2018 row: on this dataset the ENTIRE specialist deficit is error topology, not foreground quality

| dilation r | dil fg-IoU | dil AP | spk fg-IoU | spk AP |
|---|---|---|---|---|
| 0 | 1.000 | 0.782 | 1.000 | 0.782 |
| 1 | 0.835 | 0.532 | 0.836 | **0.782** |
| 2 | 0.718 | 0.324 | 0.723 | **0.725** |
| 3 | 0.615 | 0.189 | 0.626 | **0.483** |
| 4 | 0.548 | 0.131 | 0.567 | 0.188 |

| field | value |
|---|---|
| **the resolution of the dsb2018 puzzle** | Our real point is fg-IoU 0.846 / AP 0.502, which sits essentially ON the dilation curve (0.835 → 0.532). The speckle curve at the SAME foreground quality gives AP **0.782** — i.e. identical to the perfect-foreground oracle. So at dsb2018's foreground quality, non-bridging false positives cost **nothing at all**, and effectively **100% of our AP deficit there is the topology of our errors**, not their size. That is why a 0.026 foreground deficit vs cellpose_ft accompanies a 0.146 AP deficit: foreground quality was never the mechanism on this dataset. |
| **size of the prize** | If our dsb2018 errors were non-bridging at UNCHANGED foreground quality, AP would go 0.502 → ≈0.78 — above cellpose_ft's 0.647, above stardist_ft's 0.578, above microsam_ft's 0.638. **dsb2018 is winnable against fine-tuned specialists without improving the foreground at all.** |
| **contrast with monuseg** | On monuseg the foreground is genuinely too poor (0.62) to exploit this: the speckle curve there gives only ≈0.26 at our fg-IoU, so de-bridging alone buys ≈+0.04 and BOTH axes are needed. The two datasets therefore need different fixes, and the project's single "foreground is the bottleneck" framing is true for monuseg and false for dsb2018. |
| **implication for the decoder** | The affinity-watershed decoder already exists to split touching instances, and it handles speckle noise perfectly (0.782 at fg-IoU 0.836). It is specifically failing to split the bridges our OWN foreground produces. That makes de-bridging a decoder/feature problem we can attack directly, not only a loss problem. |
| honest limits | 2 seeds; dilation and speckle are models of the error, not the error itself. ctc_u373 row still running. |
| date | 2026-07-20 |

### C33b — threshold sweep, CONTROL datasets: a single global tau FAILS the control gate

| dataset | metric | tau=0.30 | tau=0.50 (current) | tau=0.65 | tau=0.80 |
|---|---|---|---|---|---|
| spheroidj | fg-IoU | 0.883 | 0.888 | 0.890 | 0.891 |
| drive | fg-IoU | 0.581 | 0.577 | 0.567 | 0.551 |

| field | value |
|---|---|
| verdict | spheroidj is flat-to-slightly-positive in tau, but **drive falls monotonically** (0.577 → 0.567 at tau=0.65, −0.010), which is twice the −0.005 control-regression threshold. So the "global tau≈0.65 is free" reading of C33 is **WRONG** — a fixed global threshold trades monuseg's +0.018 AP for a real vessel regression. The threshold must be support-derived (or morphology-gated), exactly as the project's no-fitted-constants rule requires. |
| note | The instance decoder is inactive on these binary-GT datasets (`_inst_r is None`), so only fg-IoU is meaningful here; the script now guards this instead of crashing (the first attempt died on `int(round(None))`). |
| date | 2026-07-20 |

### C36 — `bankselect` lever (Fisher-select classical bank channels from support) — fast-screen → NO-GO / DROP

| field | value |
|---|---|
| hypothesis | Hard-discard the classical HyperBank channels whose support fg-vs-bg Fisher separability is low (keep top `bankselect_ratio=0.3`), on the theory that pruning uninformative bank channels denoises the fusion input. A simplification lever (bar = DON'T HURT). |
| method / token | `head_fusion_best_cgate_film_nobank_bankselect` (al_testbed token `bankselect`; `_bank_separability` + `_bank_keep` in `head_fusion_backend.py`, default-off). |
| screen | 2 TARGET (drive vessels, monuseg dense H&E) + 2 CONTROL (spheroidj semantic blob, dsb2018 instance); 3 seeds; K=8; pool 20; test all; res 672. **SEPARATE `cache_bankselect`** (no race with the concurrent regen on `cache_final10`); jobs SEQUENTIAL. Launcher `scripts/run_bankselect_screen.py`, score_dir `results/bankselect_screen/`, GPU0. |
| results (3 seeds) | drive 0.694→**0.688** (−0.006) · monuseg 0.622→**0.609** (−0.013) · spheroidj 0.900→**0.892** (−0.008, CONTROL regression) · dsb2018 0.853→**0.849** (−0.004). NO target gain; regresses all four. |
| verdict | **NO-GO / DROP.** The lever runs correctly (keeps 16–22/35 channels) but hard univariate Fisher filtering discards JOINTLY-useful bank channels; the 1×1 head's soft per-channel weighting is strictly better than a hard discard (same reason it would hurt DINO dims). Lever left in, default-off, as a recorded negative. Do NOT re-try univariate feature-selection on this pipeline without new evidence. |
| review | code + independent code-review clean (5 findings fixed pre-launch); `_bank_keep` reset added to `reset_support_state` (seed-leak fix). |
| date | 2026-07-21 |

### C37 — `scaleconf` lever (cluster support DT radii → set N bank scales = N centroids) — fast-screen → NO-GO as a global lever / DROP

| field | value |
|---|---|
| hypothesis | Self-configure the classical bank's filter scales from the support: medial-axis distance-transform radii → 1-D log-KMeans (`cluster_scales`, silhouette>0.55, merge <1.4×, clamp [1.5,64]px) → set Frangi σ / Sauvola window / LoG σ to the N cluster centroids, replacing the fixed (1,2,4,8,16). Bar = IMPROVE a target (Δ>~+0.01) with NO control regression (Δ>~−0.005) — adds capability (right-scaled filters), not just simplification. |
| method / token | `head_fusion_best_cgate_film_nobank_scaleconf` (al_testbed token `scaleconf`; `cluster_scales` + `_support_scales` + `_bank_module` scaleconf branch in `head_fusion_backend.py`, default-off). |
| screen | 3 TARGET (drive vessels, monuseg dense nuclei, bacteria thin rods) + 2 CONTROL (spheroidj blob, dsb2018 instance); 3 seeds; K=8; pool 20; test all; res 672. SEPARATE `cache_bankselect` (shared DINO feats, no race); INTERLEAVED per-dataset, bacteria last. Launcher `scripts/run_scaleconf_screen.py`, score_dir `results/scaleconf_screen/`, GPU1. |
| results (3 seeds) | drive 0.692±0.004→**0.705±0.003** (**+0.013** ✓ TARGET) · monuseg 0.620±0.008→**0.600±0.019** (−0.020 ✗ TARGET regresses, 2× variance) · spheroidj 0.897±0.013→**0.899±0.024** (+0.002 ~) · dsb2018 0.853±0.006→**0.846±0.008** (−0.007 ✗ CONTROL regression) · bacteria 0.903±0.005→**0.903±0.003** (0.000 ~, near-ceiling, no room). |
| scale diagnostic | drive built **3 scale clusters [1.5, 2.1, 3.3]px** (matched vessel widths, dropped the useless fixed 8/16px), stable across seeds (~70k radii/seed). Confirms the mechanism works; the problem is morphology, not implementation. |
| verdict | **NO-GO as a global lever / DROP.** scaleconf helps essentially ONLY drive: Frangi tuned to object radii helps ridge-like vessels (drive +0.013) but bacteria (also thin) was neutral (near-ceiling), and on blobs/nuclei Frangi *suppresses* the interior, so tuning to blob radii wastes the bank AND drops the broad default coverage → regression (monuseg −0.020, dsb2018 −0.007). Not even broadly tubular — drive-specific. Violates the control-regression bar. NOT folded into best_v2. A morphology-gated variant (tune only when support is tubular) was considered and rejected — narrow gain (drive-only +0.013, already strong in Table 1) not worth the added complexity + re-screen. Related risk = size-shift sensitivity (support-coarse → test-fine drops the fine scales the fixed bank always keeps); see the size-robustness note. |
| review | code + independent code-review clean; `cluster_scales` sub-pixel-filament bug fixed (filter `x>=0.3`, rely on clamp); `_bank_scales` reset added to `reset_support_state`. |
| date | 2026-07-22 |

### C40 — nnU-Net as an annotation-efficiency rival (K=8): epoch budget set from its own convergence

| field | value |
|---|---|
| trigger | The paper brands itself "self-configuring", nnU-Net's term. Citing it answers the positioning objection with an argument; training it on the SAME support masks answers with evidence. |
| method | `scripts/nnunet_bench.py`. nnU-Net v2.8.1 (`~/nnunet_env`), 2D config, `-f all`, binary-foreground target. Same support draw as sota_final (`np.random.default_rng(seed).choice(len(pool), K, replace=False)`), same fixed test split, same metric, records via `write_score_record` so the arms are paired-comparable. |
| screen | monuseg, drive (hard) + spheroidj, dsb2018 (control), K=8, 3 seeds, two parallel streams on the A100. |
| epoch budget | **100** (`nnUNetTrainer_100epochs`, one of nnU-Net's own documented variants). Chosen from ITS OWN convergence, not for convenience: on monuseg the pseudo-Dice trajectory is 0.783 (ep 1), 0.955 (21), 0.981 (61), 0.988 (121), and then oscillates — the last 60 epochs bought 0.007 and the final value was BELOW the peak. nnU-Net defines an epoch as a fixed 250 iterations regardless of dataset size, so 100 epochs is 25,000 iterations over 8 images, roughly 3,100 passes per image. Evidence retained at `campaign_logs/nnunet_convergence/`. |
| first attempt | 250 epochs, killed after ~2 h. Two reasons, both worth recording: the convergence above showed the extra budget buys nothing, and epoch time drifted 31 s → 96 s once two trainings and the stem screens shared the A100, which would have made the screen 40+ h rather than 13 h. Contention, not the model, was the dominant cost. |
| fairness notes | No instance labels (nnU-Net is a semantic segmenter and the paper's metric is semantic; asking it for instances would handicap it on a task neither method is scored on). No 5-fold CV (a five-way split of eight images is not validation). `nnUNet_compile=f` because this host's GLIBC is older than the triton kernel wants — eager mode changes no result but its absence surfaced only as "background workers are no longer alive". |
| status | **COMPLETE 2026-07-26**, 11/11 datasets, 10 seeds. See the results block below. |
| date | 2026-07-22 |

### C41 — ECA channel attention on the frozen backbone grid — NEGATIVE, and the reason generalises

| field | value |
|---|---|
| trigger | "What is the optimal way to combine 1024 backbone channels?" C36 had already ruled out hard Fisher selection (regressed 4/4; channels are jointly useful), leaving soft weighting. ECA (Wang et al., CVPR 2020) is soft, needs no bottleneck, and costs k=5 parameters where a Squeeze-and-Excitation block on 1024 channels costs 131,072. |
| method | `ECA` in `head_fusion.py`, applied to the raw DINOv3 grid before the stem's channel reduction. Token `_eca`. Parity at init (zero-init weights, 2*sigmoid gate, verified `max|ECA(x)-x| = 0`). |
| screen | monuseg, drive, spheroidj, dsb2018, K=8, 3 seeds, vs the reported head. |
| results | `wide+eca`: monuseg **-0.010**, drive **-0.007**. `flat+eca` vs `flat` alone: drive 0.7269 vs 0.7291, spheroidj +0.016 vs +0.013, dsb2018 +0.004 vs -0.004 — inside noise either way. |
| verdict | **NO-GO / DROP.** Not a tuning failure, a domain mismatch. |
| WHY, measured | ECA weights each channel from a 1-D convolution over its INDEX NEIGHBOURS, so its entire inductive bias assumes adjacent channel indices are related. On cached DINOv3 features (C=1024, 1764 positions) the mean absolute correlation between adjacent channels is **0.1685** against **0.1714** for pairs at least 5 apart — a ratio of **0.98**. Channel index order in a frozen transformer is arbitrary, so that convolution mixes unrelated quantities. This is a property of ViT embeddings (permutation-invariant dimensions), not of this dataset or this k. |
| do not re-try | ECA with a different kernel size, or any channel operator whose locality is over the channel INDEX, on frozen transformer features. A permutation-invariant operator (full SE, 131k parameters) would not have this defect — but at K=8 that is a poor trade for an uncertain gain, and the 1x1 already supplies soft per-channel weighting. |
| date | 2026-07-22 |

### C42 — Head parameter budget: the lean-stem ladder and its confound control

| field | value |
|---|---|
| trigger | "Can the head be made more parameter-efficient?" 95% of its 3,103,287 parameters sit in two stem convolutions and 76% in one: Conv3x3(1024->256) = 2,359,552, which pays the 9x kernel multiplier on the widest channel transition in the network. |
| method | `stem` = wide (3x3,3x3) / lean (1x1,3x3) / flat (1x1,1x1), with `hidden`, `depth`, `hidden_film` as width/depth knobs. Tokens `_lean _flat _h<N> _d<N> _f<N>`. Default unchanged and asserted so. |
| screen | monuseg, drive (hard) + spheroidj, dsb2018 (control), K=8, 3 seeds, vs the reported head. Gate: no dataset below -0.005. |
| results | flat/128 (296,759, -90%) mean **+0.015** worst **+0.0005** PASS · flat/256 (481,847) +0.018 / -0.0035 PASS · lean/256 (1,006,135) +0.013 / +0.0011 PASS · lean/128 (427,831) +0.007 / **-0.019** FAIL · flat h64 d1 f16 (104,359) +0.002 / -0.008 FAIL |
| CONFOUND CONTROL | The head trains a FIXED 60 full-batch steps with no validation split, and the logged loss is still descending at 60 and at 120 — so "a smaller head converges faster at a fixed budget" was a live alternative explanation with the opposite conclusion. `wide ep120` (2x budget, architecture unchanged) recovers most of the hard-dataset gain (drive +0.036 vs flat's +0.039) but regresses the spheroidj control -0.006 and **FAILS the gate**. The effect is therefore capacity, not optimisation: the pure-optimisation arm fails exactly where the architectural arm passes. |
| mechanism | Removing the SPATIAL kernel beats narrowing: flat/256 (482k, no 3x3) outscores lean/128 (428k, keeps a 3x3) despite MORE parameters. The head's own 3x3 re-derives locality that arrives twice for free — DINOv3 attention mixes globally before it, and the native classical priors re-introduce structure after the upsampler. |
| side finding | No fixed epoch count can be right: 60 under-trains dense nuclei and vessels, 120 over-trains spheroids. Motivates `es` (training-loss plateau stop) plus explicit regularisation, since with weight_decay=0 and no dropout the short budget WAS the only regulariser. |
| status | 3 seeds, directional. Full panel + 10 seeds owed before the default changes; adopting it invalidates every number in the ISBI paper. |
| date | 2026-07-22 |

### C43 — Matcher anchor reproduction (FSS-1000 one-shot) — the other half of the fairness claim

| field | value |
|---|---|
| trigger | Matcher became a Table 1 column, and the paper asserted baseline fairness while naming only PerSAM's anchor as reproduced. A reviewer is entitled to ask whether our Matcher runs the configuration its authors published. |
| method | `scripts/fss.sh` from the Matcher repo, UNMODIFIED, fold 0, on GPU1. FSS-1000 downloaded from the authors' Google Drive link (678 MB, 1000 classes) into the layout `datasets/README.md` specifies. |
| **result** | **fold 0 mIoU 87.15, FB-IoU 91.59** against Matcher's published ~87.0. Reproduced. |
| four obstacles, each solved the least invasive way | (1) `tensorboardX` missing — dry-run confirmed purely additive (protobuf was absent, so no version conflict in the environment running the campaign). (2) `fss.sh` calls a bare `python` — fixed by putting the env on PATH rather than editing their script. (3) `detectron2` is imported eagerly by the LVIS and PACO loaders, which FSS never uses; building it against torch 2.5.1 inside the campaign environment risks dragging torch and breaking the campaign, so an import-only stub was placed on PYTHONPATH whose every symbol RAISES — the run completing proves the stub was never touched. (4) `np.int` was removed in numpy 2; one line on the FSS path (`matcher/Matcher.py:368`), replaced with `int`, which numpy's own error text documents as behaviour-preserving. Original kept at `Matcher.py.orig`, diff is one line. |
| verdict | Matcher's configuration in this repo is the published one. Both SAM-based baselines are now anchor-verified. |
| date | 2026-07-23 |

### C44 — Training-free correspondence as a REPLACEMENT for the head — NO-GO, with a useful side finding

| field | value |
|---|---|
| trigger | Goal of making the method genuinely in-context. Proposal: drop stem/gate/FiLM entirely, extract features for support and query, and segment by correspondence as INSID3 does. |
| method | `scripts/corr_fused_prescreen.py`, training-free, leave-one-support-out, K=8, 2 seeds. Prototype discriminant (p_fg − p_bg)·x per pixel in three spaces: DINOv3 only (INSID3-like), the 35 classical channels only, and both with each block L2-normalised. Threshold chosen ORACLE-optimally per image, so these are UPPER bounds. |
| results (oracle-threshold fg-IoU) | monuseg dino 0.404 / bank **0.507** / fused 0.513 · drive 0.262 / 0.376 / 0.422 · spheroidj **0.815** / 0.364 / **0.512** · dsb2018 0.776 / 0.782 / 0.807. Trained head for reference: monuseg 0.628, spheroidj 0.902, dsb2018 0.846. |
| verdict | **NO-GO.** Loses 0.11 to 0.39 against the trained head WITH an oracle threshold a real method could not use. |
| mechanism | Equal-weight fusion is worse than the better single block wherever the blocks disagree: on spheroidj DINO alone reaches 0.815 and fusing drags it to 0.512, because the bank is uninformative there (0.364) and carries equal weight anyway. Learning that per-dataset block weighting is exactly what the trained 1x1 does. Even an ORACLE choice of the best single block per dataset (0.507 / 0.815 / 0.782) stays well below the trained head, so fixing the weighting closed-form would not rescue it. |
| side finding, worth keeping | On the two hardest datasets the CLASSICAL BANK ALONE beats DINOv3 correspondence: monuseg 0.507 vs 0.404, drive 0.376 vs 0.262. Hand-designed native-resolution priors carry more mask-relevant signal than the semantic features on dense nuclei and thin vessels — independent support for the paper's claim about the bank, measured without any training. |
| consistent with | C15 (clustered correspondence strictly worse than single-prototype; bottleneck is downstream of correspondence) and C26 (kNN 0.325 vs MLP 0.396 on raw features). |
| leaves open | The meta-training route (train the head ONCE across datasets, support enters via FiLM and the corr channel at forward time). That is a different question and `scripts/lodo_head.py` measures its prerequisite. |
| date | 2026-07-23 |

### C45 — Warm-start refits for interactive annotation: 3-4x faster at equal accuracy

| field | value |
|---|---|
| trigger | The deployed tool refits after every mask the user draws, so the number that decides usability is one step's latency. Measured cold fit at K=8: 149 s on MoNuSeg (1000 px), 38 s on DRIVE (584 px). Two minutes per mask is not an interactive tool. |
| method | `fit(support, warm=True, epochs=N)` keeps the previous head's weights and takes N steps from there. Prototypes ARE re-derived (a new mask changes what foreground looks like, and FiLM and the correspondence channel read them); the colour choice is NOT (re-selecting it mid-session would invalidate every cached classical map and make the output jump for a reason invisible to the user). |
| protocol | `scripts/hitl_bench.py` simulates the loop rather than timing a fit in isolation: masks added one at a time K=1..8, both policies refit and score the same fixed test set after each addition. Latency AND accuracy are reported, because a faster refit is only worth having at equal quality -- on latency alone "do nothing" wins. |
| results (MoNuSeg, flat/128, 10 warm steps) | K=2 58.2s/0.6128 vs **18.0s/0.5895** · K=4 107.3/0.6248 vs **24.9/0.6201** · K=5 144.7/0.6250 vs **35.8/0.6264** · K=6 129.5/0.6205 vs **43.8/0.6378** · K=7 199.2/0.6299 vs **49.8/0.6336**. Speedup 3.0-4.3x, accuracy equal from K=3 and BETTER at K=6 and K=7. |
| reading | Ten steps from the previous head beat sixty from random init. The previous support differed by one image, so random init discards a nearly correct answer and spends the budget recovering it. |
| what still binds | Warm latency grows with K (18s -> 50s) because it runs N epochs over ALL k images. At 1000 px that is ~0.7 s per image-step, and the cost is the ACTIVATION at native resolution (1000x1000x99 floats per forward), not the 300k parameters. The remaining lever is therefore fit AREA, not head size -- consistent with DRIVE (584 px) fitting 4x faster than MoNuSeg (2.9x the area). |
| supporting timings | cold encode 0.15-0.29 s per image (this was previously misreported as 0.006 s, which was a .npy cache read); predict 0.39-1.23 s per image. In a HITL loop the pool is encoded once, so the per-interaction cost is fit + predict. |
| DRIVE (584 px), 10 warm steps | K=2 12.2s/0.7118 vs **2.6s/0.7296** · K=4 12.0/0.7299 vs **2.6/0.7439** · K=6 26.9/0.7259 vs **6.9/0.7503** · K=8 48.3/0.7390 vs **9.0/0.7473**. Warm is faster AND more accurate at every single K. |
| MoNuSeg with 3 warm steps | K=2 30.8s vs **12.3s** (0.6276 vs 0.6241) · K=4 54.8 vs **18.1** (0.6110 vs 0.6137). Three steps hold the accuracy ten steps reached. |
| why warm is MORE accurate | Not noise: the HITL loop accumulates optimisation. By K=8 the warm head has had 60 + 7x10 = 130 steps against the cold fit's 60 from scratch, and C42 established that 60 steps UNDER-trains this head. The changing support supplies the regularisation a longer budget otherwise needs, so the interactive loop happens to fix both problems at once. |
| date | 2026-07-24 |

### C46 — z-norm (nnU-Net's data-driven input normalisation) on the new best — NO-GO, and why

| field | value |
|---|---|
| trigger | nnU-Net beats us where applicable, and its own ablations credit data-driven intensity normalisation as the single largest lever. Tested the support-only analogue on the new best config. |
| method | `znorm` token on `flat_es_ep500_wd4_do5_mix`: subtract the median support-foreground intensity, divide by its IQR, on the chosen channel, AFTER CLAHE (order verified — a z-scored channel breaks CLAHE's [0,1] precondition and silently disables it). 3 seeds. |
| results | monuseg 0.6390→**0.6361** (−0.003), drive 0.6423→**0.6420** (−0.000), spheroidj 0.9170→**0.9162** (−0.001). Screened on the two datasets nnU-Net beats us on most + a control. |
| verdict | **NO-GO / DROP.** Mildly negative everywhere, worst on monuseg. |
| why it does not transfer | We already have the invariance nnU-Net gets from z-score, by other means: support-driven colour-channel selection, per-channel bank normalisation (C-banknorm), and adaptive CLAHE. z-score is nnU-Net's ONLY input normalisation; here it is a fourth layer on an already-normalised input and only adds noise. This is the useful finding: our self-configuration already covers the lever nnU-Net's ablations credit, from a different direction — a direct answer to "why no intensity normalisation like nnU-Net". Kept default-off behind the `znorm` token. |
| date | 2026-07-24 |

### C47 — Leave-one-out ablation of best_v3: the architectural novelties went to zero

| field | value |
|---|---|
| trigger | CLAUDE.md requires a fold-in to be re-validated as a whole. `best_v3` (C42 stem + regularisation + plateau stop, promoted by `final_winner.py`) had never had its components ablated in the NEW configuration; Table 2 of the paper still reports the ablation of the OLD one. |
| method | `ablate_v3.sh`. Base `head_fusion_best_cgate_film_nobank_flat_es_ep500_wd4_do5_mix`; each arm removes exactly one component via its token. Screen protocol: monuseg, drive (hard) + spheroidj, dsb2018 (control), K=8, pool 20, `--test 10000`, `--metric_override fg_iou`, res 672, **3 seeds**, cache `/disk1/prusek/cache_final10`, score_dir `results/ablate_v3/<arm>`. Reference = `results/winner_panel/flat_es_ep500_wd4_do5_mix`, first 3 seeds, seed-matched. |
| launcher | `bash /disk1/prusek/active-segmenter/ablate_v3.sh` (9 arms x 4 datasets = 36 cells, completed 2026-07-25 20:11, 0 errors) |
| results (mean over the 4 screen datasets, delta vs best_v3 0.7630) | −classical bank **−0.1192** (drive 0.640→0.352, monuseg 0.642→0.510) · −regularisation **−0.0200** (dsb2018 0.854→0.796) · −flat stem, i.e. back to wide 3x3 **−0.0145** · −plateau stop (fixed 60 ep) **−0.0116** · −colour selection **−0.0114** · −both (fixed 60 + no reg) **−0.0073** · −adaptive loss **−0.0033** · −FiLM **−0.0025** · −competitive gate **−0.0023** |
| **finding 1 — the novelties are now noise** | `cgate` and `FiLM` contribute −0.0023 and −0.0025; on drive the method is marginally BETTER without the gate (0.6413 vs 0.6403). In the OLD configuration the paper measured the gate at +0.016 on vessels (10 seeds, p=0.02). Regularisation and the plateau stop appear to supply what the gate and FiLM used to. Both components are prominent in the paper's pipeline figure and Method section, so adopting best_v3 forces that narrative to be rewritten, not just the numbers. |
| **finding 2 — the prior-work bank dominates** | The classical bank is worth −0.1192, an order of magnitude more than everything else combined; the whole closed-form self-configuration is ~−0.015. The paper already concedes the bank is prior work; in best_v3 the ratio is starker. |
| **finding 3 — training budget and regularisation are NON-ADDITIVE** | Removing BOTH regularisation and the plateau stop (−0.0073) hurts LESS than removing either alone (−0.0200, −0.0116). Long training without regularisation overfits; a short budget regularises by itself. So (long + regularised) and (short + unregularised) are both viable and (long + unregularised) is the bad corner. Confirms the sign-flip recorded in memory `lean-head-goal`, and is exactly why CLAUDE.md requires the fold-in to be re-validated as a whole rather than inferred from the parts. |
| status | **SCREEN ONLY — 3 seeds, 4 datasets, directional.** Every arm that moved needs the full panel at 10 seeds before Table 2 is rewritten. `cgate`/`FiLM` may still contribute on the thin-structure datasets absent from the screen (hrf, fisbe, isbi2012em, bbbc010), which is where their original justification came from. |
| date | 2026-07-25 |

### C48 — Cold-fit cost of the plateau schedule: es_ep500 is 4x slower, and the user chose to keep it

| field | value |
|---|---|
| trigger | Positioning against nnU-Net moved to the cost axis ("when we are not more accurate, we are cheaper"), so the fit cost had to be measured rather than asserted. |
| method | `scripts/timing_bench.py`, logged to `campaign_logs/timing_all.log`; cold fit, encode and predict per image, on monuseg (1000 px) and drive (584 px). |
| results (fit / encode / predict, seconds) | `head_fusion_best_cgate_film_nobank` (wide, 60 ep): monuseg **148.76** / 0.150 / 0.841 · drive **37.52** / 0.279 / 0.532. `_flat_h128` (60 ep): monuseg **189.44** · drive **28.51**. `_flat_h128_es_ep500_wd4_do5`: monuseg **601.66** / 0.219 / 0.885 · drive **153.36** / 0.142 / 0.245. |
| finding | The plateau early stop makes the cold fit **~4x more expensive** (148.8 s → 601.7 s on monuseg). Against nnU-Net's ~3100 s per support draw (100 epochs x 250 iterations at ~31 s/epoch uncontended, C40) the cold-fit advantage is only **~5x**, NOT the 100-1000x that quoting C45's old-configuration number would suggest. |
| **paper defect this exposes** | The Implementation section claims "Fitting one support set takes under a minute on a single graphics processing unit". That is false for the CURRENTLY reported configuration (148.8 s on monuseg) and would be false by 10x for best_v3. Corrected in `main.tex` 2026-07-25. |
| decision | **User decided 2026-07-25: do NOT shorten the schedule.** One configuration, accuracy first. The cost argument therefore has to rest on the WARM refit (nnU-Net has no warm start and retrains 25,000 iterations per added mask), which still needs measuring under the plateau stop, and on the K→1 regime. |
| caveat for any future timing | The measured config is `flat_h128` (candidate c2), not the promoted c1 `flat/256+mix`, which has ~1.6x the parameters plus a mix block — so 601 s is a LOWER bound for best_v3. And nnU-Net must be timed on a FREE GPU: a contended epoch drifts 31→96 s, and timing it under load against our uncontended fit would inflate our advantage ~3x, an error in our own favour. |
| date | 2026-07-25 |

### C49 — ilastik-style random-forest pixel classifier: the missing biologist-workflow baseline

| field | value |
|---|---|
| trigger | External review, 2026-07-25: ilastik/Labkit are what biologists actually use for this exact workflow (paint a few labels, fit a random forest over a filter bank, classify the dataset) and the paper neither compared against them nor cited them. Two risks at once: a missing baseline an ISBI microscopy reviewer will ask for, and a novelty framing risk ("this is ilastik plus DINOv3") we could not answer with a number. |
| method | `scripts/ilastik_bench.py`, new. NOT the ilastik software: a `RandomForestClassifier` (100 trees, `min_samples_leaf=4`) over **our own** classical prior bank, i.e. `HyperBank(frangi_sigmas=(1,2,4,8,16), sauvola_windows=(15,51,151), struct_sigmas=(2,8), use_log=True)`, byte-identical to the composition `head_fusion_backend._bank_module()` builds. Reported as `ilastik_rf`, never as "ilastik". |
| deliberately a STEELMAN | The bank is the one tuned for these morphologies across this project, so this classifier starts from BETTER features than an off-the-shelf pixel classifier. It also gets the raw colour channels as extra features on colour data (38 features vs 35 on monochrome), because that is what an ilastik user actually has. Any paper claim must say this: beating it is a statement about the classifier and the frozen semantics on top, NOT about having better filters. |
| deliberately withheld | All support-derived self-configuration (colour-channel selection, adaptive CLAHE, morphology-driven loss). Those are the paper's contribution; giving them to the baseline would make it a hybrid that answers no question. |
| protocol | Identical to `nnunet_bench.py` and the campaign: same `load_dataset(spec, pool_for(ds), 10000, seed=0)`, same `default_rng(seed).choice(len(pool), 8)` support draw, same per-dataset semantic metric, records via `write_score_record` so `stats()` reads this arm with the rest. K=8, 10 seeds, 11 datasets, `--max_side 1536` (the head's own training cap), 20k balanced pixels sampled per support image, probability upsampled bilinearly to native then thresholded at 0.5 (the generous choice for a baseline computed below native res). |
| launcher | `bash /disk1/prusek/active-segmenter/ilastik_panel.sh` — two streams on GPU1, `--n_jobs 10` each, score_dir `results/ilastik_k8`, logs `campaign_logs/ilastik_{a,b}.log` |
| verification before launch | Smoke test on monuseg (1 seed, 3 test images) caught a real defect: the script unpacked `pool` entries as objects when `load_dataset` yields `(image, label)` tuples. Guards then MUTATION-TESTED (`/tmp/mutate_ilastik_guards.py`): non-finite features, single-class support pixels, and a mid-run feature-width change each fired with the right message when the defect was injected, and a healthy control did not fire. NOTE: the CLAUDE.md independent-agent code review was NOT performed for this script (the user asked for solo execution); a `/code-review` pass is still owed. |
| results | RUNNING (launched 2026-07-25 ~20:50). Early full-split numbers: monuseg 0.503/0.489/0.533 (seeds 0-2) against ours 0.628 reported and 0.639 for best_v3; dsb2018 0.798-0.842 against ours 0.846 and nnU-Net 0.851. |
| early reading, to confirm at 10 seeds | The classical bank plus a forest is already near the ceiling on easy blobs (dsb2018 within ~0.01 of everything) and clearly behind on dense stained nuclei (monuseg ~0.51 vs 0.64). If that holds, the honest claim is that the frozen foundation semantics buy little where classical features already separate the objects, and buy a lot where they do not — which is a sharper statement than a flat win and is exactly what the ablation's prior-bank row (C47, −0.119) predicts. |
| date | 2026-07-25 |

### C50 — Iris / Show-and-Segment cannot be run on this panel, verified

| field | value |
|---|---|
| trigger | `\cite{showseg}` was cited in the Introduction as the recent extension of in-context segmentation and never compared against — a standard reviewer complaint. |
| verified from the paper | Iris (Gao et al., CVPR 2025, arXiv:2503.19359) trains and infers on **128x128x128 volumes with a 3D UNet**, over 19 datasets that are all volumetric CT, MR and PET (AMOS CT/MR, AutoPET, BCV, Brain, CHAOS, KiTS, LiTS, M&Ms, StructSeg, CSI, ACDC, SegTHOR, MSD Pancreas, Pelvic). **No 2D microscopy of any kind.** |
| verdict | Not runnable on this panel by construction, not by omission. One clause added to the Introduction stating the reason, which is a stronger answer than a missing row. |
| date | 2026-07-25 |

#### C50 addendum — re-read from the CVPR PDF, 2026-07-27: verdict CONFIRMED, and three numbers worth taking

| field | value |
|---|---|
| trigger | User supplied the CVPR 2025 open-access PDF and asked whether Iris should after all be a benchmark row. Read in full rather than trusting the earlier abstract-level check. |
| **primary evidence for the verdict** | Implementation Details, verbatim: *"Iris uses a 3D UNet encoder trained from scratch with one-shot learning strategy... Training and inference use 128 x 128 x 128 volume size."* Section 3.2.1 takes the reference as `(x_s, y_s) in R^{DxHxW} x {0,1}^{DxHxW}`, PixelShuffle operates on volumes, and eq. 6 emits `y_q in {0,1}^{KxDxHxW}`. Twelve training datasets (AMOS CT/MR, AutoPET, BCV, Brain, CHAOS, KiTS, LiTS, MnM, StructSeg H&N/Tho, CSI-Wat) and seven held-out (ACDC, SegTHOR, 3x IVDM3Seg, MSD Pancreas, Pelvic1K) are all CT/MR/PET volumes. No 2D microscopy anywhere. |
| **the paper's own argument for why it does not transfer** | *"a 2D-slice-based architecture (e.g. UniverSeg and Tyche) limits its capability on 3D tasks like SegTHOR. In contrast, Iris's task encoding module efficiently extracts and utilizes 3D domain-specific information from the reference examples."* Its margin over the 2D in-context methods IS the 3D encoding. On 2D microscopy that advantage is not merely absent, it is undefined -- there is no depth axis. Running it would require writing a 2D reimplementation, i.e. a self-authored baseline, which the baseline-fairness protocol exists to avoid. |
| **TAKE 1 -- Figure 6 (left) bears directly on the L3 fallback in the closed-form plan** | Performance on unseen classes against NUMBER OF TRAINING TASKS saturates only around **60-80 tasks**, with the stated conclusion that *"exposure to diverse anatomical patterns is necessary towards more robust and transferable feature learning"*. Iris had 12 datasets decomposed into dozens of per-class tasks. Meta-training a stem on this project's SIX training datasets is far outside that regime. This is external evidence against the L3 fallback and for the synthetic-prior route in the PFN direction: if the required task diversity is tens to hundreds, generating tasks is the only affordable source. |
| **TAKE 2 -- novel-class collapse, citable** | Table 2, MSD Pancreas Tumor: Iris **28.28** against supervised nnU-Net **54.56** -- roughly half, on a novel class WITHIN its own trained modality (best competing adaptive method 11.97). Pelvic 69.03 vs 94.73. Supports our own limitation claim with someone else's numbers. |
| **TAKE 3 -- decoupled task encoding corroborates the closed-form design** | Table 3, one A100, 10 query images x 15 classes at 128^3: UniverSeg-1 **659.4 s**, UniverSeg-128 1030.2 s, SAM-Med2D 648.4 s, SAM-Med3D 15.2 s, **Iris 2.0 s**; 7.4 GB, 69.4M params. Complexity `O(k+m)` against `O(kmn)`, because the task embedding is computed ONCE from the reference and reused across every query. Same structure as solving `w` once from the support and applying it densely -- different mathematics, identical shape of argument. |
| other numbers worth having | Table 1 in-distribution avg Dice: nnU-Net 83.18, Clip-driven 84.18, UniSeg 84.40, Multi-Talent 84.47, SAM 17.97, SAM-Med2D 40.58, SAM-Med3D 68.42, SegGPT 57.35, UniverSeg 58.68, Tyche-IS 61.20, **Iris 84.52**. Ablation Table 4: high-resolution processing lifts SMALL structures 62.13 -> 78.92, which is the same small-object-resolution axis this project keeps hitting. |
| decision | **Still no benchmark row.** Upgrade the Introduction clause to state the architectural reason (3D throughout, 128^3) and cite TAKE 2 in the limitations discussion. TAKE 1 goes in the design record for the in-context work, not the paper. |
| date | 2026-07-27 |

#### C49 addendum — the width guard fired in production, and what it caught

| field | value |
|---|---|
| event | 2026-07-25 ~21:40, stream A died on **spheroidj** after completing monuseg and drive: `feature width changed mid-run: 35 != 38`. |
| cause | SpheroidJ MIXES colour and grayscale images in one dataset. The colour planes were appended PER IMAGE, so a monochrome image produced 35 features where the support fit had produced 38. Without the guard the run would have trained a forest on 38 columns and predicted with 35 — sklearn would have raised much later, or worse, a mismatched-but-same-width case would have scored silently. |
| fix | Decide the modality **ONCE per dataset from the support**, mirroring the method's own `_choose_contrast_source` ("decide MODALITY per-dataset, not per-image"): `BankFeatures.decide_mode(support_images)` sets colour mode if a majority of support images are colour, and a monochrome image in a colour-mode dataset contributes its grayscale replicated three times. Width is then constant by construction, and the baseline is not understated on either kind of data. |
| verification | Mutation tests re-run: G1 non-finite, G2 single-class and G3 width-change all still fire; a NEW **G4 regression test** builds the exact SpheroidJ shape (majority-colour support, then a monochrome image) and asserts both yield 38 features. Smoke on spheroidj: `colour mode: ON (7/8 support images are colour)`, runs clean. |
| lesson | This is the "fitted constant / per-image assumption that fails on the eleventh dataset" pattern again, and the third time in this project it surfaced only because a guard was written to fail loud rather than to cope. Note the guard was written BEFORE the defect was known — it was guarding a hypothetical that turned out to be real two datasets later. |
| partial results at the time of the crash | monuseg 0.503-0.533, dsb2018 0.798-0.842, drive cldice 0.610-0.641 (10 seeds), spheroidj ~0.50-0.58 early. |
| date | 2026-07-25 |

#### C49 results — COMPLETE, 11/11, and the early reading was WRONG

Full panel, K=8, 10 seeds, each dataset on its own semantic metric (`ILASTIK PANEL DONE 2026-07-25 21:42`).

| dataset | metric | ilastik_rf | paper (old cfg) | best_v3 | nnU-Net |
|---|---|---|---|---|---|
| spheroidj | fg_iou | 0.603 | 0.902 | 0.909 | 0.822 |
| rozpad | fg_iou | 0.652 | 0.784 | 0.799 | 0.805 |
| dsb2018 | fg_iou | **0.821** | 0.846 | 0.850 | 0.851 |
| monuseg | fg_iou | 0.495 | 0.628 | 0.639 | 0.671 |
| ctc_u373 | fg_iou | 0.387 | 0.739 | 0.804 | 0.797 |
| bbbc010 | fg_iou | 0.456 | 0.571 | 0.616 | 0.645 |
| bacteria | fg_iou | 0.823 | 0.900 | 0.917 | 0.927 |
| drive | cldice | 0.624 | 0.690 | 0.771 | 0.808 |
| hrf | cldice | 0.557 | 0.680 | 0.739 | 0.795 |
| isbi2012em | cldice | 0.762 | 0.876 | 0.921 | 0.947 |
| fisbe | cldice | 0.369 | 0.669 | 0.740 | 0.770 |
| **mean (11)** | | **0.595** | 0.753 | **0.791** | — |

| field | value |
|---|---|
| **CORRECTION to the "early reading" logged above** | On three datasets (monuseg 0.50, dsb2018 0.80-0.84) the provisional reading was "the classical bank plus a forest is already near the ceiling where classical features separate the objects, and the frozen semantics buy little there". **The full panel does not support that as stated.** dsb2018 (Δ 0.029) is the ONLY dataset where the forest is close; the mean gap is **0.196**, and on spheroidj (0.31), fisbe (0.37) and ctc_u373 (0.42) it is enormous. dsb2018 is the EXCEPTION, not the rule — isolated fluorescent nuclei on dark background are precisely the task a Laplacian-of-Gaussian blob detector was designed for. Wherever the background is structured (H&E tissue, phase contrast, EM membrane) or the objects are thin, a local filter bank without semantics does not suffice. |
| why this strengthens the paper | The forest was handed OUR OWN tuned bank (a deliberate steelman) and still loses 0.196 mean, so the margin cannot be attributed to better filters. That is exactly the claim the ilastik row was added to make defensible. |
| three-level hierarchy by compute regime | nnU-Net (full supervised training on the K masks) > ours (per-episode fit over a frozen representation) > ilastik_rf (classical filter bank + forest). ilastik_rf is below nnU-Net on all eight measured datasets. This matches the taxonomy the external review proposed and is the structure Table 1 should adopt. |
| bbbc010 note | 0.456, the second lowest for the forest; overlapping worm bodies are the hardest morphology for us too (0.616). |
| date | 2026-07-25 |

### C51 — best_v3 K-scaling: the fold-in's no-regression property holds at K=1, not only at K=8

| field | value |
|---|---|
| trigger | `best_v3` was promoted on a K=8 panel ONLY (C42 ladder → `final_winner.py`). Adopting it in the paper also replaces Figure 2, whose points are K=1/4/16 — and the fold-in had never been validated at any K other than 8. Regularisation and a plateau stop are exactly the levers whose effect can differ when there is one image to overfit rather than eight, so this was open, not assumed. |
| method | `v3_kscale.sh`. `head_fusion_best_cgate_film_nobank_flat_es_ep500_wd4_do5_mix`, K=1/4/16, 11 datasets, 10 seeds, `--test 10000`, `--res 672`, cache `/disk1/prusek/cache_final10`, **separate `results/v3_k<K>` per K**, `--pool` = `run_campaign.pool_for` (15 ctc_u373, 16 isbi2012em/fisbe, else 20), `--metric_override` = the campaign convention (cldice on drive/hrf/isbi2012em/fisbe, fg_iou elsewhere). ctc_u373 excluded at K=16 (pool 15). |
| launcher | `bash /disk1/prusek/active-segmenter/v3_kscale.sh`, logs `campaign_logs/v3_k{1,4,16}.log` |
| comparison | vs `results/final10/ours_k1` (the reported config), seeds collapsed to ONE score per image before testing, paired Wilcoxon, `split_fp` asserted equal per dataset. |

**K=1 RESULTS (complete 2026-07-26 00:56):**

| dataset | metric | old K=1 | v3 K=1 | delta | p (paired) |
|---|---|---|---|---|---|
| ctc_u373 | fg_iou | 0.555 | 0.723 | **+0.168** | <0.0001 |
| fisbe | cldice | 0.509 | 0.588 | +0.078 | 0.0001 |
| drive | cldice | 0.665 | 0.732 | +0.067 | <0.0001 |
| bbbc010 | fg_iou | 0.525 | 0.568 | +0.042 | <0.0001 |
| hrf | cldice | 0.664 | 0.706 | +0.041 | <0.0001 |
| isbi2012em | cldice | 0.847 | 0.885 | +0.038 | 0.0001 |
| bacteria | fg_iou | 0.653 | 0.665 | +0.013 | 0.0024 |
| monuseg | fg_iou | 0.596 | 0.607 | +0.011 | 0.0085 |
| dsb2018 | fg_iou | 0.765 | 0.771 | +0.006 | 0.0148 |
| rozpad | fg_iou | 0.751 | 0.756 | +0.005 | 0.0126 |
| spheroidj | fg_iou | 0.804 | 0.794 | **−0.010** | 0.229 (n.s.) |

| field | value |
|---|---|
| verdict at K=1 | Ahead on **10/11**, mean **+0.042** over eleven and **+0.029** over the ten datasets the K-scaling figure averages. The single loss is not significant. The fold-in's stated property therefore holds at K=1 as well as K=8 — it was NOT safe to assume beforehand and is now measured. |
| panel mean for the paper | K=1 goes **0.678 → 0.707** over the ten K-covered datasets, against the strongest baseline's **0.600 at K=16** (INSID3 at its per-dataset better CRF mode). The Results sentence "from a single support mask we already reach 0.678" must be updated to 0.707 when best_v3 is adopted. |
| ctc_u373 is the standout | +0.168 at K=1, consistent with +0.065 at K=8 on the same dataset. Phase contrast benefits most from the regularisation, and it is the dataset where best_v3 overtakes SegGPT (0.804 vs 0.758). |
| bacteria variance at K=1 | ±0.296 across seeds: fourteen bacterial species and one support mask means one species is seen. A clean illustration that K matters differently on heterogeneous pools, and an argument for the active-selection future work. |
| status | K=4 running, K=16 queued. |
| date | 2026-07-26 |

#### C40 results — nnU-Net on the same 8 masks, COMPLETE (11/11, 10 seeds)

Paired against `best_v3` (`results/winner_panel/flat_es_ep500_wd4_do5_mix`), seeds collapsed to one
score per image before testing, `split_fp` asserted equal per dataset, Wilcoxon signed-rank.

| dataset | metric | best_v3 | nnU-Net | v3 − nnU | p (paired) |
|---|---|---|---|---|---|
| spheroidj | fg_iou | **0.909** | 0.822 | **+0.087** | 0.0079 |
| ctc_u373 | fg_iou | 0.804 | 0.797 | +0.007 | 0.568 n.s. |
| dsb2018 | fg_iou | 0.850 | 0.851 | −0.001 | 0.382 n.s. |
| rozpad | fg_iou | 0.799 | 0.805 | −0.006 | 0.439 n.s. |
| fisbe | cldice | 0.740 | 0.770 | −0.030 | 0.241 n.s. |
| bacteria | fg_iou | 0.917 | 0.927 | −0.010 | <0.0001 |
| isbi2012em | cldice | 0.921 | 0.947 | −0.025 | 0.0001 |
| bbbc010 | fg_iou | 0.616 | 0.645 | −0.029 | <0.0001 |
| monuseg | fg_iou | 0.639 | 0.671 | −0.032 | 0.0002 |
| drive | cldice | 0.771 | 0.808 | −0.037 | <0.0001 |
| hrf | cldice | 0.739 | 0.795 | −0.056 | <0.0001 |
| **mean** | | **0.791** | **0.803** | **−0.012** | |

| field | value |
|---|---|
| **verdict** | nnU-Net leads significantly on **6 of 11**, ties on **4**, and loses significantly on **1** (spheroidj, where we are +0.087). Mean gap **0.012** in nnU-Net's favour. |
| where it leads | Thin structures above all (hrf −0.056, drive −0.037), then dense/crowded fields (monuseg −0.032, bbbc010 −0.029). The pattern is consistent: a network trained end to end at native resolution recovers thin geometry a frozen backbone plus a light head does not. |
| ctc_u373, the one v3 flipped | 0.804 vs 0.797 in our favour (n.s.). Under the OLD reported configuration (0.739) we would have lost this dataset by 0.058, so adopting best_v3 is what turns it into a tie-or-better. |
| honest framing for the paper | Not "we beat nnU-Net". The defensible sentence is: nnU-Net on the same eight masks is 0.012 more accurate on average, at roughly **5x the fit cost** (100 epochs x 250 iterations ≈ 3100 s per support draw vs ~600 s, C48). **CORRECTED 2026-07-28: strike the "no warm-start path" clause.** nnU-Net accepts `-pretrained_weights` and can be fine-tuned from a previous checkpoint, so "retrains from scratch after every added mask" describes HOW WE RAN IT, not what the method can do — asserting it as a property of nnU-Net is an unfair characterisation of a baseline, the failure mode the baseline-fairness protocol exists to prevent. Worse, C40 records that OUR OWN warm refit "still needs measuring", so the incremental-cost argument rested on an asymmetry neither side had measured. The paper now claims only the measured quantity: 3100 s vs 600 s, both COLD, with the incremental case stated as unmeasured for both. The remaining open question is the low-K regime, which `nn_kscale.sh` measures. |
| what this does NOT settle | Whether nnU-Net collapses at K=1. That is the crossover hypothesis and the single most interesting remaining measurement; `nn_kscale.sh` is queued behind the v3 K-scaling by deliberate ordering. |
| date | 2026-07-26 |

**C51 K=4 RESULTS (complete 2026-07-26 06:07):** same pattern as K=1 — ahead on **10/11**, mean **+0.038**,
and the only loss is again spheroidj (−0.008, p=0.73, not significant).

| dataset | old K=4 | v3 K=4 | delta | p |
|---|---|---|---|---|
| ctc_u373 | 0.691 | 0.770 | +0.079 | <0.0001 |
| fisbe | 0.637 | 0.712 | +0.075 | 0.0001 |
| drive | 0.687 | 0.761 | +0.073 | <0.0001 |
| hrf | 0.674 | 0.733 | +0.059 | <0.0001 |
| bbbc010 | 0.566 | 0.608 | +0.042 | <0.0001 |
| isbi2012em | 0.873 | 0.913 | +0.040 | 0.0001 |
| rozpad | 0.773 | 0.791 | +0.018 | <0.0001 |
| bacteria | 0.871 | 0.887 | +0.016 | <0.0001 |
| dsb2018 | 0.806 | 0.818 | +0.012 | 0.0007 |
| monuseg | 0.604 | 0.613 | +0.009 | 0.0353 |
| spheroidj | 0.872 | 0.864 | −0.008 | 0.726 n.s. |

**Panel mean over the ten K-covered datasets (what Figure 2 plots):**

| K | old (reported) | best_v3 | shift |
|---|---|---|---|
| 1 | 0.678 | **0.707** | +0.029 |
| 4 | 0.736 | **0.770** | +0.034 |
| 8 | 0.754 | **0.790** | +0.036 |
| 16 | 0.757 | (running) | — |

| field | value |
|---|---|
| **a speculation NOT supported** | It was suggested mid-run that if best_v3 lost ground at low K the curve would change SHAPE, which would be a more interesting finding than a shift. It does not: the shift is near-uniform (+0.029 / +0.034 / +0.036). Figure 2 moves up, it does not change form. Recorded so the more exciting version is not repeated. |
| **spheroidj is the consistent exception, and it is coherent** | The only dataset where best_v3 fails to help at ANY K (−0.010 at K=1, −0.008 at K=4, +0.007 at K=8, none significant). This matches C47, where removing the classical bank IMPROVES spheroidj (0.9196 vs 0.9169). Both say the same thing: on large high-contrast spheroids DINOv3 alone is sufficient and the added machinery is neutral-to-slightly-harmful. A characteristic, not a defect, and worth one sentence in the paper. |
| headline consequence | "From a single support mask we already reach 0.678" becomes **0.707**, against the strongest baseline's 0.600 at sixteen masks. |
| date | 2026-07-26 |

### C52 — adopting best_v3 falsifies the paper's "the label budget is not binding" clause on MoNuSeg

| field | value |
|---|---|
| trigger | The v3 K=16 MoNuSeg cell came in at 0.657 against 0.639 at K=8, which contradicts a load-bearing sentence in the paper's limitation paragraph. |
| measurement (paired per image, seeds collapsed) | old reported config: K=8 **0.6278** → K=16 **0.6277**, delta **−0.0001**, p=0.81. best_v3: K=8 **0.6387** → K=16 **0.6566**, delta **+0.0179**, p=**0.0002**. |
| what breaks | §Results currently argues "Nor is the budget binding: doubling it to sixteen masks moves the score by −0.000 (paired, p=0.92) ... so the ceiling most plausibly lies in the frozen features." Under best_v3 that is FALSE — doubling the budget helps significantly. |
| **why, and it is a confounded inference, not noise** | The old head trained a FIXED 60 full-batch steps, and C42 measured its loss still descending at 60 and at 120 — it was UNDER-TRAINED. It therefore could not exploit more support images regardless of how many it was given, so "the budget does not bind" was a statement about the HEAD's training length, not about the features being saturated. The plateau early stop (`es_ep500`) removes that limit; regularisation (`wd4_do5`) is what keeps the longer training safe. Per C47 the two are non-additive (−0.020 alone, −0.012 alone, −0.007 together), so neither explains it by itself. **User's correction, 2026-07-26: the longer training is the more direct mechanism; crediting regularisation alone is one-sided.** |
| **decision (user, 2026-07-26): OPTION 1** | DROP the budget clause from the limitation paragraph. The argument then rests on the controlled interventions alone (each moved the score by at most 0.01), which still stands. Do NOT assert the new K=16 finding in the ISBI paper. |
| timing | The clause is TRUE for the configuration the paper currently reports, so it stays until the best_v3 swap and is removed as part of that edit, together with "plateaus near 0.62" → ~0.64. |
| consistent with | C34's warning that earlier ceiling conclusions were reached with an instrument partly blind to the quantity that mattered. This is a second instance of the same class: a negative result that was really a property of the measurement setup. |
| left open (not for ISBI) | Where the budget actually saturates under best_v3 — a K=32 point would give the limitation a hard number instead of an inference. |
| date | 2026-07-26 |

**C51 K=16 RESULTS (complete 2026-07-26 22:14) — `V3 KSCALE DONE`.** Ahead on **9/10**, mean **+0.043**.
ctc_u373 has no K=16 (pool 15). Only loss is again spheroidj (−0.006, n.s.).

| dataset | old K=16 | v3 K=16 | delta | p |
|---|---|---|---|---|
| fisbe | 0.650 | 0.762 | **+0.111** | 0.0001 |
| drive | 0.692 | 0.774 | +0.082 | <0.0001 |
| hrf | 0.677 | 0.742 | +0.066 | <0.0001 |
| bbbc010 | 0.576 | 0.629 | +0.054 | <0.0001 |
| isbi2012em | 0.877 | 0.927 | +0.050 | 0.0001 |
| monuseg | 0.628 | 0.657 | +0.029 | 0.0002 |
| bacteria | 0.906 | 0.932 | +0.026 | <0.0001 |
| rozpad | 0.786 | 0.806 | +0.020 | <0.0001 |
| dsb2018 | 0.854 | 0.854 | +0.000 | 0.958 |
| spheroidj | 0.923 | 0.916 | −0.006 | 0.345 n.s. |

**FINAL K-scaling curve — panel mean over the ten K-covered datasets (this is Figure 2):**

| method | K=1 | K=4 | K=8 | K=16 |
|---|---|---|---|---|
| **Ours (best_v3)** | **0.707** | **0.770** | **0.790** | **0.800** |
| Ours (old, currently in the paper) | 0.678 | 0.736 | 0.754 | 0.757 |
| INSID3 (per-dataset better CRF mode) | 0.428 | 0.468 | 0.535 | 0.577 |
| SegGPT | 0.474 | 0.506 | 0.508 | 0.511 |
| Tyche | 0.354 | 0.469 | 0.500 | 0.519 |
| UniverSeg | 0.225 | 0.394 | 0.457 | 0.493 |

| field | value |
|---|---|
| headline the paper can now make | **One support mask (0.707) beats every competing method at sixteen (best: INSID3 0.577), by 0.13.** |
| **CORRECTION to the K=4 entry above** | That entry recorded the shift as "near-uniform (+0.029/+0.034/+0.036) — Figure 2 moves up, it does not change form". **With K=16 that is wrong**: the shift is +0.043 there. The OLD config SATURATES between K=8 and K=16 (0.754 → 0.757, +0.003) while best_v3 keeps climbing (0.790 → 0.800, +0.010). The curve does change SHAPE. |
| why, and it ties C52 to the whole panel | Same mechanism as C52's MoNuSeg finding: the old head trained a fixed 60 steps and was under-trained, so it could not exploit additional support images; the plateau stop removes that limit. C52 saw it on one dataset, the K=16 panel mean shows it is general. |
| spheroidj remains the single consistent exception | −0.010 / −0.008 / +0.007 / −0.006 at K=1/4/8/16, none significant. Coherent with C47 (removing the classical bank IMPROVES spheroidj): on large high-contrast spheroids DINOv3 alone suffices. |
| date | 2026-07-26 |

### C53 — DiffKillR integration: environment, data adapter, and the upstream defects hit on the way

| field | value |
|---|---|
| trigger | The external-review claim "no in-context method compares against nnU-Net" was RETRACTED (see below). DiffKillR (Liu et al., ICASSP 2025 Oral, arXiv:2410.03058) is the counterexample: it benchmarks few-shot cell segmentation against UNet, **nn-UNet**, medical transformer, PSM, LACSS, SAM, SAM2, SAM-Med2D and MedSAM, on microscopy. It also uses **MoNuSeg**, one of our own datasets, so its published anchor is reproducible here. |
| repo | `github.com/KrishnaswamyLab/DiffKillR`, cloned to `/disk1/prusek/DiffKillR` |
| environment | `/disk1/prusek/diffkillr_env` — a CLEAN self-contained venv (python 3.10, torch 2.5.1+cu121, numpy<2, dipy 1.9.0, einops, opencv-headless, omegaconf, wandb). NOT their conda recipe (torch 1.12.1 / cudatoolkit 11.3): the core path imports nothing version-fragile and modern torch runs it. Two traps: `--system-site-packages` mixes `~/.local` and `/usr/local` and must NOT be used, and `dipy>=1.10` fails to build on this host's gcc 8.5 — 1.9.0 has a wheel. Always run with `PYTHONNOUSERSITE=1`. |
| **data adapter (ours, deliberately)** | `scripts/diffkillr_prep.py` + `scripts/diffkillr_patchify.py`. Their preprocessing parses the original MoNuSeg **XML** into masks; we already hold MoNuSeg as per-instance uint16 label maps, so only the XML parse is replaced. `patchify_and_save` and `find_background_patches` are IMPORTED from their module and called unmodified, so patch geometry, file naming and on-disk format stay byte-compatible. This is strictly fairer than running their XML path: DiffKillR then sees the identical masks and identical train/test split as every other arm on the panel, so a score difference is the method and not the data prep. |
| split fidelity | `diffkillr_prep.py` carries THEIR by-cancer TCGA id lists verbatim and asserts every id exists in our copy. All 18 train / 6 test ids matched, so their exact breast/colon/prostate split is reproducible. Our MoNuSeg is 37 train / 14 test with TCGA filenames preserved. |
| **upstream defects found (documented, minimally fixed)** | (1) `augment_MoNuSeg.py` imports `Organ2FileID` but `Metas.py` only defines `MoNuSeg_Organ2FileID` — an unpropagated rename. One-line fix, original kept at `augment_MoNuSeg.py.orig`. (2) It writes `./config/MoNuSeg_data.yaml` into a directory it does not create. (3) `main_DiffeoInvariantNet.py:561` appends `_{organ}` to `--dataset-path`, but the augment script emits no such directory, and `MoNuSegDataset` globs a FLAT `image/` dir while the augmented output is per-augmentation-method subfolders. (4) `class_labels.csv` is written only by the XML path we bypassed. Resolved by building a per-organ patch folder holding exactly the 10% subset their augmenter selected (221 of 2215 breast cell patches) plus the CSV — no further code change. |
| status | **RUNNING.** DiffeoInvariantNet trains: 2-epoch smoke gives test instance mAP **0.569** (their published breast anchor is 0.954 at 50 epochs, so the direction is right). Next: full DiffeoInvariantNet, then DiffeoMappingNet, then inference + stitch + `evaluate_monuseg.py`, then the anchor comparison. |
| **scope limit, architectural not practical** | DiffKillR works on 96x96 crops centred on cell CENTROIDS and its premise is that cells are near-diffeomorphic copies of an archetype. That holds for nuclei and compact cells; it cannot hold for vessel trees (DRIVE/HRF), long filaments (FISBE), membranes (ISBI2012-EM) or overlapping worm bodies (BBBC010), where no archetype exists whose warps generate the others. So this is a TARGETED comparison on nucleus/cell datasets (MoNuSeg above all — our hardest dataset and their claimed strength), not an eleven-dataset Table 1 column. That limit must be stated as the method's design scope, not as a failed run. |
| date | 2026-07-26 |

#### C53 addendum — DiffKillR anchor does NOT reproduce at the few-shot setting; investigating

| field | value |
|---|---|
| what was run | `main_DiffeoInvariantNet.py --dataset-name MoNuSeg --organ Breast --percentage 0.1` (their 10% few-shot subset, 221 of 2215 breast cell patches), `--DiffeoInvariantNet-model AutoEncoder --max-epochs 50`, their default everything else. |
| **anchor mismatch** | Their Table 1 (unit test, cell matching on histology, Breast Cancer) reports instance **MAP 0.954 ± 0.023**. Ours: test instance **mAP 0.656** (clustering accuracy 0.297, top-3 0.489). That is far outside their stated deviation, so the integration is NOT yet validated and no DiffKillR number may be reported on our panel. |
| leading hypothesis | Their Table 1 is a UNIT TEST of the matching network and does not state a data percentage; the 10% few-shot condition belongs to their Fig. 4 segmentation experiment. Training on 10% of the patches would plausibly cost exactly this much matching quality. |
| test in flight | Re-running identically but with the FULL breast subset (2215 patches, `--percentage 1.0`, folder `MoNuSegFull_patch_96x96_Breast`). If mAP approaches 0.954 the integration is validated and the 10% number is simply the few-shot condition; if it does not, something in our data adapter or their pipeline differs and must be found before anything is reported. |
| naming trap noticed | `main_DiffeoInvariantNet.py` formats the checkpoint as `fewShot-{percentage:.1f}%`, so `--percentage 0.1` renders as "0.1%" while `augment_MoNuSeg.py` treats the same value as the FRACTION 0.1 = 10% (it selected 221 of 2215). The string is misleading; the data is 10%. |
| protocol note | This is the baseline-fairness rule working as intended: reproduce a published anchor BEFORE trusting any number. Had the 10% run been reported as "DiffKillR on our panel", a reviewer comparing to the published 0.954 would have concluded we broke their method. |
| date | 2026-07-27 |

#### C53 verdict — DiffKillR's headline few-shot experiment is NOT reproducible from the released code

| field | value |
|---|---|
| how far it got | Environment built, data adapter written, their own patchifier driven with our masks, their exact by-cancer split reproduced, **both networks trained to completion** (DiffeoInvariantNet 50 epochs, DiffeoMappingNet VM-Diff 50 epochs, checkpoints on disk). |
| anchor status | Their Table 1 (unit test, cell matching, Breast) reports instance MAP **0.954 ± 0.023**. Ours: **0.656** at their 10% few-shot setting, **0.854** at 100% data. The data fraction explains most of the gap but ~0.10 remains, consistent across all three of their metrics (ours 0.819/0.780/0.854 vs theirs 0.954/0.949/0.912). Untested remaining candidate: augmentation multiplier (argparse default 2, but their own example path reads `m3`). **Anchor NOT reproduced.** |
| **the blocker** | The released repo contains **no script that produces the DiffKillR predictions their evaluation consumes**. `evaluate_monuseg.py` reads stitched prediction PNGs from `../results/<folder>/<dir>/<model>_stitched/`; `main_DiffeoInvariantNet.py` and `main_DiffeoMappingNet.py` write only side-by-side *figures* for visual logging, never prediction masks; and `main_inference.py` is a single-image DEMO hard-coded to `../data/A28-87_..._patch_96x96/image/EpithelialCell_H3191_W6445_patch_96x96.png`, a file from their **unreleased A28 dataset**. So the path from trained checkpoints to the Fig. 4 numbers (the ones compared against nn-UNet) is not in the release. |
| why we stop here | Writing that export step ourselves means implementing their inference-to-stitched-mask pipeline from scratch. Any number it produced would be OUR reimplementation, not their published method, and with the Table 1 anchor already 0.10 low there would be no way to tell a faithful result from a broken one. Reporting it as "DiffKillR" would be exactly the unfair-baseline error this project's protocol exists to prevent. |
| **ten upstream defects found** | (1) `augment_MoNuSeg.py` imports `Organ2FileID`; `Metas.py` defines only `MoNuSeg_Organ2FileID`. (2) It writes `./config/MoNuSeg_data.yaml` into a directory it never creates. (3) `main_DiffeoInvariantNet.py:561` appends `_{organ}` to the dataset path; the augmenter emits no such directory. (4) `MoNuSegDataset` globs a FLAT `image/` dir while the augmenter writes per-augmentation-method subfolders. (5) `class_labels.csv` is produced only by the XML path. (6) `main_DiffeoMappingNet.py` rejects `--DiffeoInvariantNet-model`. (7) It does NOT append `_{organ}` although `main_DiffeoInvariantNet.py` does — the two scripts need different paths for the same data. (8) `main_inference.py` resolves the model via `globals()[name]`, which can never match `'VM-Diff'`, and never imports VoxelMorph — as released it runs only `UNet`, not the VM-Diff their Table 2 selects. (9) Training and inference use incompatible checkpoint naming schemes. (10) The prediction-export step is absent entirely. |
| what IS reportable | That the integration reached trained checkpoints, and that the anchor did not reproduce, is itself a finding about the reproducibility of this baseline. It does NOT license any DiffKillR number in our tables. |
| **what survives for the paper** | The retraction stands on its own and does not depend on running DiffKillR: their paper **does** benchmark few-shot cell segmentation against nn-UNet (Fig. 4, ~parity at 10% training data on breast/colon/prostate histology), so our earlier "nobody compares against nnU-Net" framing is false and must not be used. DiffKillR belongs in RELATED WORK as the counterexample, cited from its published numbers, not as a column in our tables. |
| artefacts | `/disk1/prusek/DiffKillR` (repo + `.orig` backups of every patched file), `/disk1/prusek/diffkillr_env`, `scripts/diffkillr_prep.py`, `scripts/diffkillr_patchify.py`, logs `campaign_logs/dk_*.log`. |
| date | 2026-07-27 |

### C54 — best_v3 adopted into the paper; Table 2 is now STALE and must be re-run

| field | value |
|---|---|
| what changed | Table 1 and Figure 2 regenerated on `best_v3` (`oursv3n_k8`, `oursv3_k{1,4,16}`), with **ilastik-RF** and **nnU-Net** added as same-budget columns. Prose rewritten: abstract, intro contribution, results, specialist paragraph, limitation paragraph, implementation, baselines, figure caption. Compiles clean, 0 undefined refs. |
| **metric-label trap found and fixed** | The winner-panel run that produced best_v3 at K=8 used a BLANKET `--metric_override fg_iou`, so on drive/hrf/isbi2012em/fisbe its records carry `metric="fg_iou"` with clDice parked in `metric_native` -- while nnU-Net, ilastik and every baseline write `metric="cldice"`. `make_semantic_tables.stat()` matches on `metric`, so the raw dir would have dropped those four datasets from the K=8 column only, bending the curve at exactly one K. `scripts/normalize_v3_metric.py` re-keys them into `results/final10/oursv3n_k8` (a relabelling of values measured in the same run, originals untouched, derived means verified against an independent recomputation: drive 0.7715, hrf 0.7389, isbi2012em 0.9214, fisbe 0.7397). This is the exact failure `run_campaign.cmd_for` documents; our own `v3_kscale.sh` avoided it. |
| headline change, stated honestly | With nnU-Net in the table, "Ours" is bold on **1 of 11** rows (spheroids), not 8 of 11. The paper no longer claims to be the most accurate method. It claims: best against every FORWARD-PASS method on all eleven datasets and at every K; behind a from-scratch nnU-Net by 0.012 mean at K=8; ahead at K=1 by an amount that tracks dataset heterogeneity; ~5x cheaper to fit and the only one of the three with a warm start. |
| **OPEN AND BLOCKING: Table 2 is stale** | `tab_ablation.tex` and its prose still ablate the OLD configuration (backbone-only 0.392/0.464, +bank 0.605/0.660, gate+FiLM +0.018, colour -0.028/-0.029, all-self-config -0.044). Those were measured on best_v2. C47's leave-one-out of best_v3 gives a DIFFERENT picture -- cgate -0.0023 and FiLM -0.0025, i.e. both at noise -- but that is a 3-seed 4-dataset SCREEN and cannot go into a paper table. So Table 2 currently ablates a method the paper no longer reports. Either re-run the ablation arms at 10 seeds on the reported panel, or state in the caption that Table 2 characterises the base configuration. Re-running is the correct fix. |
| also open | Holm-adjusted significance not shown in Table 1; frozen-backbone probe column running; nnU-Net K=1 running (4/11); page count 6, technical spill onto p5 down from 30 to 22 lines but still an ISBI reject until trimmed. |
| date | 2026-07-27 |

### C55 — Frozen-backbone probe: how much of the margin is "we got to train something"?

| field | value |
|---|---|
| trigger | The paper says plainly that we fit a head by gradient descent while UniverSeg/Tyche/SegGPT/INSID3 consume the support in a forward pass. The reviewer question that follows immediately is how much of the margin is simply that we trained. This arm answers it. |
| method | `head_fusion_best_cgate_film_nobank_flat_es_ep500_wd4_do5_mix_nocls_nocolor_noloss` — best_v3 with the classical prior bank ZEROED (`nocls`) and every support-derived rule OFF (`nocolor`, `noloss`; contrast_norm follows nocolor). Frozen DINOv3 plus the same light head, same training regime (flat stem, plateau stop to 500, wd 1e-4, dropout 0.5, mix), so this is a control on the FEATURES, not a confound with how long the head trains. |
| launcher | `bash /disk1/prusek/active-segmenter/probe_panel.sh` (completed 2026-07-27 22:30, 11/11) |
| protocol | K=8, 10 seeds, `--test 10000`, res 672, `--pool` = `run_campaign.pool_for`, `--metric_override` = campaign convention, cache `/disk1/prusek/cache_final10`, score_dir `results/final10/probe_k8`. Paired-comparable with every other arm. |

| dataset | metric | probe | best_v3 | delta | n_img |
|---|---|---|---|---|---|
| drive | cldice | 0.5112 | 0.771 | **−0.2598** | 200 |
| rozpad | fg_iou | 0.5964 | 0.799 | **−0.2026** | 240 |
| bacteria | fg_iou | 0.7938 | 0.917 | −0.1232 | 1480 |
| fisbe | cldice | 0.6179 | 0.740 | −0.1221 | 140 |
| bbbc010 | fg_iou | 0.4945 | 0.616 | −0.1215 | 600 |
| hrf | cldice | 0.6210 | 0.739 | −0.1180 | 250 |
| monuseg | fg_iou | 0.5216 | 0.639 | −0.1174 | 140 |
| isbi2012em | cldice | 0.8081 | 0.921 | −0.1129 | 140 |
| dsb2018 | fg_iou | 0.7883 | 0.850 | −0.0617 | 500 |
| ctc_u373 | fg_iou | 0.7567 | 0.804 | −0.0473 | 190 |
| spheroidj | fg_iou | 0.8863 | 0.909 | −0.0227 | 240 |
| **mean (11)** | | **0.6723** | **0.7914** | **−0.1190** |

| field | value |
|---|---|
| **finding 1 — the answer to the reviewer question** | Frozen DINOv3 plus a trained head, with no bank and no self-configuration, reaches 0.672 mean against best_v3's 0.791. So the gradient step alone is NOT where the margin comes from: the bank and the support-derived rules are worth 0.119 mean on top of it. For scale, `ilastik_rf` (C49, a forest over the bank ALONE) scores 0.595 — so neither block carries the method by itself. |
| **finding 2 — the contribution is wildly dataset-dependent, by a factor of 11** | drive −0.260 and rozpad −0.203 at one end against spheroidj −0.023 and ctc_u373 −0.047 at the other. Thin structures and fine-grained boundaries need the native-resolution bank; large well-separated objects on clean backgrounds do not. This is the same per-dataset structure C44 measured from the other direction: equal-weight fusion fails because the right block weighting differs per dataset, and learning it is what the 1x1 does. |
| rozpad is the surprise | −0.203 on binary crumb foreground, where semantics might have been expected to suffice. The boundaries are at native resolution and the patch-16 grid blurs them, which is the same mechanism that puts drive at the top. |
| use in the paper | This is the negative control the Introduction's in-context framing needs. It is also the honest bound on what a purely forward-pass variant of THIS pipeline could reach without the bank. |
| date | 2026-07-27 |

### C56 — Lambda sensitivity: is the support-LOO selecting a real optimum, or decorating a flat curve?

| field | value |
|---|---|
| trigger | A silent-failure review measured that the reduced-fidelity LOO's row subsample perturbs the lambda score table by more than the gap between adjacent lambdas on SYNTHETIC data at 1% and 5% foreground, picking a different lambda in 3 of 4 draws. If the table is flat within noise, then "lambda self-configures from the support" — one of the paper's three claims — is selecting noise. Nobody had ever measured whether the choice of lambda changes the final answer at all. |
| method | `--fixed_lam` (new flag) bypasses the LOO and fits at a FIXED lambda, so the final score can be read off per grid value. Rung L0, K=8, 3 seeds, `--test 10000`, res 672, `--gpu 0`, cache `/disk1/prusek/cache_final10`, `normalize_penalty=True` so lambda is per-sample. Two datasets chosen to bracket the panel's foreground range: drive (~1% fg, cldice) and monuseg (dense nuclei, the highest fg, fg_iou). |
| launcher | `bash scripts/lamsweep.sh` on tulen (10 cells, completed 2026-07-27 23:43). Originally run from `/tmp/lamsweep.sh`, which does not survive a reboot; reconstructed into the repo at `scripts/lamsweep.sh` from this entry's method/parameters (review finding 4) so it is re-runnable verbatim. |

| lambda | drive (cldice, n=60) | monuseg (fg_iou, n=42) |
|---|---|---|
| 1e-4 | 0.5763 | 0.5030 |
| **1e-3** | **0.5979** | 0.5589 |
| **1e-2** | 0.5412 | **0.5902** |
| 1e-1 | 0.4719 | 0.5710 |
| 1.0 | 0.4786 | 0.5296 |
| **spread** | **0.1261** | **0.0872** |

| field | value |
|---|---|
| **finding 1 — lambda is load-bearing** | Spread 0.087–0.126 across the grid, i.e. four to six times the project's 0.02 decision gate. A badly chosen lambda costs more than the entire difference between competing methods. |
| **finding 2 — the optimum is dataset-dependent** | 1e-3 on drive, 1e-2 on monuseg. Fixing a single lambda would cost 0.031 (if fixed at 1e-3) or 0.057 (if fixed at 1e-2) on whichever dataset it is wrong for. So "fix lambda and stop claiming it is selected" is NOT an available simplification — the choice genuinely has to be made per dataset. |
| **finding 3 — the support-LOO finds the optimum** | It selected 1e-3 on drive and 1e-2 on monuseg: on both datasets, exactly the best value in the grid. The LOO-selected drive run scored cldice 0.598 against the fixed-1e-3 sweep's 0.5979 — agreement to a thousandth. |
| resolution of the review finding | The flatness the review measured came from synthetic data at fixed foreground fractions. On real data the LOO table has genuine curvature and the argmax is a real choice, not an argmax over noise. The narrow top-2 margin seen on one drive seed remains a caveat about the FINE selection, but both configurations still picked the true optimum. |
| what this licenses | The self-configuration claim for lambda is measured and load-bearing, not decorative. This sweep is the sensitivity analysis a reviewer would ask for and it belongs in the paper. |
| caveat | Two datasets, rung L0, 3 seeds. Grid endpoints (1e-4 and 1.0) are both clearly suboptimal on both datasets, so the grid brackets the optimum rather than truncating it — but a dataset whose optimum sits outside the grid would not be visible from this measurement. |
| date | 2026-07-28 |

### C57 — Does the trained head transfer across datasets? NO, and it decides the in-context redesign

| field | value |
|---|---|
| trigger | Making the method genuinely in-context requires adapting without gradient updates, which requires the head to be learned ONCE and held fixed — meta-training, as UniverSeg and Tyche do. `scripts/lodo_head.py` was written to measure that prerequisite and had never been run. The user's proposal that prompted it: meta-train the stem across datasets, then solve only the 1x1 classifier in closed form on the 99-channel penultimate (64 projected backbone dims + 35 classical priors), which would put the solve at d=99 rather than d=1062 and bring the sub-second target into reach. |
| method | `scripts/lodo_head.py --holdout monuseg,drive,dsb2018 --seeds 3`, method `head_fusion_best_cgate_film_nobank_flat_h128_nocolor`, K=8, cache `/disk1/prusek/cache_final10`, GPU 0 (the A5000 OOM'd: nnU-Net held 15.5 of its 24 GB). Pairwise rather than union — both arms then see exactly EIGHT support images, so the comparison isolates WHERE the support came from rather than how much of it there was. Colour selection disabled on both arms so head transfer is not confounded with a colour rule applied to the wrong modality. |
| the transferred arm is a faithful proxy | Weights fitted on the SOURCE, but the support-derived conditioning (FiLM prototypes, correspondence channel) REFRESHED from the held-out dataset's own support. That is exactly what a meta-trained head would do at deployment: frozen weights, conditioning read from the target. No training happens in that refresh; the prototypes are closed-form. |
| sources | spheroidj, dsb2018, drive, isbi2012em (first three excluding the holdout), spread across morphologies so a "transfers" result could not be an artefact of picking a near-identical source. |

| holdout | metric | per-dataset fit | transferred | delta |
|---|---|---|---|---|
| monuseg | fg_iou | 0.6276 | 0.0891 | **−0.5385** |
| drive | cldice | 0.7004 | 0.1108 | **−0.5896** |
| dsb2018 | fg_iou | 0.8533 | 0.0362 | **−0.8171** |

| field | value |
|---|---|
| **verdict** | **Total collapse, universal.** Transferred scores of 0.04–0.11 are effectively empty masks, not degraded predictions. The script's own docstring set the decision rule: *"If transferred lands close to per-dataset, a meta-trained head is viable and the adaptation collapses to a closed-form solve. If it collapses, the in-context redesign is a dead end."* |
| **finding — it is not a morphology effect** | dsb2018 was the arm most likely to transfer: its sources include spheroidj, the same blob morphology. It collapsed HARDEST (−0.817). So the reading "transfers within a morphology" is falsified, and the stronger conclusion stands: the 99-channel representation has no shared meaning across datasets at all. |
| mechanism, consistent with C55 and C44 | 35 of those 99 channels are the classical bank, whose contribution C55 measured as varying by a factor of **11** across the panel (drive −0.260, spheroidj −0.023). A weight vector learned elsewhere therefore points at channels that mean something different here. C44 found the same thing from the other side: equal-weight fusion fails because the right block weighting is per-dataset, and learning it is exactly what the trained layer does. |
| what this rules out | Meta-training the stem (or any fixed head) and freezing it. Not because ten datasets are too few — Iris (CVPR 2025) Figure 6 puts saturation at 60–80 tasks — but for the prior reason that the target representation is not shared. |
| what it leaves open | Amortizing the RULE that computes the weights rather than the weights themselves: a hypernetwork or set-encoder that reads the support and emits the readout, so nothing per-dataset is ever frozen. This result is an argument FOR that direction, not against it. |
| cost of learning this | One hour of GPU, against the weeks a meta-training build would have taken before failing. |
| date | 2026-07-28 |

### C58 — Closed-form supervised projections as a stem replacement: NO-GO, and the diagnostic was wrong

| field | value |
|---|---|
| trigger | After C57 killed the meta-trained-and-frozen stem, the remaining route to removing per-task training was to COMPUTE the projection from the support masks instead of learning it. The failed feature maps so far (random ReLU expansion, Nystrom over support landmarks) share one property: they are built WITHOUT the labels. A literature survey named two closed-form supervised projections that use them — Partial Least Squares, which escapes the rank-1 trap that disqualifies Fisher LDA for a binary target because its weights span the Krylov subspace K_r(X'X, X'y), and Fukunaga-Koontz, which is driven by the covariance DIFFERENCE rather than the mean difference and therefore should suit thin structures, where mean separation is weak and texture separation is strong. |
| method | `/tmp/proj_prescreen.py` on tulen (GPU 0). Training-free: for each candidate projection built from the support masks, fit a ridge readout in the projected space on K-1 support images and score the held-out one. r=32, matching the stem's output width, so the comparison is like-for-like. Metric is per-pixel AUC (threshold-free, so a projection is not penalised for where it puts the operating point). DINO block only — the projection replaces the stem, which reads the backbone, not the bank. K=8, seed 0, leave-one-support-image-out. |
| arms | `raw` all 1024 dims (the ceiling); `pca` unsupervised control; `pls` NIPALS with deflation; `fkt` Fukunaga-Koontz with Ledoit-Wolf-style shrinkage. |

| dataset | raw (1024) | pca-32 | pls-32 | fkt-32 | L0's measured IoU loss |
|---|---|---|---|---|---|
| dsb2018 | 0.9939 | 0.9933 | **0.9946** | 0.9884 | −0.013 |
| drive | 0.9016 | 0.8825 | 0.8958 | 0.7896 | −0.173 |
| hrf | 0.8312 | 0.8164 | 0.8259 | 0.7083 | −0.223 |
| monuseg | 0.8294 | 0.7892 | 0.8255 | 0.7793 | −0.049 |

| field | value |
|---|---|
| **finding 1 — dimensionality reduction is not the bottleneck** | Unsupervised PCA to 32 dims retains 98–99.9% of the raw AUC on every dataset. Compressing 1024 channels to 32 costs 0.006–0.040 AUC and needs no labels at all. So whatever the trained stem contributes, it is not compression. |
| **finding 2 — supervised projection buys almost nothing** | PLS over PCA: +0.013 (drive), +0.010 (hrf), +0.036 (monuseg), +0.001 (dsb2018). An order of magnitude below the 0.103–0.223 gap it was meant to close. |
| **finding 3 — the covariance hypothesis is falsified** | FKT is the WORST arm on every dataset, including both thin-structure sets where it was predicted to win. The reasoning was that vessels have weak mean separation and strong texture separation; the measurement says otherwise. |
| **finding 4 — AUC does not track the IoU gap, and this is the useful part** | Ranked by AUC, drive (0.902) beats monuseg (0.829). Ranked by L0's IoU loss, drive (−0.173) is 3.5x WORSE than monuseg (−0.049). If the gap were about feature quality the two orderings would agree. They do not. What separates drive and hrf from monuseg and dsb2018 is foreground prevalence (1–9% against tens of percent), which is exactly where the probabilistic optimum and the overlap optimum diverge. |
| **the diagnostic was the wrong one** | Saito & Rehmsmeier (PLOS ONE 2015, 4802 citations) show ROC/AUC is deceptive under class imbalance and that AUPRC at the true prevalence predicts achievable performance. This screen used AUC, which is why it ranked drive above monuseg while the IoU gap runs the other way. Any repeat should use AUPRC at prevalence. |
| **verdict** | **NO-GO for closed-form supervised projections.** Three routes are now closed by measurement: reduction is not the bottleneck (PCA proves it), supervision of the projection adds ~0.01–0.04 AUC, and the covariance-difference route is worse than the unsupervised control. |
| what it redirects to | The gap is downstream of the representation. A linear readout already ranks pixels at AUC 0.83–0.99; what it does badly is convert that ranking into overlap. The literature's answer is the Dice/F1-optimal threshold (Lipton et al. ECML 2014: threshold at F*/2 for calibrated scores, self-referential but computable by an O(n log n) sort-and-sweep) and, for the fit itself, the Lovász-Jaccard surrogate, which stays convex when composed with a LINEAR readout. Neither needs per-task gradient descent. |
| **the limit that neither fixes** | clDice (Shit et al., CVPR 2021) argues that one missing pixel on a thin vessel costs ~0 Dice but severs a component, so no per-pixel rule, however Dice-optimal in expectation, can be topology-correct. That matches C-nnunet-K1: we lose on all three clDice datasets (mean −0.051) and win on the seven fg_iou ones (mean +0.056). The split is by METRIC, not by morphology or dataset size. |
| date | 2026-07-28 |

### C59 — The gap was in the DECISION RULE, not the representation: a swept threshold recovers most of it

| field | value |
|---|---|
| trigger | C58 closed the last representation-side route and left a contradiction: a linear readout on the frozen features RANKS pixels well (AUC 0.83–0.99) yet segments badly (IoU 0.013–0.223 below the trained head), and the two orderings do not agree. Literature (Lipton, Elkan & Naryanaswamy, ECML 2014) says why: a probabilistically calibrated classifier is not Dice-optimal, and the two optima diverge most at low foreground prevalence — the axis that separates our worst datasets from our best. The Dice-optimal cut is self-referential (threshold at F*/2) but computable exactly by an O(n log n) sort-and-sweep, with no gradient descent. |
| method | `/tmp/thresh_sweep.py` on tulen. For each dataset, fit ONE readout per leave-one-support-image-out fold with `fit_irls` — the same solver, balanced weights, lambda and normalise-penalty convention the arm uses — then score the held-out image under three decision rules on identical weights. K=8, seed 0, lambda 1e-2, 15 Newton steps. |
| arms | `fixed .5` (what the arm does now); `sweep-LOO` (threshold swept on the K−1 training images, applied to the held-out one — the honest candidate); `oracle` (threshold swept on the held-out image itself — NOT a method, a ceiling that bounds what any threshold rule can win). |

| dataset | fg% | fixed .5 | sweep-LOO | oracle | LOO−fixed | oracle−LOO |
|---|---|---|---|---|---|---|
| drive | 9.0 | 0.5991 | **0.7221** | 0.7426 | **+0.1231** | 0.021 |
| hrf | 8.4 | 0.6139 | **0.7118** | 0.7230 | **+0.0980** | 0.011 |
| bbbc010 | 4.1 | 0.5410 | 0.5730 | 0.6351 | +0.0320 | 0.062 |
| dsb2018 | 14.7 | 0.9051 | 0.9174 | 0.9415 | +0.0123 | 0.024 |
| spheroidj | 7.5 | 0.7626 | 0.7746 | 0.9491 | +0.0120 | 0.175 |
| monuseg | 28.0 | 0.7486 | 0.7525 | 0.7775 | +0.0039 | 0.025 |

| field | value |
|---|---|
| **finding 1 — the gain lands exactly where the gap was** | drive +0.123 and hrf +0.098, the two datasets where the closed-form arm trailed worst (−0.173, −0.223); everything else gains under 0.032. Against best_v3 the remaining deficit falls from −0.173 to −0.049 on drive and from −0.223 to −0.027 on hrf, i.e. the swept threshold recovers 71% and 88% of the gap for the cost of a sort. |
| **finding 2 — the leave-one-out estimate is nearly optimal where it matters** | On drive and hrf the LOO threshold lands within 0.021 and 0.011 of the oracle, so Lipton et al.'s low-prevalence instability warning does not bite there. It does elsewhere: spheroidj leaves 0.175 on the table and bbbc010 0.062, so prevalence alone does not predict where the estimate works — the threshold also has to be STABLE across support images, and on overlapping worms and mixed-modality spheroids it is not. |
| **finding 3 — this reframes five earlier negatives** | The projection ladder (L0/L1/L2), the meta-trained stem (C57) and the closed-form supervised projections (C58) were all searching the representation. The measurement says the representation was adequate and the decision rule was not. |
| **A DEFECT IN THE FIRST VERSION OF THIS EXPERIMENT, recorded because it nearly produced a false headline** | v1 fitted a plain weighted RIDGE rather than IRLS and reported gains of +0.52. Its `fixed .5` column scored 0.215 on drive where the real arm scores 0.598 — the baseline was three times too low and most of the "gain" was repairing the harness. The cause is documented in the production code itself (`incontext_backend.py` line 487): balanced class weights make the fitted intercept encode a 50% prior while the images are 1–10% foreground, so 0.5 is wrong BY CONSTRUCTION for a readout with no intercept calibration. IRLS calibrates it; ridge does not. The rule this enforces: a threshold experiment must reproduce the arm's own baseline number before its deltas mean anything. |
| **OPEN, and it decides whether this is usable** | These are Dice numbers. drive, hrf, isbi2012em and fisbe are reported on centreline Dice, which penalises breaks in a thin structure. A threshold optimised for Dice has no reason to be good for clDice, and Shit et al. (CVPR 2021) argue no per-pixel rule can be topology-correct: one missing pixel costs ~0 Dice but severs a component. The oracle bound is also suggestive — 0.743 on drive against the trained head's 0.771, so even a perfect threshold does not close vessels entirely. Must be re-measured on clDice before any claim. |
| date | 2026-07-28 |
### C60 — The Dice-swept threshold does NOT transfer to clDice: C59's positive closed

| field | value |
|---|---|
| trigger | C59 found that a Dice-optimal swept threshold recovers 71% (drive) and 88% (hrf) of the closed-form readout's deficit, and named this the one positive result of the closed-form investigation. But C59 measured on **Dice**, while drive and hrf are reported on **clDice**. An improvement that exists only in the metric it is tuned on is not an improvement. |
| method | The same sort-and-sweep threshold selection, scored on clDice instead of Dice. `/tmp/thresh_cldice.py` on tulen. |
| results | drive **+0.0030** (0.6760 -> 0.6790 against a 0.6962 ceiling); hrf **-0.0015** (0.6656 -> 0.6641 against 0.6809). |
| finding | Worth approximately nothing, and on one of the two datasets it is negative. The threshold sweep recovers overlap, and clDice does not measure overlap -- one pixel removed from a vessel centreline costs ~0 Dice while severing a component, which is Shit et al.'s entire argument. |
| **what it closes** | C59's positive does not survive the reported metric. All six routes in C55-C59 are now closed, including the one that looked like an escape. The decision rule is not where the clDice gap lives either. |
| what it redirects to | If the gap on the three centreline datasets is neither the representation (C58) nor the decision rule (C59+C60), it is the OBJECTIVE. That is what the `convex_loss` arm tests: whether an objective assembled from convex terms (Lovasz + skeleton recall + an anti-bridging weight derived from the GT) can carry topology without clDice's non-convex Tprec factor. |
| **THIS ENTRY IS WRONG — SUPERSEDED BY C61 (2026-07-29)** | Its unswept BASELINE is too high on both datasets, which is what made the sweep look worthless. C61 measures the same quantity at the same seed, K and lambda and gets drive $0.5645$ where this says $0.6760$, hrf comparably. The two experiments AGREE on where the swept arm lands (drive $0.6788$ vs $0.6790$; hrf $0.6594$ vs $0.6641$) and disagree only on the baseline. C61's arm A reproduces C59's Dice to $\|d\|\le0.005$ on both datasets; this entry's clDice baseline was never anchored to anything — the exact defect C59 recorded as its own lesson, in the opposite direction. **The sweep is worth +0.113 (drive) and +0.060 (hrf) on clDice, not +0.003 / −0.002.** |
| date | 2026-07-29 |

### C61 — Dinkelbach fractional programming: NO-GO on 4/4; and it CORRECTS C60

| field | value |
|---|---|
| trigger | Koyejo et al. (NeurIPS 2014, Thm 2) prove the Bayes-optimal classifier for the linear-fractional family (Dice, Jaccard) is a THRESHOLD, so a Dinkelbach fit-time reweighting and a post-hoc threshold sweep should be the same intervention. If the equivalence bites at our sample size, fractional programming buys the operating point in one fit instead of a sweep and nothing else. Cheap and decisive, so it ran before anything was built on it. |
| method | `scripts/screen_dinkelbach.py`. Four arms sharing ONE feature map, one lambda and one support draw, so the only differences are the final solve's sample weights and the decision threshold. **A** = `fit_irls` + `_balance_weights` at threshold 0.5 (the arm as it stands); **A+sw** = A with the threshold chosen on the K−1 training support images and applied to the held-out one; **B** = Dinkelbach, re-solved with per-pixel cost (2−λ) on fg and λ on bg, λ ← pooled Dice; **C** = B plus the same leave-one-out sweep. Threshold shifts applied in LOGIT space before the resize, since `p > τ` is not `z > logit(τ)` once `_finalize_mask` resizes. |
| protocol | Leave-one-support-image-out (K=8 → 8 folds), 3 seeds, λ fixed at 1e-2 matching C59, `--tau_grid 15`, res 672. **The test split is never touched** — this is a pre-screen and the script refuses TEST_FIVE without `--unlock_test`. Unit of analysis = one held-out support image (24 per dataset). |
| launcher | `CUDA_VISIBLE_DEVICES=0 python scripts/screen_dinkelbach.py --datasets hrf --seeds 3 --tau_grid 15 --verbose --cache /disk1/prusek/asg_cache_dink_a100 --out /disk1/prusek/dink_logs/screen_hrf.json` and the same with `--datasets drive,dsb2018,monuseg`, `CUDA_VISIBLE_DEVICES=1`, cache `asg_cache_dink_a5000`, out `screen_rest.json`. Separate writable caches per GPU (the feature-cache race). |

| dataset | role | metric | A | A+sw | B | C | **C − A+sw** | mean\|dw\|/\|w\| | λ-nonconv |
|---|---|---|---|---|---|---|---|---|---|
| drive | TARGET | cldice | 0.5655 | **0.6788** | 0.3128 | 0.6289 | **−0.0499** | 0.410 | 20 |
| hrf | TARGET | cldice | 0.5994 | **0.6594** | 0.2553 | 0.6244 | **−0.0350** | 0.571 | 24 |
| dsb2018 | CONTROL | fg_iou | 0.8375 | 0.8488 | 0.8322 | 0.8494 | +0.0006 | 0.026 | 0 |
| monuseg | CONTROL | fg_iou | 0.5915 | 0.5925 | 0.5670 | 0.5898 | −0.0027 | 0.239 | 0 |

| field | value |
|---|---|
| **finding 1 — NO-GO, and it is not a null measurement** | C is neutral on both controls (±0.003) and clearly WORSE than a plain threshold sweep on both targets. The GO gate needs a target to gain; both lose. `mean\|dw\|/\|w\|` is 0.41–0.57 on the targets, so the loop genuinely rewrote the weights — this is a real negative, not a lever that never fired. |
| **finding 2 — the mechanism, and it is structural** | λ failed to converge on 20 and 24 folds on drive and hrf, and on ZERO folds of either control. The Koyejo equivalence check is correspondingly broken exactly there (drive swept τ=0.937 against λ/2=0.168; hrf 0.886 against 0.132) while the controls are far closer. Dinkelbach linearises a RATIO of the Dice/Jaccard family. clDice is not in that family — it is a harmonic mean of two ratios, one containing `skeleton(pred)` — so on the centreline datasets it optimises one functional and is scored by another. The threshold sweep has no such mismatch, which is why it wins. |
| **finding 3 — THIS CORRECTS C60, which closed the wrong door** | C60 concluded the Dice-swept threshold does not transfer to clDice (drive +0.003, hrf −0.002) and on that basis declared all six closed-form routes shut. It is wrong, and the two experiments locate the error precisely: they agree on the SWEPT arm (drive 0.6788 here vs C60's 0.6790; hrf 0.6594 vs 0.6641) and disagree on the UNSWEPT baseline by 0.111 and 0.066. Arm A here reproduces C59's Dice on both (drive 0.5995 vs 0.5991; hrf 0.6190 vs 0.6139); C60's clDice baseline was anchored to nothing. A third independent measurement agrees with this one: the test-slice record `results/incontext/L0` puts drive at 0.5982 at threshold 0.5. **The sweep is worth +0.113 on drive and +0.060 on hrf.** C59's positive survives the reported metric. |
| the rule this re-enforces | C59 wrote it after its own ridge-instead-of-IRLS defect: *a threshold experiment must reproduce the arm's own baseline before its deltas mean anything*. C60 did not, and inverted its own conclusion. This screen asserts the baseline in code and prints the check, which is why the disagreement was findable at all rather than being absorbed. |
| what is still open | Whether the swept threshold transfers to the TEST SLICE at K=8 (this is LOO on support). The L0 test-slice records exist at threshold 0.5 only; adding A+sw there is the one missing cell and is cheap. |
| cost | drive+dsb2018+monuseg 5.5 h on the A5000; hrf 20272 s (5.6 h) on the A100. |
| date | 2026-07-29 |

### C62 — Learned morphological post-processing from the support masks: NO-GO

| field | value |
|---|---|
| trigger | nnU-Net applies a connected-component post-processing chosen by cross-validation. We have no validation split, but we do have K support masks, so the analogue is to FIT an opening/closing structuring element on them (user direction: "nauč ale vhodný opening/closing kernel", 2026-07-29) rather than derive one. |
| method | `morph_post` flag in `head_fusion_backend.py` (`_fit_morphology` / `_apply_morphology`), screened by `scripts/screen_flag.py`. Off by default. |
| protocol | Fast-screen per CLAUDE.md: 2 TARGET (drive, hrf — where a closing should reconnect broken vessels) + 2 CONTROL (dsb2018, monuseg), 3 seeds, K=8, `--test 8`. |
| launcher | `bash /disk1/prusek/active-segmenter/morph.sh` (output landed in `chain2.log`, not `morph.log` — the chain wrapper owned the pipe). |

| dataset | role | baseline | morph_post | delta | fit speed-up |
|---|---|---|---|---|---|
| drive | TARGET | 0.7789 | 0.7790 | +0.0002 | 1.11x |
| hrf | TARGET | 0.7621 | 0.7747 | **+0.0126** | 0.77x |
| dsb2018 | CONTROL | 0.8583 | 0.8603 | +0.0019 | 1.01x |
| monuseg | CONTROL | 0.6286 | 0.6194 | **−0.0093** | 0.78x |

| field | value |
|---|---|
| **verdict** | NO-GO. hrf gains a real +0.013, but monuseg regresses −0.009, past the −0.005 control gate, and the lever is a net SLOWDOWN on both large-field datasets (0.77–0.78x). One target gain does not buy a control regression — that is the whole point of having controls. |
| why monuseg | The structuring element fitted on eight support masks of densely packed nuclei closes the gaps BETWEEN neighbouring nuclei as readily as inside them. This is the bridging failure mode [[foreground-is-the-bottleneck]] records as costing 2–3x the AP of error placed apart, arriving through a post-processing step instead of through the loss. |
| date | 2026-07-29 |

### C63 — best_v3 instance AP against the specialists: MEASURED, deliberately NOT in the paper

| field | value |
|---|---|
| trigger | A reviewer critique asked for the cheapest defence against "an ISBI cell-segmentation audience will object that you never report an instance metric": run the existing SAM-free affinity-watershed decoder over the foreground map on the datasets with per-instance ground truth and add one row. |
| method | `head_fusion_best_cgate_film_nobank_flat_es_ep500_wd4_do5_mix` (best_v3) with `--metric_override` OMITTED, so the registry's own metric (`instance_ap`) applies instead of foreground IoU. Same fit, same protocol, same support draws as the reported arm -- only the readout differs, which is precisely the claim under test. Specialists were run through `cellpose_stardist_bench.py` / `microsam_bench.py` with `--fg-scoring` DROPPED, into SEPARATE score dirs so the foreground records Table 1 reports could not be overwritten. |
| protocol | K=8, 10 seeds, `--test 10000`, res 672, pool per `run_campaign.pool_for`, cache `/disk1/prusek/cache_final10`. Specialists ignore the support and are deterministic, so each is one run replicated across seeds (`microsam_bench.py` line 177) -- the same convention Table 1 already discloses. |
| launcher | `python scripts/run_final_gap.py --workers "0,0,0,1"` (the `ap` and `specap` bands), score dirs `results/final10/oursv3_ap_k8` and `results/final10/spec_ap_{cellpose_sam,stardist,microsam}`. |

| dataset | ours (best_v3) | Cellpose-SAM | StarDist | micro-SAM |
|---|---|---|---|---|
| DSB2018 | 0.5224 ± 0.017 | **0.6606** | 0.6159 | 0.6500 |
| CTC-U373 | 0.5670 ± 0.015 | 0.6502 | 0.0073 | **0.7279** |
| MoNuSeg | 0.2557 ± 0.019 | 0.3859 | **0.4054** | 0.3852 |

| field | value |
|---|---|
| **finding 1 — the decoder produces real instances, and best_v3 improves them** | CTC-U373 0.567 against best_v2's 0.444 (+0.123) and DSB2018 0.522 against 0.494. The seed spread is tight (±0.015–0.017), so these are not noise. The foreground map IS a usable seed, which is what the paper's "seed for instance post-processing" sentence asserts. |
| **finding 2 — but we trail the specialists on their own datasets** | Behind all three on DSB2018 and MoNuSeg, and two of three on CTC-U373. Only StarDist ever collapses (CTC-U373 0.007 on AP against 0.267 on foreground) -- so "no single specialist spans the morphologies" holds MORE sharply on instances than on foreground, on the metric that favours them. |
| **finding 3 — MoNuSeg is the widest instance gap, and it is the dense-nuclei wall again** | 0.256 against 0.385--0.405, a deficit of 0.13--0.15, far wider than the 0.10--0.16 on the other two. It is the same limit the paper already names on foreground (MoNuSeg 0.639 against Cellpose-SAM's 0.701, six controlled interventions each worth at most 0.01), and it is amplified by the readout: [[foreground-is-the-bottleneck]] measured that at equal foreground IoU, BRIDGING error costs 2--3x the AP of error placed apart, and touching nuclei are where our foreground bridges. So the instance number is not an independent weakness, it is the foreground weakness magnified by a metric that punishes exactly the way we fail. |
| **DECISION (user, 2026-07-29): this does NOT go in the paper** | It would add a second losing axis to a paper that already carries one, which is what the abstract restructure was undoing; it conflicts with the standing semantic-only scope; and it invites "why is the instance decoder not part of your contribution?" about a component the paper deliberately does not claim. |
| why it was still run to completion | ISBI and MICCAI both forbid new experiments in a rebuttal. This is the pre-measured answer to "and instances?", at a cost of ~2 GPU-h for our arm and ~8 min for all nine specialist cells. |
| date | 2026-07-29 |

### C64 — The competitive gate stops earning its place once the head is regularised

| field | value |
|---|---|
| trigger | The ablation re-run in the REPORTED regime (C65, in flight) was meant to correct a regime mismatch. It also reversed a component result. CLAUDE.md records "cgate & film COMPOUND (each alone still regressed drive; together none)" from the base-head factorial; this measures the same subtraction on the lean regularised head. |
| method | `ablv3_film_k8` = FiLM ON, gate OFF, in the reported regime. The cost of turning the gate off is `film_arm − FULL`, so a NEGATIVE number means the gate helps. Compared against `abl_film_k8`, the same arm on the base head. Paired per-image Wilcoxon, seeds collapsed to one score per image first. |
| protocol | K=8, 10 seeds, `--test 10000`, res 672, pool per `run_campaign.pool_for`, cache `/disk1/prusek/cache_final10`. |

| dataset | base head | reported regime | paired p | n |
|---|---|---|---|---|
| DRIVE | −0.0178 | **+0.0029** | 0.00048 | 20 |
| Bacteria (held out) | n/a | **+0.0015** | 0.0027 | 148 |
| MoNuSeg | −0.0101 | +0.0038 | 0.035 | 14 |
| HRF | −0.0100 | −0.0000 | 1.0 | 25 |
| SpheroidJ | −0.0010 | −0.0009 | 0.51 | 24 |

| field | value |
|---|---|
| **finding — the gate is worthless or mildly harmful in the reported regime** | It went from helping on three of four base-head datasets to hurting significantly on DRIVE, Bacteria and MoNuSeg, and to exactly zero on HRF. The magnitudes are small (0.001–0.004) but the direction is reproduced on five datasets and the two strongest tests (DRIVE p=5e-4, Bacteria p=3e-3 at n=148) are the ones that say it hurts. |
| **what it falsifies** | "cgate & film COMPOUND" no longer holds. FiLM survives the regime change nearly intact (it still adds +0.0078 on DRIVE, +0.0060 on HRF); the gate does not. Of the two levers the factorial promoted together, only one earns its place once the head has a flat stem, a plateau stop, weight decay and dropout. |
| mechanism | The gate is a per-pixel softmax over fusion groups, zero-initialised to parity. It buys a DECISION the head could otherwise learn. A regularised head trained to a plateau finds that decision itself, so the gate adds capacity without adding information — the same pattern the re-run shows for the colour rule on HRF and for the adaptive loss on Bacteria. What survives everywhere is the classical bank, which supplies information (native resolution) that no amount of training on patch-16 features can recover. |
| **what this does NOT license** | Removing the gate. It would be the consistent thing to do under the project's own promotion rule, and the method would likely be a hair better without it — but it would invalidate Table 1, the K-scaling figure, the nnU-Net comparison and every ablation arm, all for an effect of 0.001–0.004. Recorded as a finding, deferred as a change. Revisit for the Nature Methods tool, where a simpler pipeline is worth more than it is here. |
| paper consequence | The specific published claim survives: "the full head beats gate-alone by +0.016 on vessels" is `FULL − cgate`, which is still +0.0078 on DRIVE. What must change is the framing of the gate and FiLM as joint contributors, and the number. |
| date | 2026-07-30 |


### C65 — the gate and FiLM are REMOVED from the method; the reported arm becomes `stripv3_k*`

| field | value |
|---|---|
| trigger | C64 recorded that the gate no longer earns its place but deferred the removal as too disruptive. The user overruled that deferral (2026-07-30, *"k čemu to ale je na isbi, když to nefunguje a jen to komplikuje? odeber to"*) and confirmed it after seeing the full curve (2026-07-31, *"odeber to a aktualizuj paper"*). |
| method | `head_fusion_best_nobank_flat_es_ep500_wd4_do5_mix` -- best_v3 with `_cgate` and `_film` dropped. The colour rule is KEPT: the three-way strip was tried and cost a further 0.0147 on DRIVE, and the user withdrew it. |
| protocol | K in {1,4,8,16}, 10 seeds, `--test 10000`, res 672, pool per `run_campaign.pool_for`, cache `/disk1/prusek/cache_final10`, all eleven datasets. Compared seed-paired against `oursv3_k*`/`oursv3n_k8` (the gate+FiLM arm) by Wilcoxon over the eleven per-seed dataset aggregates. |
| launcher | `python scripts/run_final_gap.py --workers 0,0,1` (band `strip`), score dirs `results/final10/stripv3_k{1,4,8,16}`. The seven K=8 datasets first written to `ablv3_bank_k8` were copied into `stripv3_k8` after verifying every record carries the same method string, `support` 8 and `res` 672. |

| K | full (gate+FiLM) | stripped | delta | paired p |
|---|---|---|---|---|
| 1 | 0.7086 | **0.7105** | **+0.0019** | 0.625 |
| 4 | 0.7700 | 0.7664 | −0.0036 | **0.0371** |
| 8 | 0.7915 | 0.7889 | −0.0026 | 0.084 |
| 16 | (running) | (running) | | |

| field | value |
|---|---|
| **finding 1 -- the cost is ~0.003 at K>=4 and nothing at K=1** | K=4 is the one support size where the difference passes the test (p=0.0371, driven by the centreline half at −0.0076). K=8 has a near-identical magnitude and does NOT pass (p=0.084), so this is a difference in test power, not in effect. Report it as "~0.003 at K>=4, concentrated on vessels", never as "free". |
| **finding 2 -- what is bought** | 502,475 -> 358,168 trainable parameters (−29%, 0.166% -> 0.118% of the 303 M frozen backbone), and two components out of the method description. FiLM alone was 144,006 parameters, 29% of the head, for an effect indistinguishable from zero. |
| **finding 3 -- the headline claims strengthen slightly** | Forward-pass wins stay 55/55 and Holm-significant wins rise 52 -> 53. At K=1 the margin over nnU-Net GROWS, +0.0265 -> +0.0284. Panel mean at K=8 falls 0.791 -> 0.789, and the overlap-half margin over nnU-Net falls +0.003 -> +0.001, which is now a tie and is reported as one. |
| **finding 4 -- the self-configuration ablation had to be RE-RUN** | `ablv3_nocls` and `ablv3_bank` carry neither token, so the architecture block already measured the stripped method. The three self-config arms all carried `_cgate` or `_film`, so they ablated a method that no longer exists; re-run as `ablv3s_sc_{noloss,nocolor,none}` (21 cells). `ablv3_cgate` and `ablv3_film` remain valid and are re-read as ADDING each component to the stripped method, so the table can report both as rejected. |
| paper consequence | Table 1, `numbers_stats.tex`, `numbers_probe.tex`, the K-scaling figure, the qualitative dumps, the Overview, the self-configuration paragraph ("three axes" -> two) and eleven inline numbers in the results prose all re-point at the stripped arm. `scripts/audit_paper_numbers.py` was written to recompute those inline numbers from the tree rather than hand-edit them. |
| date | 2026-07-31 |

### C66 — fit and inference cost of the stripped method, MEASURED; two paper claims were wrong

| field | value |
|---|---|
| trigger | The manuscript's cost sentences ("$21$ to $701$ s to fit", "$2$ to $25$ ms to run the head", nnU-Net "about $3260$ s per support set regardless of field size", "$4.7$ to $154$ times less") appeared nowhere in this repo. `measure_fitcost.sh` had been written for exactly this and never run. |
| method | `head_fusion_best_nobank_flat_es_ep500_wd4_do5_mix` (the reported arm) via `scripts/timing_bench.py`, which brackets `be.fit(shots)`, `enc.enc.extract(im)` and `be.foreground(im, g)` with `torch.cuda.synchronize()`. K=8, res 672, **3 repeats per dataset, each drawing a DIFFERENT support subset**, so the spread is over draws and not over repetitions of one draw. |
| conditions | RTX A5000 (GPU1). The campaign cell sharing the card was **SIGSTOPped** for the duration and resumed by an EXIT trap; `nvidia-smi` read 0% utilisation while paused, which is the evidence the device was idle. Pausing rather than killing keeps the hours already spent in that cell. GPU0 stayed busy, so CPU/PCIe contention is not excluded. |
| launcher | `/tmp/timed.sh <pid>` -> `timing_bench.py --datasets dsb2018,monuseg,drive,hrf --support 8 --repeats 3 --res 672`; bank attribution via `/tmp/bank_split.py`; nnU-Net via `/tmp/nntime.sh`. |

| dataset | field px | fit (s) median [min-max over draws] | encode (ms) | image -> mask (ms) | bank alone (ms) |
|---|---|---|---|---|---|
| dsb2018 | 256 | 19.7 [15.9-22.7] | 141 | 168 | 13 |
| DRIVE | 584 | 111.9 [101.3-140.0] | 144 | 248 | 36 |
| MoNuSeg | 1000 | 400.3 [287.2-445.5] | 149 | 850 | 88 |
| HRF | 3504 | 631.9 [605.5-726.1] | 219 | 6618 | 671 |

| field | value |
|---|---|
| **confirmed** | Fit 16-726 s against the claimed 21-701, and encode 141-219 ms against the claimed 150-220. Fit still dominates per-image inference by about two orders of magnitude at every field size. |
| **WRONG 1 -- per-image inference, out by ~250x** | The paper said "$2$ to $25$ ms to run the head". Measured image-to-mask from CACHED features is **168 to 6618 ms**. The old figure timed the head at FEATURE resolution while the model emits a NATIVE-resolution mask. |
| **WRONG 2 -- "inference is bounded by the frozen encoder"** | False. On HRF inference is 6618 ms against 219 ms to encode, 30x the encoder. The prior bank is only about a tenth of it (671 ms); the remainder is the native-resolution upsampling and readout. "The rest of a dataset is close to free" is therefore also wrong: a 45-image HRF set costs ~5 min after the fit. |
| **WRONG 3 -- nnU-Net's cost and its field-size independence** | Measured on the same idle A5000, one support set, K=8, 100 epochs: dsb2018 **2257 s** (iou 0.8543, reproducing the campaign's 0.851, so the run is valid) and HRF **>9000 s** -- it did not finish inside the cap, so that is a LOWER bound. The claimed 3260 s "regardless of field size" is wrong in both the value and the invariance: the epoch and iteration budget is fixed but preprocessing and sliding-window inference scale with the field. Ratio is therefore **14x to 115x**, not 4.7x to 154x. |
| **draw-to-draw spread is real** | MoNuSeg 287-446 s across three different support draws, a 1.55x range. The plateau stop means WHICH masks are annotated changes how long the fit runs -- relevant to [[ultimate-goal-sub-second-fit]], where a worst-case draw is what a latency budget must survive. |
| consequence for the sub-second goal | Even a free fit would not make HRF interactive: 6.6 s per image from cached features is already over the budget. The bottleneck is not only the fit. |
| open | HRF's nnU-Net cost is a lower bound; re-run without the 9000 s cap to close it. |
| date | 2026-07-31 |

### C67 — PerSAM: MEASURED on the full panel, deliberately NOT a Table 1 column

| field | value |
|---|---|
| trigger | The working tree carried an uncommitted change adding a PerSAM column to Table 1 (a `persam()` better-of-two-variants helper plus a `GROUPS` entry), while the paper's prose had gone the other way and dropped PerSAM entirely. Whoever regenerated Table 1 next would have got a twelfth column no sentence in the paper explains. |
| method | PerSAM and PerSAM-F, both published configurations, both anchors reproduced first (PerSeg mIoU 89.32 vs 89.3 published, and 95.18 vs 95.3). Same ten seeds, same support draws, same protocol as every other forward-pass column. |
| protocol | K=8, 10 seeds, res 672, pool per `run_campaign.pool_for`, score dir `results/final10/persam_k8` (records `persam__*.json` and `persam_f__*.json` side by side). |

| dataset | ours | PerSAM | PerSAM-F | Matcher (K=1) |
|---|---|---|---|---|
| SpheroidJ | 0.917 | 0.764 | 0.738 | 0.830 |
| Decay | 0.795 | 0.332 | 0.395 | 0.225 |
| DSB2018 | 0.848 | 0.056 | 0.202 | 0.206 |
| MoNuSeg | 0.637 | 0.004 | 0.031 | 0.216 |
| CTC-U373 | 0.800 | 0.188 | 0.268 | 0.570 |
| BBBC010 | 0.610 | 0.076 | 0.075 | 0.092 |
| Bacteria | 0.917 | 0.227 | 0.288 | 0.431 |
| DRIVE | 0.762 | 0.036 | 0.148 | 0.178 |
| HRF | 0.728 | 0.036 | 0.089 | 0.139 |
| ISBI2012-EM | 0.920 | 0.326 | 0.314 | 0.349 |
| FISBE | 0.742 | 0.154 | 0.222 | 0.201 |

| field | value |
|---|---|
| finding | We lead on **all eleven** against the better variant, so the abstract's "leads on all eleven datasets against every forward-pass in-context method" holds whether or not PerSAM is printed. Nothing about the claim depends on excluding it. |
| **DECISION (2026-08-01): measured, not printed** | Four reasons, in order of weight. (1) It is the WEAKEST baseline on the panel -- a better-variant mean near 0.25, below Matcher's 0.312, with cells at 0.004 and 0.036 -- so the column would add eleven more wins over a method never designed for dense multi-object fields. A table padded with methods aimed at a different problem reads as cherry-picking and *lowers* the credibility of the win count it inflates. (2) It is REDUNDANT with Matcher: same family (one-object SAM personalisation), same failure mode, and Matcher already carries that argument with its dagger and the "accuracy tracks object density" sentence. A second data point for an argument already made is not a new argument. (3) The page budget is at exactly zero -- 5 pages with page 5 restricted to ethics/acknowledgments/references -- and a twelfth column plus the sentence that would have to describe it risks technical content on page 5, which is an automatic ISBI reject. (4) `make_stats.py` keeps its OWN `FORWARD_PASS` list, so adding the column without touching it would make Table 1 and the headline `\fpWins` disagree about who counts as forward-pass. |
| why it was run to completion anyway | Same reason as [[C63]]: ISBI and MICCAI both forbid new experiments in a rebuttal, so "why no PerSAM?" needs a pre-measured answer. This table is it. |
| revert | `git checkout scripts/make_semantic_tables.py` -- the uncommitted diff was entirely PerSAM (helper, dispatch, `GROUPS`, `CITE`, caption clause). |
| date | 2026-08-01 |

### C68 — the forward-pass statistics were computed against a WEAKER INSID3 than Table 1 prints

| field | value |
|---|---|
| trigger | Found while correcting the paper's prose description of INSID3's conditional-random-field handling. Table 1 shows INSID3 at the per-dataset better of its two documented modes (`make_semantic_tables.insid3()`), but `make_stats.py` hard-coded `"INSID3": "insid3_guided_k8"` -- a single mode. The headline win count was therefore earned against a different configuration from the one printed beside it. |
| why it matters | Not cosmetic: the two modes differ enormously and in both directions. Dense refinement erases structures narrower than its kernel (DRIVE 0.005 against guided's 0.279, HRF 0.045 against 0.224, Bacteria 0.019 against 0.730) yet WINS on Decay, MoNuSeg and FISBE. Testing against guided alone meant three datasets were compared against the weaker mode -- in our favour, which is the direction a reviewer reads as deliberate. |
| fix | `FORWARD_PASS` values may now be a tuple of directories; `arm_scores()` selects the per-dataset better one, on the same quantity the table bolds, so a test and a printed number can no longer describe different configurations of one baseline. |
| **result: NO reported number changes** | Regenerated `numbers_stats.tex` is byte-identical: 55 comparisons, ours ahead on 55, 53 Holm-significant; worst adjusted p 0.9; nnU-Net ahead on 9, 6 significant, ours significantly ahead on 1. We beat both modes on all three affected datasets by a wide margin, so the correction is purely one of provenance. |
| why fix it anyway | The code is released with the paper. A reviewer who runs it and finds the statistic computed against a baseline variant the table does not show has found a discrepancy that costs more trust than the zero accuracy it was worth. |
| date | 2026-08-01 |

### C69 — the input-resolution control, re-run on the REPORTED method and on all eleven datasets

| field | value |
|---|---|
| trigger | The paper's answer to "your margin is just more pixels" rested on `results/rescontrol/ours448`, which is **best_v2** -- two method generations back. The text disclosed this as "(measured on the previous head)", so it was honest, but it was the last number in the paper describing an arm other than the reported one, and a control measured on a method the paper does not report is dismissible in one line. User instruction 2026-08-01: fix it and re-run. |
| **second finding, found while reproducing it** | The quoted "five-dataset mean of $-0.002$" is the six datasets `ours448` holds **minus FISBE**, and no record says why. Over all six the delta is **$+0.002$** -- at 448 the method is marginally BETTER. The reported subset therefore *understated our own control*: the excluded dataset was the one most favourable to us (FISBE $+0.023$). Conservative, so not a fairness problem, but "a five-dataset mean" with no statement of which five is exactly what a reviewer probes. |
| best_v2, 448 vs 672, per dataset | spheroidj $-0.014$, dsb2018 $-0.004$, monuseg $+0.001$, drive $+0.001$, hrf $+0.005$, fisbe $+0.023$; six-dataset mean $+0.0020$, five-dataset (drop fisbe) $-0.0022$ |
| method | `head_fusion_best_nobank_flat_es_ep500_wd4_do5_mix` (the reported arm) at `--res 448`, the smallest input any baseline uses (SegGPT's native size). |
| protocol | IMPORTED from `run_campaign` rather than restated -- seeds 10, `--test 10000` (full split), `--pool` per `pool_for`, K=8, and the clDice/fg_iou metric-override convention. The only intentional difference from the campaign is the resolution. All **eleven** datasets, so the control covers the same panel Table 1 reports and the unexplained subset disappears. |
| launcher | `python scripts/run_rescontrol.py --gpu 0` (smoke: `--smoke`), score dir `results/rescontrol/stripv3_448`, log `res448.log` |
| cache | SEPARATE (`/disk1/prusek/cache_res448`). The campaign cache is prebuilt at 672 and read-only by convention; this run misses on every entry and would write. `EmbeddingCache` keys on image bytes plus the full encoder config so a collision is impossible, but CLAUDE.md's rule is that concurrent benchmark processes never share a writable cache, and a cache of our own is cheaper than an argument about why sharing would have been safe. |
| smoke | dsb2018, 1 seed: exit 0 in 131 s, fg_iou 0.842 at 448 against 0.848 at 672 -- plausible, so the command and the metric override are right before committing hours. |
| **cost: the pre-launch estimate was wrong by ~5x, and in the instructive direction** | Estimated "~6 GPU-h" on the assumption that a smaller input is cheaper. It is NOT. The fine pathway tiles each image into ENCODER-RESOLUTION windows, so shrinking the encoder input from 672 to 448 *increases* the tile count for the same native field -- rozpad at 2048 px needs about 5x5 windows at 448 against 3x3 at 672. Measured: spheroidj alone took 7359 s (2.04 h) against the 74 min the same dataset costs at 672, a factor of 1.65, and rozpad ran over 7 h. Eleven datasets is realistically 30-50 h of GPU time, not 6. **A resolution control at a lower input is more expensive than the run it controls** -- the opposite of the intuition, and worth remembering before the next one is scheduled. |
| consequence | It was competing with the nnU-Net band, which is the paper's critical path: nnU-Net measured 2.14 seeds/h (33/210 after 15.5 h) with this job holding 16.3 GB and a large share of GPU0. Suspension proposed to the user 2026-08-02 so nnU-Net gets the card back; the cell in flight keeps its progress under SIGSTOP because a record is only written when a whole cell ends. |
| **RESULT (complete 2026-08-02, 11/11)** | Panel mean 672 **0.7889** against 448 **0.7851**, delta **-0.0038**. Paired per-image Wilcoxon on identical support draws and test slices. |

| dataset | metric | 672 | 448 | delta | p |
|---|---|---|---|---|---|
| SpheroidJ | fg_iou | 0.9169 | 0.8986 | -0.0183 | 0.0018 |
| Decay | fg_iou | 0.7954 | 0.7950 | -0.0004 | 0.94 |
| DSB2018 | fg_iou | 0.8481 | 0.8455 | -0.0026 | 0.16 |
| MoNuSeg | fg_iou | 0.6374 | 0.6424 | +0.0050 | 0.22 |
| CTC-U373 | fg_iou | 0.7999 | 0.7741 | **-0.0259** | 2.7e-05 |
| BBBC010 | fg_iou | 0.6104 | 0.6099 | -0.0005 | 0.94 |
| Bacteria | fg_iou | 0.9171 | 0.9134 | -0.0037 | 0.011 |
| DRIVE | cldice | 0.7616 | 0.7590 | -0.0026 | 0.0004 |
| HRF | cldice | 0.7285 | 0.7369 | **+0.0084** | 1.2e-07 |
| ISBI2012-EM | cldice | 0.9198 | 0.9137 | -0.0061 | 0.0009 |
| FISBE | cldice | 0.7424 | 0.7474 | +0.0050 | 0.67 |

| field | value |
|---|---|
| **the split matters and favours us** | overlap (7 datasets) **-0.0066**, centreline (4) **+0.0012**. At the baselines' input the method is marginally BETTER on the four datasets where its margin over forward-pass methods is largest. The "your margin is just more pixels" objection is therefore dead exactly where it would have bitten hardest. |
| worst case | CTC-U373 -0.026: large low-contrast phase-contrast cells, where halving the grid costs real context. Worth quoting alongside the mean rather than hiding in it. |
| supersedes the paper's old sentence | "-0.002 on a five-dataset mean (measured on the previous head)" becomes "-0.004 over all eleven datasets" on the arm the paper actually reports. Larger in magnitude, incomparably better in provenance, and no unexplained subset. |
| paper wiring | DEFERRED to the single final update after the nnU-Net band lands (user instruction 2026-08-02), so the text is touched once rather than twice. |
| status | COMPLETE. 11/11 cells, 2026-08-01 15:31 to 2026-08-02 20:00. |
| date | 2026-08-01, cost corrected 2026-08-02 |

### C70 — method FROZEN for the ISBI paper; the AB lever is parked for the next one

| field | value |
|---|---|
| **DECISION (user, 2026-08-01)** | "už mojí best metodu nevylepšuj -- paperovou metodu necháme tak, jak je a další vylepšení si necháme do dalšího paperu." The reported arm stays `head_fusion_best_nobank_flat_es_ep500_wd4_do5_mix`. No further lever may change it before submission. |
| what this parks | The **AB** lever (support-derived logit threshold `taucal` + morphology-oriented bank `morphbank`), which PASSED its screen: A alone $+0.0035$, B alone $+0.0025$, **AB $+0.0060$** with hrf $+0.0130$, interactions all under $|0.0016|$ (additive). The full panel (`results/final10/abv3_k8`, ~13 GPU-h) and its ablation arms (~12 GPU-h) were queued behind nnU-Net and are now cancelled for this paper. |
| why parking is right | Folding AB in would change the reported method, which regenerates Table 1, Table 2, the statistics, the K-scaling figure and every number in the prose -- for $+0.006$, roughly a seventh of the centreline deficit against nnU-Net, and after the nnU-Net band has already been spent measuring the current arm. The gain does not change any claim the paper makes. |
| what remains to do for the paper | Only: (a) nnU-Net K=4/K=16 to finish, then the wiring in `docs/ISBI-SUBMISSION-TODO.md`; (b) this resolution control ([[C69]]); (c) author-only items (co-author review, funding/COI, IEEE PDF eXpress, official template, repo re-sync). |
| date | 2026-08-01 |

### C71 — Closed-form PoC: the resolution mismatch is not a wall, and the LEARNED STEM is a liability

| field | value |
|---|---|
| trigger | User asked what is needed to fit every parameter in closed form at the reported accuracy, then objected — correctly — that DINO on a 42x42 patch grid and 35 priors at native resolution cannot be concatenated: X is 2083 channels x 8.2M px = 63.5 GB per HRF image, times eight. |
| stage 1 (`scripts/poc_closed_form_cost.py`) | Ridge needs X'X, not X. With U the bilinear upsampling operator and X = [UD \| B], native resolution enters only through U'U, U'B and B'B, so DINO is never upsampled: U' is the ADJOINT of upsampling, i.e. 35 prior channels come DOWN to the grid instead of 2048 feature channels going UP. |

| route | dsb2018 (0.4 Mpx) | monuseg (1.0) | rozpad (4.2) | hrf (8.2) |
|---|---|---|---|---|
| naive | 9.1 s / 11.6 GB | **OOM** | **OOM** | **OOM** |
| streamed tiles | 0.9 s / 7.4 GB | 2.5 s / 7.4 GB | 10.0 s / 7.8 GB | 19.3 s / 8.3 GB |
| **factored** | **0.19 s / 288 MB** | **0.29 s / 326 MB** | **0.63 s / 758 MB** | **1.34 s / 1.27 GB** |

Exact to 2e-9 against naive. X'X is 33 MB at every size; Cholesky at d=2083 is 0.042 s. HRF K=8 fit
would be ~11 s against the measured 632 s (registry C66), i.e. ~57x. Of the factored route's 1.27 GB,
1.15 GB is the bank tensor the method already pays for.

| stage 2 (`scripts/poc_closed_form_ladder.py`) | 3 seeds, 12 test images, K=8, res 672, on kajman L40S. Each rung changes ONE thing; all scored on the same test images. |
|---|---|

| rung | what it isolates | monuseg (fg_iou) | drive (clDice) |
|---|---|---|---|
| A full trained head | reference | 0.6357 | 0.7729 |
| B closed form over the LEARNED 99-ch stem | the solver | 0.6253 (−0.0104) | 0.5554 (**−0.2175**) |
| C closed form over RAW 2083, no stem | the stem | 0.6184 (−0.0173) | 0.6666 (−0.1063) |
| D = C + support-swept threshold | the decision rule | 0.6255 (−0.0102) | 0.7062 (**−0.0666**) |

| field | value |
|---|---|
| **finding 1 — the learned stem HURTS a closed-form readout** | On drive, C beats B by **+0.111**: removing the stem entirely is better than re-solving over it. The stem is trained jointly with its classifier under the topology-aware loss, so its 99-dim bottleneck is optimised to be read by THAT classifier and has discarded what a plain logistic ridge would use. This kills "meta-train the stem, solve the classifier" from the other side than C57 did: even the task's OWN stem is the wrong basis. The right closed-form design has no stem. |
| **finding 2 — the decision rule is worth more than the representation on thin structures** | C to D is +0.0396 on drive with tau* = −1.0 reproduced on all three seeds, confirming C59 in the full pipeline rather than on LOO folds. On monuseg it is +0.0071. |
| **finding 3 — what remains is topology** | After both fixes: −0.010 on overlap, −0.067 on centreline. The residual is the gap between a per-pixel logistic objective and the clDice-weighted loss the trained head optimises — the same per-pixel-decomposability wall the paper's conclusion names, reached from the estimator side. |
| revises C55-C59 | Those concluded a linear readout on frozen features "is not competitive". They tested features with no per-task training, or the learned bottleneck. Raw full-dimensional features plus a swept threshold reach within 0.067 on the hardest centreline dataset — two thirds of the gap the body-transfer screen showed closes. |
| honest limits | 3 seeds, 12 test images, TWO datasets. Not a panel. hrf/isbi2012em/fisbe untested, and drive is the easier of the vessel pair. Naive panel extrapolation would be ~0.758 against 0.789, consistent with [[ultimate-goal-sub-second-fit]]'s 0.74–0.76 but at its top. |
| compute | kajman (2x L40S, idle) with `ASG_DATA_ROOT=/data/prusek`, cache `/scratch/prusek/cache_final10`. Tulen was left entirely to the nnU-Net band. |
| date | 2026-08-02 |

### C72 — The fully closed-form arm on the panel: −0.030 at K=8, and level with nnU-Net on overlap

| field | value |
|---|---|
| what | `scripts/closed_form_bench.py`: ridge/IRLS over the raw frozen sources (coarse DINO 1024 ⊕ fine DINO 1024 ⊕ 35 native priors = 2083 channels), **no stem, no gradient step anywhere**. The run aborts if a head object is ever built, so gradient-freedom is asserted rather than argued. Design follows [[C71]]: no stem (solving over the learned 99-ch bottleneck is WORSE), λ by leave-one-support-image-out (C56 measured a bad λ at up to 0.057), threshold fixed at 0.0 — identical to the gradient arm's `_tau`, so this measures the ISOLATED cost of the substitution rather than what closed form could reach with extra levers. |
| protocol | 11 datasets, 10 seeds, K∈{1,8}, res 672, full test splits, campaign `pool_for`, campaign metric-override convention, campaign record contract incl. split fingerprint. Run on kajman (2× L40S) with `ASG_DATA_ROOT=/data/prusek`, leaving tulen entirely to the nnU-Net band. |

| K=8 | panel | overlap (7) | centreline (4) |
|---|---|---|---|
| gradient (reported method) | 0.7889 | 0.7893 | 0.7881 |
| **closed form** | **0.7585** | 0.7700 | 0.7383 |
| nnU-Net | 0.8035 | 0.7884 | 0.8298 |
| **cost of going closed form** | **−0.0304** | −0.0193 | −0.0498 |
| closed form vs nnU-Net | −0.0450 | **−0.0184** | −0.0915 |

| K=1 | panel | overlap (7) | centreline (4) |
|---|---|---|---|
| gradient | 0.7105 | 0.7009 | 0.7274 |
| **closed form** | **0.6631** | 0.6611 | 0.6666 |
| nnU-Net | 0.6821 | 0.6422 | 0.7521 |
| **cost of going closed form** | **−0.0475** | −0.0398 | −0.0608 |
| closed form vs nnU-Net | −0.0190 | **+0.0189** | −0.0855 |

| field | value |
|---|---|
| **finding 1 — the cost is concentrated, not spread** | At K=8 eight of eleven datasets cost under 0.021 (spheroidj −0.004, ctc_u373 −0.008, dsb2018 −0.008, monuseg −0.012). Three carry almost all of it: drive −0.082, bbbc010 −0.063, hrf −0.063. |
| **finding 2 — on overlap, a gradient-free solve matches a fully trained nnU-Net** | −0.018 at K=8 and **+0.019** at K=1, over the seven overlap datasets, from eight (or one) masks with no training. The deficit against nnU-Net is entirely the four centreline datasets (−0.092 / −0.086). |
| **finding 3 — the K=1 λ fix, and the number it invalidated** | The first K=1 run had NO λ selection: leave-one-support-image-out has nothing to hold out at K=1, so it used a constant 1e-3. That gave panel 0.6059. Selecting λ on spatial QUADRANTS of the single image (still closed form, still test-blind) gives **0.6631, +0.057** — almost exactly the 0.057 C56 predicted for a badly chosen λ, and on monuseg alone +0.205 (0.374→0.579). The quadrant rule is biased low (adjacent pixels correlate, so a held-out quadrant is easier than a held-out image); it is recorded in each record's `lam_rule`. |
| **finding 4 — the swept threshold, complete: +0.009 (K=8) and +0.013 (K=1) on the panel** | It pays on exactly the three datasets where the fixed cut was worst — bbbc010 **+0.045**, hrf **+0.044**, drive **+0.029** — is under ±0.001 on seven others, and HURTS fisbe (**−0.020**). A coherent pattern rather than noise: where foreground is sparse and structure thin, the calibrated cut sits below 0.5. Best closed-form arm is therefore **0.7675 at K=8** (−0.021 vs the gradient method, −0.036 vs nnU-Net) and **0.6764 at K=1** (−0.034 vs gradient, **−0.006 vs nnU-Net**, and **+0.029 above nnU-Net on overlap**). |
| **THE THRESHOLD LITERATURE IN THIS REGISTRY DOES NOT AGREE WITH ITSELF** | C59 measured the sweep on Dice (+0.123 drive). C60 measured it on clDice and got +0.003, calling C59's positive dead. C61 then declared C60's BASELINE wrong and restated the sweep at +0.113 on drive. This entry, on the full panel at ten seeds with full test splits, measures **+0.029**. Four measurements of one quantity, no two agreeing. The three earlier ones were all leave-one-support-image-out folds at one seed; this is the only one at panel scale, so it is the one to trust — but the spread itself is the finding, and it says single-seed LOO screens of this quantity are not reliable. |
| honest limits | Fit cost at K=8 is 96–98 s against the gradient arm's 112 s on drive: the λ-LOO (40 IRLS fits) dominates and eats nearly all of the speed-up, so the closed-form arm is ~1.15x faster, NOT the 7x measured without λ selection or the 57x the Gram microbenchmark suggests. At K=1 a fit is 35–47 s. Attacking that cost (coarser λ grid, warm-started path, fewer Newton steps inside the LOO) is untried. |
| **what it means for the paper's claim** | The centreline deficit against nnU-Net is −0.078 at K=8 and −0.067 at K=1 — essentially the SAME at both support sizes, while the overlap deficit vanishes. A gap that does not shrink with eight times the data is not a data problem; it is the objective. A per-pixel logistic loss cannot express connectivity no matter how many masks it is given, which is [[worst-link-loss]]'s premise arrived at from the estimator side. |
| date | 2026-08-03 |

### C73 — Is the prior bank's gain predicted by object thickness over patch stride? DIRECTION YES, MECHANISM NOT SUPPORTED

| field | value |
|---|---|
| trigger | A reviewer-style critique proposed that Table 2's largest row hides a law: DINOv3 resolves nothing narrower than a patch, the bank's filters run at native resolution, so the bank's contribution should fall as objects thicken relative to the grid ($\bar\rho/\mathrm{stride}$). If monotone, an ablation row becomes a predictive statement. |
| method | `scripts/probe_bank_law.py`. Descriptors from the model's OWN `_mask_descriptors` on the campaign's fixed support pool, so the predictor is the same $\bar\rho$ the adaptive loss reads. $\Delta_{\mathrm{bank}}$ = `stripv3_k8` − `ablv3_nocls_k8`. CPU only. |
| **the confound this had to handle** | The panel scores seven datasets by foreground IoU and four by centreline Dice, and the metric switches on EXACTLY the axis under test — thin structures get clDice. A pooled correlation across all eleven cannot be told apart from "clDice responds to the bank more than IoU does", so every relationship is reported WITHIN metric. Only 7 of 11 datasets have the no-bank arm (it runs on the ablation set), giving n=5 for fg_iou and n=2 for clDice. |

| dataset | metric | side | $\bar\rho$ | stride | $\bar\rho$/stride | full | no-bank | $\Delta$ |
|---|---|---|---|---|---|---|---|---|
| spheroidj | fg_iou | 980 | 41.17 | 23.3 | 1.76 | 0.917 | 0.915 | +0.002 |
| ctc_u373 | fg_iou | 520 | 11.46 | 12.4 | 0.93 | 0.800 | 0.773 | +0.027 |
| dsb2018 | fg_iou | 301 | 4.59 | 7.2 | 0.64 | 0.848 | 0.791 | +0.057 |
| monuseg | fg_iou | 1000 | 4.83 | 23.8 | 0.20 | 0.637 | 0.517 | +0.120 |
| bacteria | fg_iou | 600 | 4.36 | 14.3 | 0.31 | 0.917 | 0.787 | +0.131 |
| drive | cldice | 565 | 1.60 | 13.5 | 0.12 | 0.762 | 0.517 | +0.245 |
| hrf | cldice | 2336 | 3.65 | 55.6 | 0.07 | 0.728 | 0.603 | +0.126 |

| field | value |
|---|---|
| result (within fg_iou, n=5) | Spearman($\Delta$, $\bar\rho$/stride) = **−0.900**; one-tailed p = 0.042, two-tailed 0.083. Direction as predicted, magnitude strong, significance marginal at n=5. |
| **finding — the stride term does NO work** | Spearman($\Delta$, $\bar\rho$ alone) is **also −0.900**, identical. Dividing by the patch stride adds nothing, and the stride is the entire content of the proposed law. Without it the claim reduces to "the bank helps thin objects", which is close to a restatement of what a Frangi filter is — not a mechanism worth a discussion section. |
| **counterexample in the clDice pair** | HRF has $\bar\rho$/stride = 0.07, the most sub-patch value on the panel, and gains **0.126**; DRIVE at 0.12 gains **0.245**. The wrong way round. n=2, so it refutes nothing, but it warns that this axis is not the only one acting. |
| what would decide it | Varying the STRIDE at fixed $\bar\rho$: run `ablv3_nocls` at 448 as well as 672 and compare $\Delta_{\mathrm{bank}}$ within each dataset. That is a PAIRED test inside a dataset rather than a five-point correlation across them, and it is the only design that separates "thin objects" from "sub-patch relative to the grid". Not run. |
| verdict | Not a law yet. Do not build a paper section on it without the stride-variation experiment. |
| date | 2026-08-02 |

### C74 — nnU-Net K-scaling band: K=4 COMPLETE, and the crossover is now a located number

| field | value |
|---|---|
| what | `scripts/run_final_gap.py --only nn --workers 0,0,1,1` on tulen, log `gap_nn.log`. 21 cells (K=4 on 11 datasets, K=16 on 10 — ctc_u373's pool of 15 cannot supply 16), 210 seed-trainings. nnU-Net v2, 2D config, 100 epochs, ResEncUNetPlanner, trained from scratch on each support set under our own protocol. |
| durability fix that made it survivable | `nnunet_bench.py` now flushes the record after EVERY seed and reconstructs partial state from it, so a kill costs at most one seed. The previous version wrote once per cell and an interrupted relaunch silently discarded ~8 GPU-h of completed trainings. `scripts/nn_status.py` reports at seed granularity for the same reason: a directory listing reads 0/21 for the first hours of every cell and cannot tell "running" from "wedged". |
| launched | 2026-08-01 09:07. K=4 complete 2026-08-03; K=16 in progress. Measured rate 2.1–2.9 seeds/h depending on what else shared the GPUs. |

| K=4 | ours (`stripv3_k4`) | nnU-Net | delta |
|---|---|---|---|
| panel (11) | 0.7664 | 0.7698 | **−0.0033** |
| overlap (7) | 0.7631 | 0.7472 | **+0.0159** |
| centreline (4) | 0.7722 | 0.8093 | −0.0371 |

Per dataset we lead on spheroidj (+0.210, p=1.2e-07) and fisbe (+0.003, n.s.); nnU-Net leads elsewhere, largest on hrf (−0.070) and drive (−0.054).

| field | value |
|---|---|
| **the crossover, now located** | Panel delta against nnU-Net: **K=1 +0.028, K=4 −0.003, K=8 −0.015**. The two methods cross between ONE and FOUR masks, and nnU-Net's lead grows with K. The paper currently states this as an interval between one and eight; it becomes a number, and \cref{fig:kscale}'s caption — which says nnU-Net is drawn "only at the two support sizes it was run at" — must be rewritten. |
| the split is stable across K | Overlap: +0.059 (K=1), +0.016 (K=4), +0.000 (K=8). Centreline: −0.025, −0.037, −0.042. We hold overlap at every support size and lose centreline at every support size. The panel mean is the average of two opposite trends, which is why the paper splits it. |
| date | 2026-08-03 |

### C75 — Closed form vs the gradient fit, TIMED: the speed-up grows with field size, and λ-selection eats it on small ones

| field | value |
|---|---|
| trigger | Three incompatible speed numbers were in circulation from this investigation — 7.1x, 1.15x, and 57x from a Gram microbenchmark that omitted the bank and the feature gathering. None was an answer, because each came from a different machine, contention level and configuration. |
| method | `scripts/closed_form_timing.py`. Both paths back to back on ONE idle L40S (kajman), same starting point (support examples in hand) to the same end point (a model that can segment), K=8, 2 seeds, median. Three configurations: the gradient `fit()`; the closed-form arm as benchmarked in [[C72]] (λ by leave-one-support-image-out); and the same solve with λ pinned, which is NOT deployable on its own but isolates how much of the cost is the selection. |

| dataset | field | gradient (s) | cf + λ-LOO (s) | cf fixed λ (s) | ×(LOO) | ×(fixed) |
|---|---|---|---|---|---|---|
| dsb2018 | 520×696 | 17.0 | 87.8 | 8.7 | **0.19×** | 1.96× |
| drive | 584×565 | 65.2 | 92.7 | 9.3 | 0.70× | 6.99× |
| monuseg | 1000×1000 | 253.4 | 68.6 | 9.9 | **3.70×** | **25.65×** |
| hrf | 2336×3504 | 367.9 | 115.4 | 36.1 | 3.19× | 10.20× |

| field | value |
|---|---|
| **the shape of the answer** | The closed-form cost is nearly FLAT in field size (69–115 s with λ-LOO, 9–36 s without), because it works on a fixed 20 000 sampled pixels × 2083 dims per support image. The gradient cost SCALES with the field (17 s → 368 s). So the speed-up is not a constant: it crosses 1× at roughly a 600–1000 px field and reaches 3.7× at the top. On the smallest dataset the closed form is **5× SLOWER**. |
| **λ selection is the whole story on small fields** | The LOO column costs 60–105 s more than the fixed column, and that overhead is field-independent: 40 IRLS fits (8 folds × 5 λ) over 2083 dimensions. On dsb2018 that fixed overhead is six times the entire gradient training. Attacking it — 3 λ instead of 5, 3 folds instead of 8, a tighter Newton cap inside the LOO — is arithmetic, not research, and would move the LOO column most of the way to the fixed one. Untried. |
| honest caveat | The gradient times here (drive 65 s, monuseg 253 s, hrf 368 s) are BELOW C66's A5000 measurements (112 / 400 / 632 s) because an L40S is faster. The ratios are same-device and therefore valid; the absolute seconds are not comparable across the two entries. |
| relation to the Gram microbenchmark | [[C71]] stage 1 measured the factored Gram at 1.34 s for an HRF-sized field, implying ~11 s for a K=8 fit. The real path costs 36 s with λ pinned, because it also pays for the classical bank on eight support images and the feature gathering. The microbenchmark was right about the algebra and silent about everything else. |
| date | 2026-08-03 |

### C76 — Where a closed-form fit actually spends its time: λ selection is 57–93% of it

| field | value |
|---|---|
| trigger | [[C75]] measured the closed-form fit at 64–115 s but not where it goes. "Optimise the closed form" is not actionable until the stages are separated, and two suspects pull opposite ways: the λ-LOO is field-INDEPENDENT, the classical bank is field-DEPENDENT. |
| method | `scripts/closed_form_profile.py`, K=8, one idle L40S, six stages timed per support image where that is the natural unit. |

| dataset | field | encode | bank | fine | gather | **λ** | solve | total |
|---|---|---|---|---|---|---|---|---|
| dsb2018 | 520×696 | 0.0 | 0.4 | 0.5 | 0.6 | **80.2** | 4.4 | 86.1 |
| drive | 584×565 | 0.0 | 0.5 | 0.5 | 0.5 | **83.3** | 5.5 | 90.3 |
| monuseg | 1000×1000 | 0.0 | 1.2 | 1.9 | 0.6 | **56.9** | 3.2 | 63.9 |
| hrf | 2336×3504 | 0.1 | 9.7 | 17.8 | 0.5 | **67.0** | 4.6 | 99.8 |

| field | value |
|---|---|
| **λ selection dominates at every field size** | 57–83 s, i.e. **93% of the fit on small fields and 67% on the largest**. It does NOT scale with the image — it works on a fixed 20 000-pixel sample — so its share is largest exactly where the gradient arm is cheapest, which is why [[C75]] measured 0.19× on dsb2018. Its variation (57 vs 83 s) tracks CONDITIONING, not field: better-conditioned Grams converge inside the 15-step Newton cap sooner. |
| feature construction grows with the field, and only there does it matter | bank + fine is 0.9 s on dsb2018 but **27.5 s on HRF** (9.7 + 17.8). So the floor after λ is removed is ~6 s on small fields and ~33 s on HRF. A mid-course claim in this session that "the remainder is feature construction" was right for HRF and wrong for the small fields, where the 3–6 s IRLS solve is the larger part. |
| the solve itself is never the problem | 3.2–5.5 s at every size, flat. 160 000 rows × 2083 dims × 15 Newton steps is ~1e13 FLOPs, which is seconds on this hardware. |
| **projection for λ-by-eigendecomposition** ([[lambda-by-eigendecomposition]]) | Removing the λ stage leaves 5.9 / 7.0 / 6.9 / 32.7 s against the gradient arm's 17.0 / 65.2 / 253.4 / 367.9 s, i.e. roughly **3× / 9× / 36× / 11×**. To be MEASURED once implemented, not quoted from here. |
| date | 2026-08-03 |

### C77 — nnU-Net K-scaling band COMPLETE: the crossover is located and the overlap lead is bounded

| field | value |
|---|---|
| what | The full band, 21 cells × 10 seeds = **210/210**, every cell verified at exactly ten seeds. `run_final_gap.py --only nn --workers 0,0,1,1` on tulen, nnU-Net v2 2D / 100 epochs / ResEncUNetPlanner, trained from scratch on each support set under our own protocol. |
| wall clock | launched 2026-08-01 09:07, finished 2026-08-05 ~09:10. Measured rate 1.8–2.95 seeds/h depending on contention and on K (a K=16 seed trains on four times the data of a K=4 one). |
| **interrupted once** | tulen **rebooted at 09:58 on 2026-08-04** (admin, kernel update; `last` shows root 09:16–09:54 then `system boot`). The per-seed flush held: 162 seeds survived and nothing was lost. Two resume defects surfaced and were fixed — see below. |

| K | ours | nnU-Net | delta | overlap (7) | centreline (4) |
|---|---|---|---|---|---|
| 1 | 0.7105 | 0.6821 | **+0.0284** | **+0.0588** | −0.0247 |
| 4 | 0.7664 | 0.7698 | −0.0033 | +0.0159 | −0.0371 |
| 8 | 0.7889 | 0.8035 | −0.0146 | +0.0009 | −0.0418 |
| 16 | 0.7983 | 0.8201 | −0.0218 | −0.0024 | −0.0509 |

| field | value |
|---|---|
| **the crossover is located** | Between **one and four masks**. The paper stated it as an interval between one and eight; it is now a number, which is what a reader deciding how many polygons to draw actually needs. |
| **the overlap lead is bounded, and we say so** | +0.059 → +0.016 → +0.001 → −0.002. Our advantage on the seven overlap datasets erodes monotonically and is gone by sixteen masks. A reviewer reads that straight off the figure, so the results paragraph now states it. |
| the centreline deficit widens monotonically | −0.025 → −0.037 → −0.042 → −0.051. Consistent at every support size, which makes "we hold overlap, we lose topology" a property of the approach rather than a K=8 coincidence. |
| K=16 per dataset | We lead only on spheroidj (+0.101, p=0.128, n.s. at n=24). Largest losses hrf −0.067 (p=3e-07), fisbe −0.065 (p=0.020), bbbc010 −0.049 (p=2e-11). |
| **NO prose number in the paper changes** | Verified with `audit_paper_numbers.py` against the campaign tree: every claim the paper states is at K=1 or K=8 and both halves were already complete. Panel 0.789/0.803, overlap 0.789/0.788, centreline 0.788/0.830, held-out −0.025 vs design-time −0.009, K=1 0.711 and +0.028, K=1 spheroidj 0.804 vs 0.494, 89% of K=16 — all reproduce. |
| paper edits this enabled | \cref{fig:kscale} redrawn with nnU-Net at all four support sizes and the caption's "three support sizes it was run at" clause removed; the located crossover and the bounded overlap lead added to the results; the 448 resolution control rewritten from best_v2's five-dataset −0.002 to the reported arm's eleven-dataset −0.004 with centreline +0.001 ([[C69]]). Five pages, page five back matter only, no overfull boxes. |
| **two resume defects, both fixed** | (1) `run_final_gap.done()` tested record EXISTENCE, not completeness, so the four K=16 cells the reboot left at 4–9 seeds were skipped: the band would have "finished" at 192/210 and the figure would have averaged cells with unequal seed counts, silently. It now requires full seed coverage and prints what it resumes. (2) `nn_status.py` reported "driver ALIVE \| 0 trainings" for a dead driver, because `pgrep -f run_final_gap` matches its own shell. Bracketed, plus a warning when a live driver has zero trainings. |
| **a third defect the rebuild caught** | Regenerating the tables ON TULEN reintroduced a **PerSAM column** and an undefined `persam` citation: the generator revert of [[C67]] was made locally and never synced. An md5 sweep of all seven paper generators found `make_ablation_table.py` had drifted too. Both synced; the sweep is the fix, and it should precede any regeneration. |
| date | 2026-08-05 |

### C78 — the three advertised-but-unmeasured components, in the reported regime (DONE)

| field | value |
|---|---|
| date | 2026-08-25, launched 22:4x, driver pid 4062979 on tulen |
| why | The paper advertises a two-scale backbone read, a prior-guided upsampler and a post-concat mixing block, and Table 2 measures none of them. §9.2 of the pre-submission blockers; the claim stands without them, so this buys robustness against "did you measure what you advertise?" |
| method | four arms, each differing from the reported config in EXACTLY ONE setting (verified by constructing all four and diffing backend fields): |
| | `ablv3_coarseonly` = `head_fusion_best_nobank_flat_es_ep500_wd4_do5_mix_coarseonly` (dino_scale coarse) |
| | `ablv3_fineonly`  = `..._mix_fineonly` (dino_scale fine) |
| | `ablv3_noup`      = `..._mix_noup` (upsampler None = plain bilinear) |
| | `ablv3_nomix`     = `head_fusion_best_nobank_flat_es_ep500_wd4_do5` (the `_mix` token dropped) |
| baseline | `stripv3_k8` (the reported method, same regime, already in the tree) |
| K/pool/test/seeds | K=8, pool 20 (15 for ctc_u373 via `pool_for`), test 10000 (full split), 10 seeds |
| res | 672 |
| datasets | monuseg, drive, spheroidj, dsb2018, ctc_u373, hrf, bacteria (the seven ablation datasets) |
| cache | `/disk1/prusek/cache_abl4`, **pre-built single-threaded first** (435 extractions) so four concurrent workers never race to write it |
| score_dir | one per arm: `results/final10/ablv3_{coarseonly,fineonly,noup,nomix}_k8` |
| launcher | `ASG_PY=/scratch/prusek/envs/seg/bin/python ASG_FEAT_CACHE=/disk1/prusek/cache_abl4 python scripts/run_final_gap.py --only abl --workers "0,0,0,1"` |
| log | `/disk1/prusek/abl4.log` |
| cost | 28 cells, ~24 GPU-h, ~6 h wall clock on 3 A100 workers + 1 A5000 worker |

**Code change:** one token. `noup` added to `al_testbed.py`'s allowed/named sets and
`upsampler=(None if "noup" in toks else sf_up.get(name))`. `coarseonly`/`fineonly` already existed,
and the `mix` ablation needs no code at all — `_mix` is a token the reported name carries, so the
arm is the name without it. `tests/test_ablation_arms.py` gained `"upsampler"` in `FIELDS` and a
`noup` token-walk case; without that the distinctness guard was blind to the new arm and would have
compared it EQUAL to the full method.

**Pre-launch review caught a blocker.** The hand-written command had no `--metric_override`. Four of
the seven datasets (monuseg, dsb2018, ctc_u373, bacteria) carry `metric="instance_ap"` in the
registry, so they would have been scored and recorded as AP: incomparable to `stripv3_k8` and fatal
to `make_ablation_table.py`, which skips non-matching records and then exits on the hole. Six hours
of GPU for a plausible-looking table that measures something else. Fixed by routing through
`run_final_gap.py`, whose `_cmd` supplies the override per dataset. Also caught: `~/dinov3_env` no
longer exists on tulen (envs consolidated to `/scratch/prusek/envs` today), so the driver's default
interpreter was dead — the preflight would have refused to dispatch.

**Caveats the eventual rows must carry** (from the same review, verified by construction):
* A single DINO scale necessarily also switches scale-fusion off (`head_fusion_backend.py:1461`), so
  those two arms concatenate 32+35=67 channels rather than 99. Label them "single DINO scale", never
  imply they share the 99-channel penultimate the paper describes.
* The fine branch is capped at `fine_max_grid=160`, so `fineonly` is NOT "native resolution".
* `GuidedUp` is zero-initialised, i.e. exactly bilinear at step 0, so `noup` measures the LEARNED
  edge-guided residual, not the presence of guidance.
* Every arm early-stops on its own plateau, so each measures the block plus how the plateau rule
  responds to it. Honest for the shipped method, but say it.
* Head parameters: reported 358,168 · coarse/fine 352,760 (−1.5%) · noup 347,540 (−3.0%) ·
  nomix 348,268 (−2.8%). Too small to confound weight decay or dropout.

**RESULTS (2026-08-26, all 28 cells complete; split fingerprints match `stripv3_k8` on all seven
datasets, so the arms are subtracted over identical test images).** Seed-paired panel mean over the
seven ablation datasets:

| arm | monuseg | drive | spheroidj | dsb2018 | ctc_u373 | hrf | bacteria | MEAN | Δ |
|---|---|---|---|---|---|---|---|---|---|
| FULL (reported) | 0.637 | 0.762 | 0.917 | 0.848 | 0.800 | 0.728 | 0.917 | **0.8014** | — |
| coarse scale only | 0.634 | 0.758 | 0.906 | 0.850 | 0.795 | **0.695** | 0.913 | 0.7929 | −0.0084 |
| fine scale only | 0.636 | 0.752 | 0.917 | 0.838 | **0.775** | 0.730 | 0.897 | 0.7920 | −0.0093 |
| no guided upsampler | 0.631 | 0.755 | 0.913 | 0.847 | 0.800 | 0.721 | 0.917 | 0.7978 | −0.0036 |
| no mixing block | 0.634 | 0.752 | 0.908 | 0.841 | 0.797 | 0.718 | 0.916 | 0.7951 | −0.0062 |

Paired Wilcoxon vs the reported arm, seeds collapsed to one score per test image: coarse-only
significant on 5/7, fine-only 5/7, no-upsampler 4/7, no-mix 5/7.

**THE FINDING IS NOT THE MEAN, IT IS WHERE EACH SINGLE-SCALE ARM FAILS.** Coarse-only collapses on
**HRF (−0.033, p<1e-4)** — 1–3 px vessels need the fine branch. Fine-only collapses on **CTC-U373
(−0.025, p<1e-4)** and bacteria (−0.020) — phase-contrast cells need global context. The two
branches fail on DIFFERENT datasets, which is the same complementarity the paper's headline claims
for priors and frozen features, one level down. Neither failure is visible in the two columns
Table 2 prints (MoNuSeg, DRIVE), so **these belong in prose, not in the table** — table rows would
hide the result rather than show it.

**Ordering unchanged.** All three components together are worth ~0.019, still an order of magnitude
below the prior bank's 0.101, so nothing in the paper's claim or its ranking moves. This is the
robustness answer to "did you measure what you advertise?", not a new result.

**Operational note.** A UTIA-wide network outage (tulen, kajman and panda all unreachable; the route
died inside their network, past CESNET) cut the session at ~03:40. The campaign was unaffected:
`nohup` kept the driver alive, tulen never rebooted (uptime 21 days), and all cells finished. The
driver's resume logic was never needed but would have made a restart cheap.

**Still to do when the cells land:** `make_ablation_table.py`'s `ARCH`/`SELFCFG` lists are
hardcoded and will not pick these up; a third block is needed before any table exists.

### C79 — the prior bank on all eleven datasets, and whether its value is predictable from structure width

| field | value |
|---|---|
| date | 2026-08-26 |
| why | The paper's headline is what the classical bank is worth on top of frozen features. The proposed mechanism: the gain grows as structures fall further below the backbone's patch stride. That is a correlation across datasets, so it needs the full panel, and the clean single-variable arm (`ablv3_nocls`, bank off, rules ON) only spanned seven |
| what ran | four cells: rozpad, bbbc010, isbi2012em, fisbe. Same arm, same protocol, driver-supplied `--metric_override` and `pool_for` (16 for fisbe/isbi2012em). ~3 GPU-h |
| result | the arm now spans all eleven; split fingerprints match `stripv3_k8` throughout |

**THE PRE-SPECIFIED TEST FAILED.** Width was measured as the descriptor the method itself uses,
`mean_radius = dt[binary].mean()` (`head_fusion_backend.py:653`), divided by the effective patch
stride in native pixels (`16 * native / 672`). Over the seven ablation datasets that gave
**rho = -0.821, p = 0.023**; adding fisbe and isbi2012em **strengthened** it to
**rho = -0.800, p = 0.0096** (n=9). Adding rozpad broke it: **rho = -0.573, p = 0.066 (n=11)**,
not significant. rozpad takes **+0.198** from the bank at a mean/stride of 0.604, where the
relationship predicts about +0.05.

The sequence matters: the relationship held at n=7, got *stronger* at n=9, and died at n=11. Had the
four cells not been run, a figure would have been built on seven points and refuted by the paper's
own eleventh dataset, in the score tree we release.

**DIAGNOSIS: the estimator, not the hypothesis.** rozpad is decaying spheroids -- a thick core plus
fine peripheral debris -- so its DT radii are strongly right-skewed and the MEAN is not a measure of
"does this mask contain sub-patch structure". Measured directly: rozpad's mean DT radius is
**29.46 px** against a median of **5.69 px**, a 5x discrepancy, and **75.8 %** of its foreground is
thinner than one patch.

**POST-HOC, AND LABELLED AS SUCH.** Re-running the same correlation on the MEDIAN DT radius over the
stride gives **rho = -0.764, p = 0.0062 (n=11)**; on the fraction of foreground thinner than one
patch, rho = +0.610, p = 0.046. Three statistics were tried on eleven points, so the effective
p-value is inflated and there is no held-out panel to confirm on. **This is exploratory. It does not
license a tested-law claim in the paper.** What it does license is descriptive, and the descriptive
version is strong on its own:

| median DT radius / stride | datasets | bank worth |
|---|---|---|
| below 0.26 | hrf, drive, fisbe, rozpad, bbbc010, isbi2012em, monuseg, bacteria | +0.114 to +0.245 |
| 0.48 to 0.57 | dsb2018, ctc_u373 | +0.057, +0.027 |
| 1.24 | spheroidj | **+0.002** |

SpheroidJ is the endpoint that carries the argument without any statistic: the one dataset whose
objects are WIDER than a patch takes essentially nothing from the bank.

**SEPARATE FINDING ABOUT THE SHIPPED METHOD.** The gate that decides the centreline loss reads the
same `mean_radius`. On rozpad, 29.46 px drives `1 - ramp(radius; 4, 6)` to **exactly 0**, so clDice
is switched off there no matter how thin the structures are; the true median of 5.69 px would give
0.155 and engage it partially. No reported number is affected -- this is the shipped behaviour and
it is what was benchmarked, the same status as the anti-aliasing defect in `_tubularity` -- but a
bimodal mask defeats this descriptor twice over. **Method is frozen (C70): this is a Curator item,
not a fix.** Whether engaging clDice on rozpad would help is unmeasured.

### C80 — a clean frozen-backbone probe, so the paper's control needs no caveat

| field | value |
|---|---|
| date | 2026-08-26 |
| why | `\probeMean` read `probe_k8`, whose method string is `head_fusion_best_cgate_film_nobank_..._nocls_nocolor_noloss`: the reported head **plus** the competitive gate and FiLM, the two components the paper measured and rejected (C64/C65). The prose therefore had to describe the control as "the same head, plus two adaptive-weighting modules the reported configuration does not use", which invites the obvious question |
| arm | `probev3` = `head_fusion_best_nocls_nobank_noloss_nocolor_flat_es_ep500_wd4_do5_mix` — the reported head, bank zeroed, both support-derived rules off, nothing added |
| protocol | K=8, pool 20 (15 ctc_u373, 16 isbi2012em/fisbe), test 10000, 10 seeds, res 672, all eleven datasets |
| cost | 11 cells, ~9 GPU-h wall, ~5.5 h on 3 A100 workers + 1 A5000 |
| split_fp | matches `stripv3_k8` on all eleven |

**RESULT: the contamination was worth 0.0011.** Panel mean 0.6723 (dirty) against **0.6712** (clean);
every per-dataset difference is within ±0.007 and unsigned. Macros move as
`\probeMean` 0.672 → **0.671**, `\probeGap` 0.117 → **0.118**, `\probeWorstGap` 0.250 → 0.252,
`\probeBestGap` 0.031 → 0.034, `\probeSpread` 8 → 7.

This is consistent with C64/C65: the gate and FiLM are worth +0.0026 on the full method and about
+0.001 on a bare backbone. They do nothing in either place, which is why they were dropped. The
paper's control sentence now reads without a caveat.

**Operational note.** Fitting WITHOUT the bank is markedly *slower*, not faster: rozpad took 329 min
and MoNuSeg 134 against ~85 for the same dataset with the bank. Stripped of native structural
channels the head reaches its loss plateau later and `es_ep500` lets it run. The prior bank
therefore buys accuracy AND fit cost. Not in the paper (no space), but it is the pre-measured answer
to "aren't you buying that accuracy with compute?".

### C81 — nnU-Net's epoch budget, tested where it beats us

| field | value |
|---|---|
| date | 2026-08-27 |
| why | The paper trains nnU-Net for 100 epochs and called it "a budget chosen from its observed convergence". The evidence (C40) is a training-Dice trajectory on MoNuSeg and SpheroidJ, both overlap-scored — i.e. absent on exactly the metric where nnU-Net leads. "You handicapped the baseline" is the sharpest fairness objection the paper faces, and it is sharpest on centreline |
| run | `nnunet_bench.py --datasets drive --support 8 --pool 20 --test 10000 --seeds 1 --epochs 250` on the A5000 |

**RESULT, same seed, same draw, same test split:**

| budget | DRIVE clDice, seed 0 |
|---|---|
| 100 epochs (reported) | **0.8121** |
| 250 epochs | **0.8106** |

Two and a half times the budget moves the score by **−0.0015**, on the dataset where nnU-Net beats
us (0.808 against our 0.762 over ten seeds). The baseline is not under-trained, and the evidence is
now the test metric rather than a training curve.

**MISTAKE, recorded because it nearly cost the run.** `nnunet_bench.py` deletes its work directory
after scoring unless `--keep_work` is passed, and I did not pass it; the stdout log was
block-buffered and captured nothing. **The epoch-by-epoch trajectory I launched the run to collect
is gone.** What survived is the score record, which happens to answer the question better — a test
number beats a training curve — but that was luck, not design. Pass `--keep_work` for any run whose
purpose is the trajectory rather than the score.

### C82 — the missing cell: classical priors read by OUR head

| field | value |
|---|---|
| date | 2026-08-27 |
| why | The paper argues the two sources are complementary from a 2x2 with one asymmetric cell: DINO under our head (probev3, 0.671), priors under a RANDOM FOREST (ilastik, 0.595), both under our head (stripv3, 0.789). The priors-alone number came from a different reader, so "neither alone explains the margin" rested on two architectures |
| arm | `ablv3_bankonly` = `head_fusion_best_nobank_flat_es_ep500_wd4_do5_mix_bankonly`; new `bank_only` flag zeroes BOTH DINO grids in `_dino_x` and `_fine_tensor`, the exact mirror of `dino_only`. Field diff against the reported arm: exactly one field |
| protocol | K=8, eleven datasets, ten seeds, pool 20/15/16, test 10000, res 672; split_fp matches `stripv3_k8` throughout |
| cost | 11 cells, ~26 GPU-h; rozpad alone 529 min |

**THE 2x2, now one architecture throughout:**

| read by our head | panel |
|---|---|
| DINO features alone | 0.6712 |
| **classical priors alone** | **0.6995** |
| both | 0.7889 |

**Complementarity runs both ways: priors add +0.1176 to features, features add +0.0894 to priors.**
The priors alone also *beat* the frozen features alone under the same head, which the random-forest
reading (0.595) could not show.

**They dominate different datasets, in the direction the width story predicts:**

| dataset | features alone | priors alone |
|---|---|---|
| SpheroidJ (objects wider than a patch) | **0.882** | 0.608 |
| CTC-U373 | **0.760** | 0.648 |
| DRIVE (thinnest) | 0.509 | **0.733** |
| rozpad | 0.599 | **0.757** |

**CAVEATS, both verified before launch and both stated in the paper:**
1. Zeroing an input leaves the branch that reads it without gradient. Measured: `bank_only` freezes
   **262,864 of 358,168 weights (73.4 %)**, `dino_only` 1.2 %, the reported arm 0.3 %. The single-source
   arms therefore fit fewer weights than the full head. The classical group enters at the concat and
   never passes the stem in ANY arm, so the head's treatment of the priors is identical between cells.
2. **`predict()` and `_calibrate_instance` feed the REAL DINO grid to the instance decoder's
   merge-veto, so this arm's `per_image_native` instance-AP fields are a DINO/bank mixture and MUST
   NEVER BE READ.** The reported metric is unaffected: foreground comes from `_native_logits`, which is
   fully zeroed. `ablv3_nocls` has the identical structure.

**CORRECTED the same day.** The first version of the 2x2 used `probev3_k8` as the features-alone
cell. That arm has the self-configuration rules **off** (`nocls_noloss_nocolor`) while the other two
cells have them **on**, so the contrast mixed the 0.009 the rules are worth into a comparison meant
to be about the two feature families. The rules-matched cell is `ablv3_nocls_k8` (bank zeroed, rules
on). Corrected 2x2, all three in the reported configuration and differing only in which input is
zeroed:

| under one head, rules matched | panel |
|---|---|
| frozen features alone (`ablv3_nocls_k8`) | 0.6746 |
| **classical bank alone** (`ablv3_bankonly_k8`) | **0.6995** |
| both (`stripv3_k8`) | 0.7889 |

The finding survives and sharpens: the bank alone beats the features alone by **+0.025** with the
rules held constant. `probev3_k8` (0.6712) remains the right control for "what does a bare
frozen-feature head do with nothing else", and the wrong one for "what are the features worth".

The deltas are deliberately NOT printed beside the means in the paper: 0.7889−0.6995 = 0.0894 rounds
to 0.089 while the printed 0.789−0.699 gives 0.090, and a reader subtracts what is printed.

**PAIRED TEST, added 2026-08-27.** "The bank alone outscores the features alone" is a claim about the
datasets, not about a mean: paired per-image Wilcoxon with seeds collapsed, Holm over the eleven, the
bank leads on **8 of 11 and all eight survive Holm**. It loses on SpheroidJ (−0.307), CTC-U373
(−0.125) and FISBE (−0.026, not significant). The panel-mean difference is only +0.025 because
SpheroidJ's −0.307 nearly cancels eight wins, which is why the paper states the count rather than
the mean: the ordering of the two sources otherwise depends on panel composition.

**Correction to the Table 2 caption.** It briefly said Decay was excluded from the ablation panel
"for cost at 2048 pixels". That justification is false and was invented: HRF is 3504 px natively with
a comparable test split (25 against 24) and IS in the panel. No reason is documented anywhere in the
repo — the ablation set was six datasets before Bacteria was added and Decay was never among them.
The caption now names the omission without asserting a reason.

**Timing, against the prediction.** The review expected this arm to plateau EARLIER than the reported
one, since 73 % of the head is frozen. The opposite happened: monuseg 179 min against C80's 134,
spheroidj 153 against 51, rozpad 529 against 329. A head left with only the bank reaches its plateau
later, and `es_ep500` lets it run. SpheroidJ's 3x is the sharpest case and fits the width story: it is
the one dataset where the bank is worth nothing, so the head there is searching without its
informative input.


### C83 — the bank-vs-features claim gets its own declared Holm family (analysis only, no new GPU run)

**Why.** C82 established that the prior bank alone outscores the frozen features alone, and that
finding was promoted to the abstract and to contribution 1. The paper then reported "all eight
significant after Holm correction" — a third family of eleven comparisons that `make_stats.py` did
not compute and that `Experiments` did not declare. The count was right; nothing generated it.

**Setup.** No new run. `scripts/make_stats.py` gains FAMILY 3, pairing the two ablated arms against
*each other* rather than both against Exemplar: `compare()` grew a `ref=` parameter so an ablation
family can be arm-vs-arm. The claim is about the two inputs' relative order, so pairing either one
against the full method would measure a different thing and lose the per-image pairing.

| field | value |
|---|---|
| arms | `ablv3_bankonly_k8` (ref) vs `ablv3_nocls_k8` |
| datasets | all eleven, each at its designated metric |
| K / seeds | 8 / 10, seeds collapsed to one score per image before testing |
| test | paired Wilcoxon signed-rank, Holm within the eleven |
| command | `ASG_SEM_TREE=results/final10 ASG_SEM_OUT=paper/isbi2027 python3 scripts/make_stats.py` |
| emits | `\bankComparisons` `\bankAhead` `\bankSig` `\featSigVsBank` |

**Result — bank ahead on 8/11, all 8 Holm-significant; features ahead significantly on 2.**

| dataset | delta (bank − features) | Holm p |
|---|---|---|
| drive | +0.2167 | 1.5e-05 |
| rozpad | +0.1594 | 3.2e-06 |
| monuseg | +0.1127 | 0.00049 |
| bbbc010 | +0.0719 | 1.5e-05 |
| hrf | +0.0522 | 0.0036 |
| dsb2018 | +0.0512 | 1.1e-06 |
| isbi2012em | +0.0390 | 0.00073 |
| bacteria | +0.0287 | 0.00019 |
| fisbe | −0.0261 | 0.46 (n.s.) |
| ctc_u373 | −0.1248 | 2.3e-05 |
| spheroidj | −0.3067 | 1.2e-06 |

**What this changed in the paper.** The introduction claimed complementarity "in both directions"
with no test behind it; it now has one — the features win *significantly* on SpheroidJ and CTC-U373,
so the claim is symmetric evidence rather than an asymmetric finding plus a hedge. `Experiments` now
declares three families instead of two.

**Mistake worth recording.** The same pass renamed Table 2's `Backbone only` row to `Features only`.
The row and the prose macro `\featOnlyMean` read the *same* score directory but printed 0.700 and
0.675 — the seven ablation datasets against the eleven-dataset panel. Two names and two numbers for
one arm, in adjacent paragraphs. Nothing was wrong with either number; the naming made one
experiment look like two.

### C84 — nnU-Net's self-configuration, adapted to this method (SCREEN; results pending)

**Why.** User question, 2026-08-28: everything nnU-Net auto-tunes should be auto-tuned here too, and
nnU-Net can hold out validation tiles even at K=1. The reference for "what nnU-Net tunes" is the
INSTALLED `nnunetv2` on tulen, not the paper: `nnUNetTrainer` gives `initial_lr=1e-2`,
`weight_decay=3e-5`, `oversample_foreground_percent=0.33`, `num_iterations_per_epoch=250`,
**`num_val_iterations_per_epoch=50`**, `num_epochs=1000`, SGD(momentum 0.99, nesterov) +
`PolyLRScheduler`, and an augmentation suite (rotation p=0.2, scaling 0.7–1.4 p=0.2, Gaussian noise
p=0.1, blur p=0.2, brightness p=0.15, contrast p=0.15, simulate-low-res p=0.25, gamma p=0.3 and
inverted gamma p=0.1, mirroring).

**What transfers, and what does not.** Patch size, batch size, network depth, target spacing, the 3D
cascade and deep supervision are all driven by GPU memory for 3D volumes and by voxel spacing. This
method is a 0.36M-parameter head of 1x1 convolutions over a frozen backbone, trained full-image, so
none of those rules has anything to determine here. Foreground oversampling is vacuous for the same
reason: every full-image sample already contains foreground. What transfers is the augmentation
suite, the optimiser and schedule, and validation.

**New tokens (all default OFF; the reported method is unchanged).**

| token | effect | nnU-Net counterpart |
|---|---|---|
| `naug<N>` | N offline augmented copies (rotate ±25°, scale 0.85–1.18, intensity jitter, nearest-neighbour label warp), re-encoded through the frozen backbone | augmentation pipeline |
| `elas<N>` | elastic field strength N/100 inside the same augmenter | `SpatialTransform` elastic |
| `vt<N>` | hold out a strip of width N% of every support item; early-stop on foreground IoU measured there; restore the BEST head, not the last | `num_val_iterations_per_epoch=50` |
| `nnopt` | SGD(momentum 0.99, nesterov) + PolyLR `(1-e/E)^0.9` at lr 1e-2 | `configure_optimizers` |

`n_aug`/`aug_elastic` already existed in the backend with no token and **no A/B anywhere in this
registry** — implemented and switched off. Re-encoding rather than augmenting cached features is
forced: the dihedral group commutes with a patch grid, warps and intensity changes do not.

**Screen setup.**

| field | value |
|---|---|
| arms | baseline, `_naug4`, `_naug4_elas25`, `_vt25`, `_nnopt_ep250` |
| targets | monuseg, fisbe (least headroom at K=1), drive (only dataset where the strip has edge foreground) |
| controls | spheroidj (semantic/binary GT), dsb2018 (instance) |
| K / seeds | 1 / 3 |
| metric | `--metric_override fg_iou` for monuseg,dsb2018,spheroidj; native cldice for drive,fisbe |
| caches | `asg_cache_nnscreen_g0`, `_g0b` — separate per process, never shared writable |
| score_dir | `results/nnscreen` |
| GO rule | one target > +0.010 AND no control < −0.005 AND no crash |
| eval | `ASG_SCREEN=results/nnscreen python scripts/eval_nnscreen.py` |

**Three defects the smoke test caught before the panel.**

1. **`vt` was broken.** SpheroidJ is one central spheroid, so an edge strip is pure background: IoU
   0.0000 every epoch, "no improvement" fired at once, the fit stopped after 26 steps and scored
   **0.213 against ~0.81**. nnU-Net has the identical hazard and answers it with foreground
   oversampling. Fixed analogously: the strip is chosen on whichever side carries foreground, must
   hold ≥16 foreground pixels while leaving ≥50 % of the item's foreground in training, and when
   neither side qualifies the item is trained whole and the plateau rule takes over rather than a
   fabricated signal being acted on. After the fix SpheroidJ falls back correctly (0.814) and DRIVE
   validates (strip IoU 0.057 → 0.618 over 212 epochs).
2. **`UnboundLocalError`.** The new tokens were read inside the `head_fusion_best_` branch while the
   backend constructor is shared by every method name. Caught by `tests/test_ablation_arms.py`.
3. **OOM.** The screen was launched on the A5000 while the semfix K=4 run held 19.2 of its 23.5 GB.
   Moved to the A100. A record for `_vt25`/drive survived that aborted run and was **deleted**: it
   was written on a different GPU with a different cache, which is the environment-mixing this
   session already ruled out for the score tree.

**Result — `vt25` DROPPED (complete, 5/5 datasets).**

| arm | monuseg (T) | fisbe (T) | drive (T) | spheroidj (C) | dsb2018 (C) |
|---|---|---|---|---|---|
| baseline | 0.613 | 0.623 | 0.701 | 0.817 | 0.787 |
| `vt25` Δ | **−0.035** | **−0.038** | +0.000 | −0.000 | **−0.117** |

Fails BOTH halves of the GO rule at once: no target gain anywhere, and DSB2018 regresses twenty times
the −0.005 control threshold. Two details identify the mechanism rather than noise.

*SpheroidJ came out at 0.817, matching baseline to the third decimal* — that is the fallback working
exactly: one central spheroid, no edge strip carries foreground, the item declines to validate and
the plateau rule takes over. *DRIVE came out at +0.000* — and DRIVE is the dataset where the strip is
healthiest (vessels span the field; strip IoU climbs 0.057 → 0.618 over 212 epochs). Even where the
held-out signal is at its best, it buys back nothing for the data it costs.

**Why it transfers badly, stated for the next paper.** Holding out 25 % at K=1 means training on
three quarters of the only image there is, and the strip is small enough to be a noisy stopping
signal in its own right — DSB2018 seed 0 stopped on a bad strip estimate at 0.551 against a baseline
seed spread of 0.780–0.791. nnU-Net can afford validation because its validation set is a different
CASE and it has hundreds of them over 1000 epochs. A strip of the same image is less data AND a
worse signal. **Held-out validation is not freely transferable into the few-shot regime**: the price
in data is orders of magnitude higher than the value of the stopping signal.

**Follow-up implemented, not yet screened: `vloo`.** Leave-one-out over the support masks. Costs 1/K
of the data (12.5 % at K=8 against 25 % of every image) and holds out a genuinely independent case,
which is what nnU-Net's fold split actually is. Undefined at K=1 — and that is the honest answer, not
a gap to engineer around: with one annotated mask there is no held-out data, and the strip experiment
is the evidence that pretending otherwise costs more than it returns.

**Result — `naug4`, `naug4_elas25`, `nnopt_ep250`.** Pending. Early partial: `nnopt_ep250` monuseg
0.603 against baseline 0.613, and its smoke run ended "flat/rising" after 65 steps, so if it fails it
must be re-run at a lower rate before being logged as negative — otherwise the registry would record
"nnU-Net's optimiser does not work here" when what was measured is "1e-2 does not".

### C85 — can the auxiliary outline head come out? (SCREEN: NO)

**Why.** User question, 2026-08-31: it is the last piece of instance machinery in a method the paper
now presents as purely semantic, and it had **never been A/B'd** — the paper says so itself ("in
every row and never ablated"). Removing self-configuration makes the question sharper, not softer:
in the non-adaptive branch its BCE is applied UNGATED at weight 1.0 on every dataset, whereas the
adaptive constructor ramped it by component count and switched it off on SpheroidJ entirely.

**Setup.** New `noaux` token (`al_testbed.py`, gates `boundary_head`). Screen against
`ablv3s_sc_none_k8` (same arm, head on), 3 seeds, K=8, `--metric_override fg_iou` for the four
overlap datasets, native clDice for DRIVE. Targets chosen as the datasets the head exists for:
MoNuSeg (dense nuclei), DSB2018 (touching nuclei), Bacteria (119 objects/image). Controls: SpheroidJ
(one object) and DRIVE (vessels).

| dataset | with | without | delta | role |
|---|---|---|---|---|
| MoNuSeg | 0.6223 | 0.6271 | **+0.0048** | target |
| DSB2018 | 0.8507 | 0.8478 | −0.0029 | target |
| Bacteria | 0.9237 | 0.9226 | −0.0010 | target |
| SpheroidJ | 0.8985 | 0.8956 | −0.0029 | control |
| DRIVE | 0.7486 | **0.7404** | **−0.0082** | control |

**Verdict: DROP the ablation, KEEP the head.** Controls fail the −0.005 bound on DRIVE. Nothing was
gained on any target: mean +0.0003 across the three datasets with hundreds of touching objects each.

**The interesting part is WHERE it pays.** The head does nothing on dense instances — the case it was
built for as the "W2 inter-instance boundary" lever — and everything it is worth is on vessels. That
is consistent with what `_boundary_target` actually computes: `find_boundaries(mode="inner")` marks
every object's inner rim against background, not only the contact faces between neighbours (verified
2026-08-28 on a synthetic label map: an isolated square still receives a full outline). On a vessel
network the "outline" is essentially the whole structure, so the head acts as an auxiliary
thin-structure target rather than an instance separator. **The paper's Method should describe it that
way**; calling it instance machinery is both wrong and needlessly exposed in a semantic-only paper.

---

## C86 — `lean_bankonly_k8`: the third cell of the 2×2, in the REPORTED configuration (2026-09-01)

**Why.** The paper's headline measurement — "each input measured alone, under the same head,
differing only in which input is zeroed" — was reading three DIFFERENT methods: `ours` =
`stripv3_k8`, `featonly` = `ablv3_nocls_k8`, `bankonly` = `ablv3_bankonly_k8`. The last two carry
the support-derived rules the reported (lean) method no longer has, so the contrast mixed the
0.009 those rules are worth into the feature-family comparison. `lean_nocls_k8` already existed;
this run supplies the missing bank-only cell in the same configuration.

**Setup.** Method `head_fusion_best_nobank_bankonly_noloss_nocolor_flat_es_ep500_wd4_do5_mix_noaux`
(one token — `bankonly` — from the reported string). 10 seeds, pool 20 (16 for ISBI2012-EM), test
10000, K=8, res 672, `--metric_override fg_iou` for the seven overlap datasets. Run on kajman
(`/data2/prusek/asg_cache_*`, own cache per process) + tulen; kajman's copy completed 11/11 and is
the one synced to `results/final10/lean_bankonly_k8`. Driver `/tmp/r2.sh` on tulen, three commands
(overlap set / drive,hrf,fisbe / isbi2012em at pool 16).

**Gate.** `split_fp` identical across `lean_k8`, `lean_nocls_k8`, `lean_bankonly_k8` on all eleven.

| panel (11) | features only | bank only | fused |
|---|---|---|---|
| mean | 0.672 | 0.693 | **0.782** |

**Result.** Bank ahead of features on **7 of 11** (all seven Holm-significant); features
significantly ahead on 2 of the other 4 (SpheroidJ +0.304, CTC-U373 +0.115 for features; HRF and
FISBE not significant). Fused beats the better of its two inputs on **10 of 11**, eight of those
Holm-significant; the exception is DSB2018 (−0.0029, Holm 0.41 — a tie, not a loss).

**Supersedes** the numbers this arm's predecessor produced: bank-ahead was reported as eight and
"fused beats both" as all eleven, both from the mismatched arms. `make_probe_numbers.py` and
`make_stats.py` now point at the lean cells.

---

## C87 — lean component ablations, QUEUED (2026-09-01)

**Why.** `make_ablation_table.py` computed `\dCoarseOnly` / `\dFineOnly` / `\dNoUp` / `\dNoMix` as
`lean_k8 − ablv3_*_k8`. The ablated arms are the **stripv3** configuration, so the subtraction
measured the method change and flipped every sign: the paper printed "reading at two scales is
worth −0.002" where against its own reference the same arm gives **+0.008**. Against `stripv3_k8`:
coarse-only +0.0084, fine-only +0.0093, no-up +0.0036, no-mix +0.0062 (seven ablation datasets).

**Fix in the generator.** `one_token_apart(full, ablated)` — a delta is emitted only if the two
method strings differ by exactly one lever token. Otherwise the macro is NOT written and the
LaTeX build fails loudly rather than printing a plausible wrong number. The prose sentence quoting
those deltas was removed from `main.tex` in the same commit.

**Queued.** `lean_coarseonly_k8`, `lean_fineonly_k8`, `lean_nomix_k8` — same seven ablation
datasets, 10 seeds, K=8, res 672, driver `/disk1/prusek/active-segmenter/lean_abl.sh` (waits for
the K=4/K=16 campaign to clear the GPUs, then A100 runs coarse+fine, A5000 runs no-mix, separate
caches). Method strings are the reported one plus `_coarseonly` / `_fineonly`, and minus `_mix`.

---

## C88 — INSID3 anchor reproduced, and their published number is the CRF configuration (2026-09-02)

**Why.** The paper carried a column labelled "DINOv3 corr." that is OUR OWN training-free
read-out following INSID3's premise, not INSID3. The fairness protocol requires a reproduced
published anchor per baseline; Matcher and PerSAM had one, this did not, and INSID3's code is
public (github.com/visinf/INSID3, CVPR 2026 Oral) — so not running it was a choice, not a
constraint. User asked for it directly.

**Their code, unmodified.** The blocker was weights: INSID3 loads Meta's original
`dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` through `torch.hub`, and the direct download is
licence-gated (HTTP 403). Rather than adapt their pipeline to our HF encoder, the HF checkpoint —
Meta's own redistribution of the same weights — was rewritten INTO Meta's hub format, so
`_build_encoder` runs untouched. Verified three ways rather than assumed:

| check | result |
|---|---|
| `load_state_dict(strict=True)` on their architecture | passes, no missing/unexpected keys |
| parameter accounting | HF is short 98,320 = 24x1024 absent K bias + 24x3072 `qkv.bias_mask` + 16 RoPE periods, all architecture constants |
| features, same input, both models | max abs 2.7e-06, cosine 1.0000000 |

Two further traps: `torch.hub` cannot reach GitHub from tulen (the repo was placed in its cache as
`facebookresearch_dinov3_main`, again without touching their code), and their loader CACHES the
checkpoint by filename in `$TORCH_HOME/hub/checkpoints`, so the corrected file was ignored until
that copy was deleted.

**Anchor, on their own script, data and defaults** (Chest X-ray, one-shot, 600 episodes,
DINOv3-L, image_size 1024). Their Table 1 reports **78.8 mIoU**.

| configuration | mIoU |
|---|---|
| default `mask_refiner="bilinear"`, seed 0 | 77.3 |
| same, seed 1 | 76.4 |
| same, seed 2 | 77.4 |
| **`--crf-mask-refinement`, seed 0** | **78.7** |

**The finding that matters for fairness: their published number is the CRF configuration, which is
NOT their code default.** `build_insid3` defaults to bilinear and the README calls CRF "additional
refinement", so the natural reading is wrong. Benchmarking the default would have run their method
1.5 points below what they report — the silent-handicap failure this project's fairness protocol
exists to catch, and the same class as the earlier `pydensecrf` fallback on our own read-out.

Getting CRF to build took three fixes: their CRF is `github.com/netw0rkf10w/CRF`, not
`pydensecrf`; the host's default CUDA is 13.3 against the env's torch cu126 (used the host's 12.9,
same major); and the system compiler is GCC 8.5 with no `-std=c++20` (used `gcc-toolset-13`).

**Panel run.** `scripts/insid3_bench.py` — plumbing only, calling `set_reference`/`segment` exactly
as their own `inference_segmentation.py` does, with our fixed split, our seeds and the same K=8
draw (`np.random.default_rng(seed).choice(...)`, sota_final.py:319 verbatim). Split fingerprints
match our arms cell by cell. Score dir `results/insid3_k8`, method `insid3_official`, CRF on.

**Early result, and it inverts an assumption.** Real INSID3 scores BELOW our own stand-in on the
datasets where it matters: MoNuSeg 0.221 against the read-out's 0.443, DRIVE ~0.22 against 0.279,
while matching it on SpheroidJ (~0.89 against 0.867). So the substitute was generous to them, not
stingy. The pattern is coherent: INSID3 is built for one salient instance per image, and dense
nuclei and thin vessels are where that assumption breaks.

---

## C89 — INSID3 on the panel, and Tyche at its own metric (2026-09-02)

**INSID3 panel.** `scripts/insid3_bench.py`, the authors' code (setup in C88), CRF on, at K=1/4/8/16
into `results/final10/insid3_k*`. Method `insid3_official`, ten seeds, split fingerprints matching
our arms cell by cell. CTC-U373 is skipped at K=16 by an explicit guard -- its pool of fifteen
cannot supply the draw, the same reason Fig. 2 averages ten datasets.

| K | 1 | 4 | 8 | 16 (ten datasets) |
|---|---|---|---|---|
| INSID3 panel mean | 0.419 | 0.438 | 0.433 | 0.433 |

**Two findings.**

*Real INSID3 scores BELOW the read-out that stood in for it* on ten of eleven datasets -- panel
0.433 against the substitute's 0.541, SpheroidJ the only dataset where it is higher (0.877 against
0.867). The substitute was generous to them, not stingy. SpheroidJ is also the only panel dataset
with one large object per image, which is the regime INSID3 is built for.

*Its K-curve is flat and dips.* Seven of eleven datasets DROP from K=4 to K=8 (-0.002 to -0.023),
so the dip is broad, not one dataset's noise. Their `predict_mask` averages the reference prototypes
across shots, and averaging eight heterogeneous references blurs what four already fixed. Their
published results are one-shot; the multi-shot mode is documented but not what they report. Ours
over the same budgets: 0.701 / 0.762 / 0.781 / 0.792, monotone.

**The headline did not move: 54 of 55, 52 Holm-significant, exactly as with the substitute.** That
is the reassuring outcome -- replacing a baseline with a weaker one did not buy a win we did not
already have. The single loss is still SegGPT on SpheroidJ.

**Tyche at its own metric.** `scripts/tyche_bench.py` scores both aggregations in one pass. Tyche's
paper reports best-of-K (`min_k L_Dice(y_k, y)`), justified by proposing a set to a human rater; we
had always averaged the candidate maps, which understates it.

| | panel mean |
|---|---|
| mean of candidates (what we reported) | 0.473 |
| best-of-16 (their metric) | 0.530 |

The gain is not uniform: +0.119 on CTC-U373 and +0.109 on DSB2018 against +0.007 on ISBI2012-EM, so
it could not have been estimated from one dataset. **Exemplar still leads on all eleven against the
oracle selection**, ten at raw p < 0.05 (SpheroidJ p = 0.12).

*Presentation.* Table 1 and Fig. 2 keep the no-oracle mean, because an oracle number among columns
that have none misleads a reader scanning columns and a caption caveat is weak against that
impression. The Baselines paragraph states the best-of result and the outcome. Putting their metric
in the table would also have needed `tyche_bestof` at K=1/4/16 (~15 h) to keep the figure consistent.

*Validation of the new harness:* the `mean` path reproduces the shipped `tyche_k8` arm to within
0.0005 on all eleven datasets, the residual being Tyche's own stochasticity.

**Baseline protocol audit (the question that started this).** Of the five forward-pass baselines,
two were running in a protocol their authors do not report, and both errors were in OUR favour:
Tyche (above) and INSID3 (C88's CRF finding, and it was a stand-in at all). SegGPT
(`feature_ensemble=True`, its documented K-shot mechanism), UniverSeg (native support set at its
128-px training resolution) and Matcher (one-shot, anchor reproduced) were already correct.

The specialists' claim "with published anchors reproduced" was removed from the paper as
**unattainable rather than merely unsupported**: StarDist, Cellpose-SAM and micro-SAM publish
numbers for models TRAINED on the target dataset, while we use the released generalists off the
shelf. Their per-modality model selection is documented and correct (`stardist_fluo`/`_he`,
`microsam_vit_b_lm`/`_vit_l_histopathology`, `cellpose_cpsam`); that is what the text now claims.
