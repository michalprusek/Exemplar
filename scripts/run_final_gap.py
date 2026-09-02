#!/usr/bin/env python
"""The two holes left in the paper, run as one scheduled queue over both GPUs.

WHAT IS MISSING AND WHY IT MATTERS.

1. **nnU-Net has two points on a four-point figure.** Every other method in the K-scaling figure is
   measured at K=1,4,8,16; nnU-Net only at 1 and 8. The paper's headline is that the two curves
   CROSS, and with two points the crossing can only be stated as "somewhere between one and eight".
   K=4 turns that interval into a located number, which is what a reader deciding how many masks to
   draw actually needs. K=16 is run too: the argument for skipping it (it confirms the predictable --
   more data favours a network trained from scratch) is sound but invisible to a reviewer, who sees
   only a figure where our curve runs to sixteen and the baseline stops at eight.

2. **The ablation is measured in a different regime from the results it explains.** Every `abl_*_k8`
   arm is the BASE head (wide 3x3 stem, sixty fixed epochs, no regularisation) while Table 1 reports
   best_v3 (`_flat_es_ep500_wd4_do5_mix`). On DRIVE that is 0.690 against 0.771: a regime difference
   of +0.081, LARGER than the largest component effect the ablation measures (-0.069). The caption
   asserts the ordering survives the regime change; this re-runs every arm in the reported regime so
   it is shown instead. BACTERIA is added because all six existing ablation datasets took part in
   design decisions and none of the four prepared after the method was frozen is represented.

THE FULL ARM IS NOT RE-RUN. `oursv3n_k8` already IS the full method in the reported regime, over all
eleven datasets at ten seeds. Pointing the table at it instead of re-running costs nothing and buys
something better: the ablation's reference row becomes numerically identical to Table 1's row, so the
two tables cannot disagree.

CONVENTIONS ARE IMPORTED, NOT RESTATED. `run_campaign` owns the per-dataset pool, the metric override
rule, the feature-cache path and the preflight check, and its own comments record what a second copy
costs: a blanket `fg_iou` override once made our method write a metric no baseline wrote, and `stats`
then skipped every comparison on four datasets silently. A staged version of this script hand-wrote
`--pool 20 --test 24`; the reported arm uses the FULL test split and pool 15 on ctc_u373, so those
arms would have been non-comparable with the table they were meant to join.

  python scripts/run_final_gap.py --workers "0,0,0,1" [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time
from queue import Queue

ROOT = os.environ.get("ASG_REPO_ROOT", "/disk1/prusek/active-segmenter")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_campaign import (ENVS, FEAT_CACHE, HE_DATASETS, MAX_WAIT_S, MICROSAM_MODEL,  # noqa: E402
                          OFFSHELF_BACKEND, PY, SEEDS, TREE,
                          _free_mib, min_free_mib, pool_for, preflight)
from active_segmenter.eval.registry import PANEL  # noqa: E402

NNUNET_PY = os.environ.get("ASG_NNUNET_PY", os.path.expanduser("~/nnunet_env/bin/python"))

# The regime tokens that separate best_v3 from the base head. Verified before this script was written
# by constructing each arm and diffing its attributes against the base: dropout 0 -> 0.5, early_stop
# False -> True, epochs 60 -> 500, mix False -> True, stem 'wide' -> 'flat', weight_decay 0 -> 1e-4.
# A token the parser does not know is REJECTED rather than ignored, so a typo here is a dead run, not
# a quiet fallback -- which is why the construction check runs in --dry-run below.
V3 = "_flat_es_ep500_wd4_do5_mix"

# label -> the method the harness knows. The labels mirror the existing `abl_*_k8` directories so the
# table generator can switch regimes by changing a prefix. `sc_none` keeps the competitive gate and
# drops FiLM, the adaptive loss and the colour rule, exactly as `abl_sc_none_k8` does, so the two are
# comparable across regimes rather than differing in two things at once.
#
# REGIME CHANGE, 2026-07-31. The gate and FiLM were removed from the method, so the seven arms above
# now ablate a method that no longer exists. Two of them survive the change untouched and are NOT
# re-run: `ablv3_nocls` and `ablv3_bank` carry neither `_cgate` nor `_film`, so they already ARE the
# stripped architecture with and without the prior bank. The three self-configuration arms do not
# survive -- each carries `_cgate` or `_film` -- so "what does turning off the colour rule cost?" was
# answered on top of two components the paper no longer has. Re-running them is not bookkeeping: if
# Table 2's full row is the gate+FiLM method while Table 1's is the stripped one, the two tables
# report different methods under the same name, which is the exact defect the last ablation re-run
# was undertaken to remove.
ABL_ARMS = {
    "ablv3_nocls":       "head_fusion_best_nocls_nobank" + V3,
    "ablv3_bank":        "head_fusion_best_nobank" + V3,
    "ablv3s_sc_noloss":  "head_fusion_best_nobank_noloss" + V3,
    "ablv3s_sc_nocolor": "head_fusion_best_nobank_nocolor" + V3,
    "ablv3s_sc_none":    "head_fusion_best_nobank_noloss_nocolor" + V3,
    # ADDED 2026-08-25. Three components the paper advertises and never measured: the two-scale
    # backbone read, the prior-guided upsampler, and the post-concat mixing block. Each arm differs
    # from the reported configuration in exactly one setting, verified by constructing all four and
    # diffing their fields. Two caveats the row labels must carry: a single DINO scale necessarily
    # also switches scale-fusion off, so those arms concatenate 67 channels rather than 99; and the
    # guided upsampler is zero-initialised, so `noup` measures the learned edge-guided residual, not
    # the presence of guidance at initialisation.
    "ablv3_coarseonly":  "head_fusion_best_nobank" + V3 + "_coarseonly",
    "ablv3_fineonly":    "head_fusion_best_nobank" + V3 + "_fineonly",
    "ablv3_noup":        "head_fusion_best_nobank" + V3 + "_noup",
    "ablv3_nomix":       "head_fusion_best_nobank_flat_es_ep500_wd4_do5",   # V3 minus the _mix token
    # ADDED 2026-08-26. The frozen-backbone probe the paper cites (\probeMean) is `probe_k8`, which
    # carries the competitive gate and FiLM on top of the reported head -- the two components the
    # paper measured and rejected -- so the sentence describing it needs a caveat that invites the
    # question "why did you not just run it clean?". This is that arm: the reported head with the
    # bank zeroed and both support-derived rules off, and nothing else added.
    "probev3":           "head_fusion_best_nocls_nobank_noloss_nocolor" + V3,
}

# The six existing ablation datasets plus bacteria (the held-out morphology). Cost is minutes for ten
# seeds, from the measured per-support-set fit times (registry: fit = setup + E * epoch_cost), and is
# used only for longest-first ordering -- being wrong costs makespan, never correctness.
ABL_DATASETS = {"hrf": 117, "monuseg": 85, "spheroidj": 74, "bacteria": 38,
                "drive": 22, "ctc_u373": 17, "dsb2018": 4}

# ADDED 2026-08-26. The BANK arm alone runs on the full eleven-dataset panel, not the ablation seven.
# Reason: the paper's headline is what the classical bank is worth on top of frozen features, and the
# mechanism behind it -- the gain grows as structures fall further below the backbone's patch stride
# -- is a correlation over datasets. On the clean single-variable arm it is Spearman rho = -0.821
# (p = 0.023) with n = 7, which is suggestive and fragile; the only eleven-dataset arm available
# (`probe_k8`) also has the self-configuration rules off and carries the two rejected components, so
# its rho = -0.491 (p = 0.125) measures a mixture rather than the bank. Four more cells make the
# clean arm span the same eleven datasets Table 1 reports.
ABL_FULL_PANEL = {"ablv3_nocls", "probev3"}          # arms that run on all eleven, not just the ablation seven
ABL_EXTRA = {"rozpad": 100, "bbbc010": 25, "fisbe": 25, "isbi2012em": 15}

# CORRECTED 2026-08-01 (registry C66). The comment here used to say nnU-Net costs ~3260 s per
# support set "on every dataset and at every K", so this was one flat number. That figure came from
# the manuscript and had no measurement anywhere in the repo, and when it was finally measured on an
# idle A5000 it was wrong in BOTH the value and the invariance: dsb2018 (256 px) took 2257 s and hrf
# (3504 px) did not finish inside a 9000 s cap. The epoch and iteration budget is fixed, but
# preprocessing and sliding-window inference are not, and they scale with the field.
#
# So the cost is now linear in the field side, anchored on those two measurements: 376 min per
# ten-seed cell at 256 px rising to >=1500 at 3504. Getting this wrong costs makespan and never
# correctness (the ordering is longest-first), but on a band this size a flat estimate would have
# started the 25-hour hrf cells last and left three workers idle at the end.
NN_KS = (4, 16)
NN_SIDE = {"hrf": 3504, "rozpad": 2048, "spheroidj": 1296, "monuseg": 1000, "ctc_u373": 696,
           "bbbc010": 696, "fisbe": 686, "bacteria": 600, "drive": 584, "isbi2012em": 512,
           "dsb2018": 256}


def nn_cost(ds):
    """Minutes for one ten-seed nnU-Net cell, interpolated between the two measured anchors."""
    return 376 + (NN_SIDE.get(ds, 700) - 256) * (1500 - 376) / (3504 - 256)
NN_DATASETS = ["spheroidj", "dsb2018", "monuseg", "ctc_u373", "drive", "hrf",
               "isbi2012em", "fisbe", "bbbc010", "bacteria", "rozpad"]

# THE INSTANCE ROW. The paper reports a semantic foreground map and says so, but ISBI has a large
# cell-segmentation audience and "you never report an instance metric" is the likeliest substantive
# objection to it. The defence is cheap and already built: the SAM-free affinity-watershed decoder
# turns the foreground map into instances, and the three datasets with per-instance ground truth are
# scored on AP@[.5:.95] simply by NOT passing --metric_override, so the registry's own metric applies.
# Same method, same protocol, same fit -- only the readout differs -- which is exactly the claim
# being defended: that the foreground map is a usable seed rather than the end of the pipeline.
INSTANCE_DATASETS = {"monuseg": 85, "ctc_u373": 17, "dsb2018": 4}
INSTANCE_DIR = "oursv3_ap_k8"
INSTANCE_METHOD = "head_fusion_best_cgate_film_nobank" + V3

# ...AND THE ONLY COMPARATOR THAT MEANS ANYTHING FOR IT. Our AP alone answers "is it a number" but
# not "is it usable", and the instance-AP records that exist for the forward-pass baselines are six
# seeds from an older campaign at a mixed K, so pairing against them would be a fairness problem
# rather than a comparison. The specialists produce instances NATIVELY and are already a column in
# Table 1 (there on foreground, as the union of their instances), which makes them both the natural
# and the most demanding comparator. They ignore the support, so this is one deterministic run each.
#
# SEPARATE SCORE DIRECTORY, deliberately. `score_record_path` is one file per (method, dataset), so
# writing these into the specialists' existing directory would silently overwrite the foreground
# numbers Table 1 reports with AP ones -- the exact overwrite the campaign's own comments warn about.
SPEC_AP_DIR = "spec_ap"
SPEC_AP = {"cellpose_sam": 6, "stardist": 6, "microsam": 15}

# THE STRIPPED METHOD (user, 2026-07-30): competitive gate and FiLM removed, colour rule kept.
#
# The ablation measured this exact configuration at K=8 on seven datasets and it costs -0.0029 on
# the panel mean, which is NOT significant, for 28% fewer trainable parameters (492,675 -> 352,692).
# It does regress the two vessel datasets significantly (DRIVE -0.0099, HRF -0.0105) and improve
# spheroids (+0.0076) -- selective, not uniform. The user's call is that a component which does not
# move the panel mean but costs a Method sentence, an ablation row and a "why is it there" question
# is a liability in a four-page paper, and that is a defensible reading of the same numbers.
#
# `ablv3_bank_k8` already covers seven datasets at K=8, so only four are missing there; the whole
# K-scaling curve has to be re-run. The old records are untouched, so this is reversible.
STRIP_METHOD = "head_fusion_best_nobank" + V3
STRIP_K8_MISSING = ["rozpad", "bbbc010", "isbi2012em", "fisbe"]
STRIP_KS = (1, 4, 16)
# Cost per (dataset, K) in minutes for ten seeds, from the ablation's measured cell times. Used only
# for longest-first ordering inside the band; being wrong costs makespan, never correctness.
STRIP_COST = {"hrf": 240, "bacteria": 150, "monuseg": 121, "spheroidj": 83, "rozpad": 60,
              "bbbc010": 60, "isbi2012em": 40, "fisbe": 40, "ctc_u373": 32, "drive": 18,
              "dsb2018": 18}


def _override(ds):
    """The campaign's override convention, read from the registry rather than re-listed here."""
    return "cldice" if PANEL[ds].metric == "cldice" else "fg_iou"


def _abl_dir(label):
    return f"{TREE}/{label}_k8"


def _nn_dir(k):
    return f"{TREE}/nnunet_k{k}"


def jobs():
    """Cells in the order they should be RUN, which is not the order that minimises makespan.

    Pure longest-first packs best -- 58 h against 64 h here -- but it dispatches all 21 nine-hour
    nnU-Net cells before the first ablation cell, so the ablation lands at hour 58 instead of hour 11.
    That is the wrong trade: the ablation is a CORRECTNESS fix (every published arm is the base head
    while Table 1 reports best_v3, which is a legitimate "major issue"), while the nnU-Net points are
    an enhancement. Six hours of makespan buys the blocking result two days earlier.

    Within each band the sort is still longest-first, so the tail of each band packs properly.

    K=16 runs LAST rather than interleaved with K=4. The crossover is what the extra points are for,
    and it lies between one mask and eight: K=4 locates it, K=16 confirms the predictable (more data
    favours a network trained from scratch). Ordering them this way also means the K=4 answer is in
    hand while K=16 is still running, which is when a decision about K=2 can actually be made --
    K=2 is only worth 100 GPU-h if we turn out to LOSE at K=4.
    """
    # THE STRIPPED ARM FIRST. It is what Table 1 and the K-scaling figure would report if the
    # method drops the gate and FiLM, so nothing else in the paper can be finalised until it lands.
    # K=8 before the rest of the curve: Table 1 is the core, K=1 carries the abstract's headline.
    out = []
    for k in (8,) + STRIP_KS:
        band = []
        for ds in NN_DATASETS:
            if k > pool_for(ds):
                continue
            if k == 8 and ds not in STRIP_K8_MISSING:
                continue                       # ablv3_bank_k8 already measured these seven
            # kind "strip", not "abl": these cells are the K-scaling curve, not ablation arms, and
            # --only has to be able to hand one band to one driver and the other band to another
            # without the two claiming the same cell.
            band.append(dict(kind="strip", k=k, ds=ds, label=f"stripv3_k{k}",
                             method=STRIP_METHOD, sd=f"{TREE}/stripv3_k{k}",
                             cost=STRIP_COST.get(ds, 60)))
        out.extend(sorted(band, key=lambda j: -j["cost"]))
    # Instance AP: three cells, ~2 GPU-h, closes the likeliest substantive objection.
    out += [dict(kind="ap", k=8, ds=ds, label=INSTANCE_DIR, method=INSTANCE_METHOD,
                 sd=f"{TREE}/{INSTANCE_DIR}", cost=cost)
            for ds, cost in sorted(INSTANCE_DATASETS.items(), key=lambda kv: -kv[1])]
    out += [dict(kind="specap", k=None, ds=ds, label=f"{SPEC_AP_DIR}_{m}", method=m,
                 sd=f"{TREE}/{SPEC_AP_DIR}_{m}", cost=cost)
            for m, cost in sorted(SPEC_AP.items(), key=lambda kv: -kv[1])
            for ds in INSTANCE_DATASETS]
    band = []
    for label, method in ABL_ARMS.items():
        sweep = dict(ABL_DATASETS, **ABL_EXTRA) if label in ABL_FULL_PANEL else ABL_DATASETS
        for ds, cost in sweep.items():
            band.append(dict(kind="abl", k=8, ds=ds, label=label, method=method,
                             sd=_abl_dir(label), cost=cost))
    band.sort(key=lambda j: -j["cost"])
    out.extend(band)
    for k in NN_KS:                     # NN_KS is ordered (4, 16): K=4 first, deliberately
        band = []
        for ds in NN_DATASETS:
            if k > pool_for(ds):        # ctc_u373 has 15 pool images: K=16 is impossible, not failed
                continue
            band.append(dict(kind="nn", k=k, ds=ds, label=f"nnunet_k{k}",
                             sd=_nn_dir(k), cost=nn_cost(ds)))
        # Longest-first WITHIN each K, like every other band. This was missing while the cost was a
        # flat constant, where it could not matter; with the measured cost spanning 376 to 1500
        # minutes it decides the makespan, because an hrf cell started last runs ~19 h past the
        # point where the other three workers have nothing left to do.
        band.sort(key=lambda j: -j["cost"])
        out.extend(band)
    return out


def cmd_for(j, dev):
    # The support size comes from the JOB, never a literal. It was hardcoded to 8 here because this
    # branch was written for the ablation, which is always K=8 -- and then the K-scaling arms reused
    # it and every stripv3_k1/k4/k16 cell silently ran at K=8 while writing into a directory named
    # for its K. The result would have been a perfectly plausible FLAT K-scaling curve. Caught only
    # because K=1 and K=8 agreed to four decimals on spheroidj.

    """The exact command for one job. nnU-Net takes the device as a flag; the in-context harness
    takes it through the environment, which is how each script's own launcher already drove it."""
    ds, sd = j["ds"], j["sd"]
    if j["kind"] == "specap":
        # run_campaign's own command shapes, with --fg-scoring DROPPED so the registry's metric
        # (instance AP on these three) applies instead of foreground IoU. Both benches take the
        # effective metric from that one flag, so this is the whole difference. The per-modality
        # model choices are imported, not re-listed: hardcoding vit_b_lm on H&E understated
        # micro-SAM roughly threefold, and running the fluorescence StarDist on H&E is both out of
        # domain and polarity-inverted.
        m = j["method"]
        if m == "microsam":
            return (f"CUDA_VISIBLE_DEVICES={dev} {ENVS[m]} {ROOT}/scripts/microsam_bench.py "
                    f"--mode panel --model {MICROSAM_MODEL.get(ds, 'vit_b_lm')} "
                    f"--datasets {ds} --pool {pool_for(ds)} --test 10000 --seeds {SEEDS} "
                    f"--score_dir {sd}")
        backend = OFFSHELF_BACKEND[m]
        if m == "stardist" and ds in HE_DATASETS:
            backend = "stardist_he"
        seeds = " ".join(str(i) for i in range(SEEDS))
        return (f"CUDA_VISIBLE_DEVICES={dev} {ENVS[m]} {ROOT}/scripts/cellpose_stardist_bench.py "
                f"--backend {backend} --datasets {ds} --pool {pool_for(ds)} "
                f"--test 10000 --seeds {seeds} --score_dir {sd}")
    if j["kind"] == "nn":
        # A private work dir per (K, dataset) so concurrent jobs cannot collide; nnunet_bench removes
        # it on completion unless --keep_work. --pool is DELIBERATELY not passed: its default is the
        # campaign's per-dataset pool, and a flat 20 would shift the test slice on download-kind sets.
        work = f"/disk1/prusek/nnwork_k{j['k']}_{ds}"
        return (f"nnUNet_n_proc_DA=6 {NNUNET_PY} {ROOT}/scripts/nnunet_bench.py "
                f"--datasets {ds} --support {j['k']} --seeds {SEEDS} --epochs 100 "
                f"--gpu {dev} --planner ResEncUNetPlanner --work_dir {work} --score_dir {sd}")
    # The instance arm deliberately omits --metric_override so the REGISTRY's metric applies
    # (instance_ap on these three). Passing fg_iou here, as every other arm does, would score the
    # instance run on foreground and produce a row identical to Table 1 -- a plausible table that
    # measures nothing, which is the failure mode the override convention exists to prevent.
    override = "" if j["kind"] == "ap" else f"--metric_override {_override(ds)} "
    return (f"CUDA_VISIBLE_DEVICES={dev} {PY} {ROOT}/scripts/sota_final.py run "
            f"--method {j['method']} --datasets {ds} --support {j['k']} --pool {pool_for(ds)} "
            f"--test 10000 --seeds {SEEDS} --res 672 {override}"
            f"--cache {FEAT_CACHE} --score_dir {sd}")


def done(j):
    """Has this exact (arm, dataset) already produced a COMPLETE score record?

    Existence is not completeness. `nnunet_bench.py` rewrites its record after every seed so a killed
    cell keeps what it finished, and it resumes from there -- but the driver has to actually schedule
    that cell for the resume to happen. Testing existence alone made this function skip the four K=16
    cells a machine reboot had left at 4-9 seeds, so the band would have "finished" at 192/210 and the
    K-scaling figure would have averaged cells with unequal seed counts, silently.

    A record without a `seeds` list is from a method that does not write one; existence is all there
    is to test there, and it is left as it was.
    """
    for f in glob.glob(os.path.join(j["sd"], f"*__{j['ds']}.json")):
        try:
            with open(f) as fh:
                rec = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return False                       # unreadable or caught mid-flush: re-run rather than trust
        got = rec.get("seeds")
        if got is None:
            return True
        if len(got) >= SEEDS:
            return True
        print(f"  [resume] {j['label']}/{j['ds']}: {len(got)}/{SEEDS} seeds -- scheduling the rest",
              flush=True)
        return False
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", default="0,0,0,1",
                    help="comma-separated CUDA device per worker; three on the A100 and one on the "
                         "A5000 is the default because a 2D nnU-Net does not saturate an 80 GB card")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-resume", action="store_true", help="re-run cells that already have records")
    # A band filter exists so a NEWLY QUEUED band can be started alongside a driver that is already
    # running, instead of restarting that driver and discarding hours of in-flight cells. Two drivers
    # are only safe while the feature cache is warm: a second WRITER on a shared cache is the
    # documented race that once manufactured a fake HRF regression. Check that the cache has taken no
    # writes for the last hour before doing this, and give the two drivers disjoint --only bands so
    # they cannot both claim the same cell.
    ap.add_argument("--only", default=None,
                    help="comma-separated job kinds to run (strip, abl, nn, ap, specap); "
                         "default runs every band")
    args = ap.parse_args()

    devs = [int(d) for d in args.workers.split(",")]
    js = jobs()
    if args.only:
        kinds = set(args.only.split(","))
        unknown = kinds - {j["kind"] for j in js}
        if unknown:
            sys.exit(f"--only: unknown job kind(s) {sorted(unknown)}; "
                     f"known kinds are {sorted({j['kind'] for j in js})}")
        js = [j for j in js if j["kind"] in kinds]
    todo = [j for j in js if args.no_resume or not done(j)]
    print(f"{len(js)} cells, {len(js) - len(todo)} already done, {len(todo)} to run")
    for kind, name in (("strip", "K-scaling"), ("abl", "ablation"), ("nn", "nnU-Net"),
                       ("ap", "instance AP"), ("specap", "specialist AP")):
        cells = [j for j in todo if j["kind"] == kind]
        if cells:
            print(f"  {name}: {len(cells)} cells (~{sum(j['cost'] for j in cells) / 60:.0f} GPU-h)")
    print(f"  {len(devs)} workers on devices {devs} -> ~"
          f"{sum(j['cost'] for j in todo) / 60 / len(devs):.0f} h wall clock if perfectly packed")

    # Preflight EVERY distinct command shape before dispatching any of them. The cost of a rejected
    # method token or a flag the target script does not define is otherwise paid 24 hours in, as a
    # hole in the figure rather than as an error.
    cmds = [cmd_for(j, devs[0]) for j in todo]
    problems = preflight(cmds)
    if problems:
        print("PREFLIGHT FAILED -- nothing dispatched:")
        for p in problems:
            print(f"   {p}")
        return 2
    print("preflight ok")

    if args.dry_run:
        for j in todo[:6]:
            print(f"  [{j['label']}/{j['ds']}] {cmd_for(j, devs[0])}")
        print(f"  ... and {max(0, len(todo) - 6)} more")
        return 0

    q: Queue = Queue()
    for j in todo:
        q.put(j)
    failures, lock = [], threading.Lock()

    def worker(wid, dev):
        while not q.empty():
            try:
                j = q.get_nowait()
            except Exception:
                return
            tag = f"{j['label']}/{j['ds']}"
            if not args.no_resume and done(j):
                print(f"  [w{wid}/gpu{dev}] skip (already done) {tag}", flush=True)
                continue
            need = min_free_mib(j["label"], j["k"], j["ds"])
            waited = 0
            while _free_mib(dev) < need and waited < MAX_WAIT_S:
                time.sleep(60)
                waited += 60
            if _free_mib(dev) < need:
                with lock:
                    failures.append(f"{tag}: gpu{dev} never freed {need} MiB in {MAX_WAIT_S}s")
                continue
            c = cmd_for(j, dev)
            t0 = time.time()
            print(f"  [w{wid}/gpu{dev}] start {tag}", flush=True)
            r = subprocess.run(c, shell=True, cwd=ROOT, capture_output=True, text=True,
                               env={**os.environ, "PYTHONPATH": ROOT})
            mins = (time.time() - t0) / 60
            if r.returncode != 0 or not done(j):
                with lock:
                    failures.append(f"{tag}: rc={r.returncode} :: {r.stderr.strip()[-400:]}")
                print(f"  [w{wid}/gpu{dev}] FAIL {tag} ({mins:.0f} min) rc={r.returncode}", flush=True)
            else:
                print(f"  [w{wid}/gpu{dev}] done {tag} ({mins:.0f} min)", flush=True)

    ts = [threading.Thread(target=worker, args=(i, d), daemon=True) for i, d in enumerate(devs)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    # A failed cell is never silently skipped: a missing directory is a missing row in the ablation
    # or a missing point on the curve, and both look like a deliberate omission to a reader.
    if failures:
        print(f"\n{len(failures)} FAILED CELLS:")
        for f in failures:
            print(f"   {f}")
        return 1
    print("\nall cells complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
