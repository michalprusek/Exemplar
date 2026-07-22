#!/usr/bin/env python
"""Regenerate the paper's component-ablation table from the fixed harness.

`run_campaign.py` schedules the full method and every baseline. It schedules NONE of the ablation
arms, so Table 2 of the paper would keep numbers from the defective harness while every table around
it was regenerated -- which is how a stale number survives a careful rewrite.

The arms map onto method tokens `al_testbed.make_backend` already understands. Mapping from the
paper's row labels:

    Backbone only (no priors)   head_fusion_best_nocls_nobank    (`nocls` zeroes the classical bank)
    + classical prior bank      head_fusion_best_nobank
    + competitive gate          head_fusion_best_cgate_nobank
    + FiLM                      head_fusion_best_film_nobank
    + gate and FiLM (full)      head_fusion_best_cgate_film_nobank   <- this IS `ours`, NOT re-run

The full method is deliberately absent: it is the campaign's `ours` row at K=8, and re-running it
here would produce a second record of the same configuration under a different label, which is
exactly the duplicate that `stats()` refuses.

`coarseonly` is a fifth arm, not in the paper today. It answers the effective-resolution objection a
reviewer will raise -- our method reads the backbone at a coarse whole-image scale AND a native one,
while INSID3 runs at 1024 and SegGPT at 448 -- by showing what the method scores with the native
pathway removed.

PROTOCOL MUST MATCH THE CAMPAIGN EXACTLY. An ablation measured under a different number of seeds, a
different test slice or a different resolution is not comparable to the method row it is ablating,
and the difference would be invisible in the table. The constants below are therefore imported from
`run_campaign` rather than restated.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from queue import Queue

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_campaign as C           # noqa: E402  — one source of truth for the protocol

# Paper Table 2 reports the mean over the six datasets it evaluates, so the ablation runs on those.
DATASETS = ["spheroidj", "dsb2018", "monuseg", "ctc_u373", "drive", "hrf"]
K = 8                                             # the table is "component ablation at K=8"

ARMS = {
    "abl_nocls":       "head_fusion_best_nocls_nobank",
    "abl_bank":        "head_fusion_best_nobank",
    "abl_cgate":       "head_fusion_best_cgate_nobank",
    "abl_film":        "head_fusion_best_film_nobank",
    "abl_coarseonly":  "head_fusion_best_cgate_film_nobank_coarseonly",
    # SELF-CONFIGURATION ablation (the paper's novelty): hold the full architecture (bank + gate + FiLM)
    # and disable one self-config axis at a time from the full method. The full arm is the campaign's
    # `ours` at K=8 and "- FiLM" is the existing `abl_cgate` (head_fusion_best_cgate_nobank), so only
    # these three are new. `noloss`/`nocolor` flip the adaptive-loss and colour/CLAHE flags to their
    # existing OFF states (fixed loss weights; grayscale input) — a fair like-for-like comparison.
    "abl_sc_noloss":   "head_fusion_best_cgate_film_nobank_noloss",    # - adaptive-loss constructor
    "abl_sc_nocolor":  "head_fusion_best_cgate_film_nobank_nocolor",   # - input channel selection + CLAHE
    "abl_sc_none":     "head_fusion_best_cgate_nobank_noloss_nocolor", # - all self-config (loss+input+FiLM)
    # SIMPLIFICATION A/B (user): instead of the closed-form colour-channel SELECTION, disable it and feed
    # the raw R,G,B as native features so the head/gate weight colour itself. Compared against `ours`
    # (selection ON) and `abl_sc_nocolor` (grayscale, no colour) → selection vs grayscale vs raw-RGB.
    "abl_rgbfeat":     "head_fusion_best_cgate_film_nobank_nocolor_rgbfeat",
}


def jobs(arms):
    out = []
    for label in arms:
        for ds in DATASETS:
            if K > C.pool_for(ds):                # ctc_u373 has 15; K=8 fits, but declare the rule
                continue
            out.append(dict(label=label, method=ARMS[label], ds=ds))
    return out


SMOKE_DIR = os.environ.get("ASG_ABL_SMOKE_TREE", "/tmp/ablation_smoke")


def cmd_for(j, smoke=False):
    """The command for one job. In smoke mode BOTH the cost and the DESTINATION change.

    A smoke run must not write into the reported tree. Reducing only the dataset list, as the first
    version of this did, still ran the full ten-seed job and wrote a real record -- which is not a
    smoke test at all, and worse, a genuinely reduced one would leave a one-seed record sitting in
    the campaign tree looking exactly like a measurement.
    """
    if smoke:
        sd = f"{SMOKE_DIR}/{j['label']}_k{K}"
        seeds, test = 1, 4
    else:
        sd = f"{C.TREE}/{j['label']}_k{K}"
        seeds, test = C.SEEDS, 10000
    # --metric_override, computed EXACTLY as run_campaign computes it. Omitting it scores the
    # instance datasets (dsb2018, monuseg, ctc_u373) as instance AP while the `ours` row this
    # ablation ablates is scored as foreground IoU, so stats() skips every comparison on those three
    # as METRIC MISMATCH -- half the ablation table, silently. Found by running the real campaign
    # records through stats and reading the MIXED METRICS warning, not by inspection.
    #
    # This costs nothing: sota_final also records the dataset's NATIVE metric alongside the
    # overridden one, so the instance-AP numbers the paper's Table 2 quotes for MoNuSeg are still
    # written, under `per_image_native`.
    override = "cldice" if C.PANEL[j["ds"]].metric == "cldice" else "fg_iou"
    # --res 672 and the campaign's seeds/test: an ablation at a different protocol is not comparable
    # to the `ours` row it ablates.
    return (f"{C.PY} {C.ROOT}/scripts/sota_final.py run --method {j['method']} "
            f"--datasets {j['ds']} --support {K} --pool {C.pool_for(j['ds'])} --test {test} "
            f"--seeds {seeds} --res 672 --metric_override {override} "
            f"--cache {C.FEAT_CACHE} --score_dir {sd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", default="0", help="comma-separated CUDA device ids, one per worker")
    ap.add_argument("--only", default="", help="comma-separated arm labels")
    ap.add_argument("--log_dir", default=os.environ.get("ASG_ABL_LOGS", "/tmp/ablation_logs"))
    ap.add_argument("--smoke", action="store_true",
                    help="one dataset, one seed, per arm — proves every arm RUNS before the real one")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    arms = list(ARMS)
    if args.only:
        want = [a.strip() for a in args.only.split(",") if a.strip()]
        bad = [a for a in want if a not in ARMS]
        if bad:
            raise SystemExit(f"unknown arm(s) {bad}; known: {sorted(ARMS)}")
        arms = want

    js = jobs(arms)
    if args.smoke:
        # The cheapest possible proof that each arm not only RESOLVES but runs end to end and writes
        # a record. Smoke tests have already caught four separate defects in this campaign; ten
        # minutes here is worth more than a multi-hour run that dies on arm three. One dataset, one
        # seed, four test images, and a throwaway score dir -- see cmd_for.
        js = [j for j in js if j["ds"] == "spheroidj"]

    print(f"{len(js)} ablation jobs ({len(arms)} arms x {len(DATASETS)} datasets at K={K})",
          flush=True)

    # Preflight: every --method must resolve in make_backend. A token typo produces a job that dies
    # on argparse after the queue has already started, and `run_campaign` gained this check only
    # after all 40 jobs for the paper's own method were dispatched with a name that did not exist.
    # It is REUSED, not reimplemented — the arms here are exactly the kind of composed token string
    # (`head_fusion_best_nocls_nobank`) whose validation must not drift from the parser.
    problems = C.preflight([cmd_for(j, args.smoke) for j in js])
    if problems:
        print("PREFLIGHT FAILED:", flush=True)
        for p in problems:
            print(f"  - {p}", flush=True)
        raise SystemExit(1)
    print("preflight: ok", flush=True)

    if args.dry_run:
        for j in js:
            print(f"  {j['label']:<16} {j['method']:<48} {j['ds']}")
        return
    if args.smoke:
        print(f"SMOKE: 1 seed, 4 test images, throwaway score dir {SMOKE_DIR} — these records are "
              f"NOT results and must never reach the campaign tree", flush=True)

    os.makedirs(args.log_dir, exist_ok=True)
    q: Queue = Queue()
    for j in js:
        q.put(j)
    failures, lock = [], threading.Lock()

    def worker(dev):
        while True:
            try:
                j = q.get_nowait()
            except Exception:
                return
            tag = f"{j['label']}_{j['ds']}"
            log = os.path.join(args.log_dir, f"{tag}.log")
            # Same GPU headroom guard as the campaign: a job launched into a full GPU OOMs in
            # seconds and chews the queue. Requeue rather than record a failure.
            waited = 0
            while C._free_mib(dev) < C.MIN_FREE_MIB and waited < C.MAX_WAIT_S:
                time.sleep(30)
                waited += 30
            if C._free_mib(dev) < C.MIN_FREE_MIB:
                with lock:
                    print(f"  [gpu{dev}] REQUEUE {tag}: only {C._free_mib(dev)} MiB free", flush=True)
                q.put(j)
                q.task_done()
                continue
            c = f"CUDA_VISIBLE_DEVICES={dev} PYTHONPATH={C.ROOT} {cmd_for(j, args.smoke)}"
            t0 = time.time()
            r = subprocess.run(c, shell=True, stdout=open(log, "w"), stderr=subprocess.STDOUT)
            with lock:
                ok = r.returncode == 0
                print(f"  [gpu{dev}] {'ok  ' if ok else 'FAIL'} {tag} ({time.time()-t0:.0f}s)",
                      flush=True)
                if not ok:
                    failures.append(tag)
            q.task_done()

    ts = [threading.Thread(target=worker, args=(d.strip(),), daemon=True)
          for d in args.workers.split(",")]
    [t.start() for t in ts]
    [t.join() for t in ts]

    if failures:
        print(f"\nABLATION_INCOMPLETE — {len(failures)} failed: {failures}", flush=True)
        sys.exit(1)
    print("\nABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
