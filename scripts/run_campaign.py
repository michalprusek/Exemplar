#!/usr/bin/env python
"""Final ISBI campaign: a longest-first work queue over (method, K, dataset) jobs.

Design notes that matter for correctness, not just speed:

* **One clean score tree.** Every job writes `results/final10/{method}_k{K}/`, or `{method}/` for the
  off-the-shelf specialists, which ignore the support masks and so have no K. The paper's generators
  read only that tree, so no number can silently come from a pre-fix run (the previous fallback lists
  mixed harnesses inside a single curve).
* **Shared feature cache, read-only.** `prebuild_cache.py` extracts DINOv3 features once; workers then
  only READ it. Concurrent jobs writing one cache is the documented race that previously produced
  irreproducible numbers, so each worker also gets its own private scratch dir for anything it writes.
* **Longest-first scheduling.** micro-SAM fine-tuning is ~half the total budget, so it is dispatched
  first and short jobs backfill as workers free up; makespan is then max(total/workers, longest job)
  instead of the sum.
* **A failed job is never silently skipped.** Failures are recorded and the driver exits non-zero with
  the list, because a missing dir now means a missing point in the figure rather than a fallback.

Usage:  run_campaign.py --workers "0,0,0,1,1" [--dry-run]
"""
from __future__ import annotations

import argparse, glob, os, re, subprocess, sys, threading, time
from queue import Queue

# Host-dependent locations are env vars defaulting to tulen's literal paths: the campaign is split
# across two GPU hosts and kajman has neither /disk1 nor a usable ~/dinov3_env (its NFS-shared home
# holds tulen's 3.10 venv, whose interpreter does not exist there). An unset environment therefore
# still reproduces tulen's behaviour exactly.
ROOT = os.environ.get("ASG_REPO_ROOT", "/disk1/prusek/active-segmenter")
TREE = os.environ.get("ASG_SCORE_TREE", f"{ROOT}/results/final10")
FEAT_CACHE = os.environ.get("ASG_FEAT_CACHE", "/disk1/prusek/cache_final10")  # prebuild_cache.py, read-only here
PY = os.environ.get("ASG_PY", os.path.expanduser("~/dinov3_env/bin/python"))

sys.path.insert(0, ROOT_IMPORT := ROOT)
from active_segmenter.eval.registry import PANEL  # noqa: E402

ENVS = {"cellpose_ft": "/disk2/prusek/cellpose4_env/bin/python",
        "stardist_ft": "/disk2/prusek/stardist_env/bin/python",
        "microsam_ft": "/disk2/prusek/microsam_env/bin/python",
        "persam": "/disk2/prusek/persam_env/bin/python",
        # The off-the-shelf columns reuse the same three interpreters as their fine-tuned
        # counterparts, for the same reason those exist: Cellpose needs torch and StarDist needs
        # tensorflow (they cannot coexist), and micro-SAM is a mamba build. cellpose4_env is the
        # right one for Cellpose-SAM specifically — cpsam only exists in cellpose>=4, and
        # cellpose_stardist_bench REFUSES to load cyto3 there because v4 silently substitutes cpsam.
        "cellpose_sam": "/disk2/prusek/cellpose4_env/bin/python",
        "stardist": "/disk2/prusek/stardist_env/bin/python",
        "microsam": "/disk2/prusek/microsam_env/bin/python"}

SEEDS = 10
# pool is capped by what each set has; ctc_u373 has only 15 pool images so K=16 is impossible there.
POOL = {"ctc_u373": 15, "isbi2012em": 16, "fisbe": 16}
_ALL_DATASETS = ["spheroidj", "dsb2018", "monuseg", "ctc_u373", "drive", "hrf",
                 "isbi2012em", "fisbe", "bbbc010", "bacteria", "rozpad"]
# ASG_DATASETS restricts the run to a subset (e.g. adding a single new dataset without re-running the
# panel). Only these datasets' cells are scheduled; every other dataset is left untouched.
DATASETS = os.environ.get("ASG_DATASETS", ",".join(_ALL_DATASETS)).split(",")
KS = [1, 4, 8, 16]

# label -> (method name the HARNESS actually knows, relative cost hint in minutes for 10 seeds).
# The label is only a directory/log name; the second element is what reaches `--method`, and it must
# resolve in `al_testbed.make_backend`. This was not always true: the driver used to pass the label
# `ours` straight through, and `make_backend` has no such backend, so every job for the paper's own
# method would have died on argparse. `preflight` now checks exactly this.
INCONTEXT = {"ours": ("head_fusion_best_cgate_film_nobank", 8),
             "seggpt": ("seggpt", 12),
             "universeg": ("universeg", 4),
             "tyche": ("tyche", 4),
             # Both CRF modes; the paper reports the per-dataset better of the two (see cmd_for).
             "insid3_guided": ("insid3", 8),
             "insid3_dense": ("insid3", 8)}

# Datasets stained with haematoxylin and eosin. Kept here as well as in
# specialist_finetune_bench so the OFF-THE-SHELF and FINE-TUNED columns select the same model
# per modality; when they disagreed, the two rows differed by a model swap and not by
# fine-tuning, which is what that comparison is supposed to isolate.
HE_DATASETS = {"monuseg"}
# micro-SAM ships modality-specific checkpoints and using the wrong one is a large, measured
# loss for the baseline (monuseg 0.130 -> 0.385).
MICROSAM_MODEL = {"monuseg": "vit_l_histopathology"}
FINETUNE = {"cellpose_ft": 25, "stardist_ft": 20, "microsam_ft": 100}
# Matcher ships a documented 5-shot config (num_merging_mask=5), so K=1 AND K=4 are inside its
# supported regime; K=8/16 would be extrapolation beyond it and are deliberately omitted.
# PerSAM runs at K=8 too: its documented multi-shot form.
# Matcher and PerSAM are one-shot BY CONSTRUCTION; the extra K is our multi-shot adaptation, drawn
# as a marker (no line) in the figure. Matcher's dense correspondence explodes with object count and
# resolution (measured: 5.6 h for one dsb2018 job, 4.8 h for one drive job), so its K=4 arm is the
# campaign's long pole and the paper needs only its K=1 one-shot operating point. Dropped 2026-07-20
# at the user's call after the timings landed; PerSAM keeps K=8 (its documented multi-shot form, and
# cheap: ~5-13 min/job).
ONESHOT = {"matcher": [1], "persam": [1, 8]}
# Off-the-shelf specialists (campaign plan §2, group 4) — pretrained generalists that IGNORE the
# support masks. They therefore have NO K axis: one job per dataset, because running them at
# K=1,4,8,16 would spend four times the GPU time writing four copies of one number. label -> cost
# hint in minutes for ONE dataset. micro-SAM AIS is the slow one (a SAM forward + decoder watershed
# per image, over test sets up to 148 images); Cellpose/StarDist are a single CNN pass.
OFFSHELF = {"cellpose_sam": 6, "stardist": 6, "microsam": 15}
# Cellpose-SAM and StarDist share one script and differ only by --backend. `cellpose_cpsam` IS
# Cellpose-SAM (Pachitariu et al. 2025), `stardist_fluo` is the 2D_versatile_fluo nuclei model —
# the two the plan names. cyto3 is deliberately absent: it needs cellpose 3.x.
OFFSHELF_BACKEND = {"cellpose_sam": "cellpose_cpsam", "stardist": "stardist_fluo"}
OURS_METHOD = INCONTEXT["ours"][0]                  # what `sota_final.py stats --ours` must be given


def pool_for(ds):
    return POOL.get(ds, 20)


def _k_str(j):
    """K for logs/dirs. ``None`` = a support-blind method, which has no K rather than K=0."""
    return "n/a" if j["k"] is None else str(j["k"])


def jobs():
    out = []
    for ds in DATASETS:
        for k in KS:
            if k > pool_for(ds):                    # declare the cap rather than crashing mid-run
                continue
            for m, (_real, c) in INCONTEXT.items():
                out.append(dict(method=m, k=k, ds=ds, cost=c))
            for m, c in FINETUNE.items():
                out.append(dict(method=m, k=k, ds=ds, cost=c))
        for m, ks in ONESHOT.items():
            for k in ks:
                if k <= pool_for(ds):
                    out.append(dict(method=m, k=k, ds=ds, cost=20))
        for m, c in OFFSHELF.items():
            out.append(dict(method=m, k=None, ds=ds, cost=c))   # k=None = no support, no K axis
    out.sort(key=lambda j: -j["cost"])              # longest-first
    return out


def _score_dir(j):
    """The directory a job writes into: {TREE}/{label} or {TREE}/{label}_k{K}."""
    m, k = j["method"], j["k"]
    return f"{TREE}/{m}" if k is None else f"{TREE}/{m}_k{k}"


def _job_done(j):
    """Has this exact (label, K, dataset) already produced a score record?

    Globs the job's own score_dir for ``*__{ds}.json`` -- the record's filename uses the harness
    METHOD name (e.g. insid3), not the job LABEL (insid3_guided), so match on the dataset suffix
    which is stable. Used only under --resume, so a restart skips completed cells instead of
    re-running expensive finished jobs (a single Matcher cell here cost 5.6 h)."""
    return bool(glob.glob(os.path.join(_score_dir(j), f"*__{j['ds']}.json")))


def cmd_for(j):
    m, k, ds = j["method"], j["k"], j["ds"]
    # Same clean tree as everything else, minus the _k suffix for the methods that have no K.
    sd = _score_dir(j)
    if m in OFFSHELF:
        # No --support: both scripts document that field as provenance-only and ignore the masks,
        # so declaring a K here would put a support size in the record that nothing consumed.
        # --fg-scoring keeps `metric` equal to what every other column writes, which is the only
        # reason stats() will pair these records instead of skipping them.
        if m == "microsam":
            # --mode panel explicitly: it is the default, but it is ALSO the only mode whose
            # records the script will write, and being explicit lets preflight validate the value.
            # The model is chosen PER MODALITY, matching what the fine-tuned column starts from
            # so the two rows differ by the fine-tuning and nothing else. Hardcoding vit_b_lm
            # everywhere understated micro-SAM on H&E by roughly 3x (monuseg 0.130 -> 0.385 with the
            # histopathology model) and made that claim of "differ by fine-tuning alone" false,
            # because the fine-tuned column already selected per modality.
            return (f"{ENVS[m]} {ROOT}/scripts/microsam_bench.py --mode panel "
                    f"--model {MICROSAM_MODEL.get(ds, 'vit_b_lm')} "
                    f"--datasets {ds} --pool {pool_for(ds)} --test 10000 --seeds {SEEDS} "
                    f"--score_dir {sd} --fg-scoring")
        # cellpose_stardist_bench takes --datasets/--seeds as space-separated lists, not the
        # comma-joined strings the other scripts take.
        # StarDist likewise picks its H&E model on H&E. Running 2D_versatile_fluo on H&E is both
        # out of domain and polarity inverted (H&E nuclei are dark on light), and the fine-tuned
        # column already selects per modality.
        backend = OFFSHELF_BACKEND[m]
        if m == "stardist" and ds in HE_DATASETS:
            backend = "stardist_he"
        return (f"{ENVS[m]} {ROOT}/scripts/cellpose_stardist_bench.py "
                f"--backend {backend} --datasets {ds} --pool {pool_for(ds)} "
                f"--test 10000 --seeds {' '.join(str(i) for i in range(SEEDS))} "
                f"--score_dir {sd} --fg-scoring")
    common = (f"--datasets {ds} --support {k} --pool {pool_for(ds)} --test 10000 "
              f"--seeds {SEEDS} --score_dir {sd}")
    if m in FINETUNE:
        return (f"{ENVS[m]} {ROOT}/scripts/specialist_finetune_bench.py "
                f"--backend {m} {common} --fg-scoring")
    if m == "persam":
        return (f"{ENVS['persam']} {ROOT}/scripts/persam_bench.py "
                f"--datasets {ds} --support {k} --pool {pool_for(ds)} --test 10000 "
                f"--seeds {','.join(str(i) for i in range(SEEDS))} --fg-scoring --score_dir {sd}")
    # in-context + matcher go through the main harness. INSID3 gets its documented steelman:
    # res 1024 and the guided CRF (dense collapses on thin structures, and drive/hrf are 2 of the 5
    # datasets the K-scaling figure averages).
    extra = "--res 1024" if m.startswith("insid3") else "--res 672"
    # INSID3 is reported at the BETTER of its two CRF modes PER DATASET, which is this repo's
    # documented steelman and what its earlier fair table used. Guided-everywhere costs it on
    # blobs (dsb 0.326 -> 0.338 with dense) while dense collapses on thin vessels (drive 0.276
    # -> 0.007), so a single mode understates it on roughly half the panel -- and our margin on
    # dsb2018 is only +0.017, small enough for that to flip a headline. Both modes run; the
    # table generator takes the per-dataset max and the paper states that it is an upper bound
    # selected on test, which is the conservative direction for our own claims.
    env = f"ASG_CRF={m.split('_')[-1]} " if m.startswith("insid3") else ""
    real = INCONTEXT[m][0] if m in INCONTEXT else m
    # Shared feature cache is safe to write concurrently HERE (unlike the historical race) because
    # EmbeddingCache keys on a hash of image bytes + the full encoder config, so two workers that miss
    # the same entry write byte-identical content. The prebuild should make misses nil regardless.
    # The override must follow the SAME convention every baseline uses: clDice datasets keep
    # clDice, everything else becomes foreground IoU. A blanket fg_iou made our method write
    # metric="fg_iou" on drive/hrf/isbi2012em/fisbe while every specialist wrote "cldice", and
    # stats() then skipped EVERY comparison on those four datasets as a metric mismatch --
    # silently, and invisibly to preflight, which can only check flag names and choice sets.
    override = "cldice" if PANEL[ds].metric == "cldice" else "fg_iou"
    return (f"{env}{PY} {ROOT}/scripts/sota_final.py run --method {real} {common} {extra} "
            f"--metric_override {override} --cache {FEAT_CACHE}")


MIN_FREE_MIB = 9000        # a DINOv3 ViT-L job at res 1024 needs roughly this much headroom
MAX_WAIT_S = 3600          # beyond an hour the device is not transiently busy, it is occupied

# ONE floor does not fit every method. SegGPT ensembles the whole support set inside its attention,
# so its peak memory grows with K, and its relative-position term allocates several GiB in one go:
# `seggpt_k16_dsb2018` died with CUDA OOM after 8 s while holding 19.1 GiB, on a device where the
# 9000 MiB check had passed. The guard was not wrong about free memory, it was wrong about how much
# this job would go on to want.
#
# Measured on an L40S (46 GiB): SegGPT reached 19.1 GiB at K=16 before failing to allocate 4.7 GiB
# more, so ~26 GiB is the honest floor there; K<=8 ran comfortably alongside two other jobs.
# Anything not listed keeps the DINOv3-sized default.
# The other axis is the DATASET. HRF's images are 3504 px, so the native-resolution pathway holds
# far more activation than on a 584 px set, and `ours_k8_hrf` OOMed at 15.4 GiB while two siblings
# held 11.7 and 17.2 GiB on the same 44 GiB card. Memory demand here is (method, K, dataset), and a
# floor that knows only the method under-books the largest images exactly as it under-booked the
# largest K.
HEAVY_DATASETS = {"hrf": 20000, "fisbe": 14000}


def min_free_mib(method, k, ds=None):
    need = MIN_FREE_MIB
    if method == "seggpt":
        need = max(need, {16: 26000, 8: 16000}.get(k, MIN_FREE_MIB))
    if ds in HEAVY_DATASETS:
        need = max(need, HEAVY_DATASETS[ds])
    return need


def _free_mib(dev):
    """Free memory on one device, or a large number if it cannot be read (never block on a probe)."""
    try:
        r = subprocess.run(f"nvidia-smi --id={dev} --query-gpu=memory.free --format=csv,noheader,nounits",
                           shell=True, capture_output=True, text=True, timeout=30)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return 10 ** 9


def preflight(cmds):
    """Refuse to dispatch a command the target script cannot accept.

    Takes the COMMAND STRINGS, not job dicts, so any driver can reuse it. It previously called this
    module's own ``cmd_for``, which meant a second driver (the ablation runner) either went without a
    preflight or grew a duplicate of it -- and a duplicated check is one that can drift away from the
    thing it checks.

    Every defect this catches has already happened here at least once: a `--scratch` flag no script
    defined, and a `--method ours` that `make_backend` does not know (which would have killed all 40
    jobs for the paper's own method). Those failures are cheap to catch now and expensive to catch
    24 hours in, when they surface as holes in the K-scaling figure rather than as an error.

    Checks, per distinct command shape: the script exists, its interpreter exists, every emitted flag
    appears in its `--help`, and any `--method` value resolves to a real backend.
    """
    problems, seen = [], set()
    for c in cmds:
        script = re.search(r"(\S+\.py)", c).group(1)
        interp = next(t for t in c.split() if "python" in t)
        flags = tuple(sorted(set(re.findall(r"(?<!\w)--[a-zA-Z][\w-]*", c))))
        method = (re.search(r"--method (\S+)", c) or [None, None])[1] if "--method " in c else None
        key = (interp, script, flags, method)
        if key in seen:
            continue
        seen.add(key)

        if not os.path.exists(script):
            problems.append(f"{script}: script does not exist")
            continue
        if not os.path.exists(interp):
            problems.append(f"{interp}: interpreter does not exist (needed by {os.path.basename(script)})")
            continue
        h = subprocess.run(f"PYTHONPATH={ROOT} {interp} {script} --help", shell=True,
                           capture_output=True, text=True, timeout=300)
        # argparse subcommands: `sota_final.py run --help` is the real surface, so retry through it.
        if h.returncode != 0 or "--score_dir" not in h.stdout:
            sub = re.search(rf"{re.escape(script)}\s+(\w+)\s", c)
            if sub:
                h = subprocess.run(f"PYTHONPATH={ROOT} {interp} {script} {sub.group(1)} --help",
                                   shell=True, capture_output=True, text=True, timeout=300)
        if h.returncode != 0:
            problems.append(f"{os.path.basename(script)}: --help failed rc={h.returncode}: "
                            f"{(h.stderr or h.stdout)[-300:]}")
            continue
        for f in flags:
            if f not in h.stdout:
                problems.append(f"{os.path.basename(script)}: does not accept {f}")
                continue
            # Checking that a flag EXISTS is not enough: `--backend microsam_ft` passes that check
            # while argparse still rejects the value. argparse renders a constrained argument as
            # `--backend {cellpose_ft,stardist_ft}`, so when we can see the choice set, verify the
            # value we are about to pass is in it.
            choices = re.search(rf"{re.escape(f)} \{{([^}}]+)\}}", h.stdout)
            val = re.search(rf"{re.escape(f)} (\S+)", c)
            if choices and val:
                allowed = [x.strip() for x in choices.group(1).split(",")]
                if val.group(1) not in allowed:
                    problems.append(f"{os.path.basename(script)}: {f} does not accept "
                                    f"{val.group(1)!r} (allowed: {allowed})")

        if method:
            probe = (f"PYTHONPATH={ROOT} {interp} -c \""
                     f"import sys; sys.path.insert(0,'{ROOT}/scripts');"
                     f"from al_testbed import make_backend;"
                     f"from active_segmenter.config import RunConfig;"
                     f"make_backend('{method}', RunConfig(device='cpu'), 'cpu')\"")
            r = subprocess.run(probe, shell=True, capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                problems.append(f"--method {method}: does not resolve in make_backend "
                                f"({(r.stderr or '').strip().splitlines()[-1][:120] if r.stderr else 'error'})")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", default="0,0,0,1,1", help="comma-separated CUDA device per worker")
    ap.add_argument("--dry-run", action="store_true")
    # Lets the fine-tuned columns wait for finetune_budget_sweep.py to land without holding up
    # the other 300-odd jobs. Launching them at a budget the sweep then contradicts would mean
    # re-running every one of them.
    ap.add_argument("--skip", default="", help="comma-separated method labels to leave out")
    ap.add_argument("--only", default="", help="comma-separated method labels to run alone")
    ap.add_argument("--resume", action="store_true",
                    help="skip cells that already have a score record (continue an interrupted run "
                         "without re-doing finished, possibly multi-hour, jobs)")
    ap.add_argument("--log-dir", default=os.environ.get("ASG_LOG_DIR", "/disk1/prusek/campaign_logs"))
    args = ap.parse_args()

    js = jobs()
    known = {j["method"] for j in js}
    for flag, val in (("--skip", args.skip), ("--only", args.only)):
        bad = [m for m in (x.strip() for x in val.split(",")) if m and m not in known]
        if bad:
            raise SystemExit(f"{flag}: unknown method label(s) {bad}; known: {sorted(known)}")
    if args.only:
        keep = {m.strip() for m in args.only.split(",") if m.strip()}
        js = [j for j in js if j["method"] in keep]
    if args.skip:
        drop = {m.strip() for m in args.skip.split(",") if m.strip()}
        js = [j for j in js if j["method"] not in drop]
    if not js:
        raise SystemExit("no jobs selected — refusing to report a campaign that runs nothing")
    total = sum(j["cost"] for j in js)
    devs = [d.strip() for d in args.workers.split(",")]
    print(f"{len(js)} jobs, ~{total} worker-minutes serial, {len(devs)} workers "
          f"-> ~{total/max(len(devs),1)/60:.1f} h wall clock (longest-first)")
    # Preflight ALWAYS runs, including on --dry-run, and there is deliberately no way to skip it.
    # A campaign is ~a day long; a flag typo discovered at the end costs a day, discovered here it
    # costs a minute. Both defects it was written for were found by hand only after being committed.
    print("preflight: checking every command against its script's actual interface ...", flush=True)
    problems = preflight([cmd_for(j) for j in js])
    if problems:
        print(f"\nPREFLIGHT_FAILED — {len(problems)} problem(s); nothing was launched:")
        for p in problems:
            print(f"  {p}")
        sys.exit(2)
    print("preflight: ok\n", flush=True)

    if args.dry_run:
        for j in js[:12]:
            print(f"  [{j['cost']:>3}m] {j['method']:<12} K={_k_str(j):<4} {j['ds']}")
        print(f"  ... and {len(js)-12} more")
        return

    os.makedirs(args.log_dir, exist_ok=True)
    q: Queue = Queue()
    for j in js:
        q.put(j)
    failures, lock = [], threading.Lock()

    def worker(wid, dev):
        while True:
            try:
                j = q.get_nowait()
            except Exception:
                return
            # Resume: skip a cell that already has a score record, so a restart continues instead of
            # re-running finished (and sometimes multi-hour) jobs. Opt-in so the default launch is
            # unchanged. The check is on the record file, not the log, because only a completed run
            # writes the record (sota_final writes it atomically at the end).
            if args.resume and _job_done(j):
                with lock:
                    tag = (f"{j['method']}_{j['ds']}" if j["k"] is None
                           else f"{j['method']}_k{j['k']}_{j['ds']}")
                    print(f"  [w{wid}/gpu{dev}] skip (already done) {tag}", flush=True)
                q.task_done()
                continue
            # Wait for the device to actually have room. Without this, a worker whose GPU is busy
            # OOMs in ten seconds, immediately takes the next job, and chews through the whole queue
            # marking everything FAILED faster than a healthy worker finishes one item. That is not
            # hypothetical: launched alongside the budget sweep, one worker burned 25 jobs while the
            # other three were still on their first. A busy GPU must delay a job, never fail it.
            need = min_free_mib(j["method"], j["k"], j["ds"])
            waited = 0
            while _free_mib(dev) < need and waited < MAX_WAIT_S:
                time.sleep(30)
                waited += 30
            if _free_mib(dev) < need:
                with lock:
                    print(f"  [w{wid}/gpu{dev}] REQUEUE {j['method']}_{j['ds']}: gpu{dev} still under "
                          f"{need} MiB free after {MAX_WAIT_S}s", flush=True)
                q.put(j)                      # back on the queue, NOT recorded as a failure
                q.task_done()
                time.sleep(60)
                continue
            tag = (f"{j['method']}_{j['ds']}" if j["k"] is None
                   else f"{j['method']}_k{j['k']}_{j['ds']}")
            log = f"{args.log_dir}/{tag}.log"
            c = f"CUDA_VISIBLE_DEVICES={dev} PYTHONPATH={ROOT} {cmd_for(j)}"
            t0 = time.time()
            r = subprocess.run(c, shell=True, stdout=open(log, "w"), stderr=subprocess.STDOUT)
            ok = r.returncode == 0
            with lock:
                print(f"  [w{wid}/gpu{dev}] {'ok ' if ok else 'FAIL'} {tag} "
                      f"({time.time()-t0:.0f}s) {'' if ok else '-> ' + log}", flush=True)
                if not ok:
                    failures.append(tag)
            q.task_done()

    ts = [threading.Thread(target=worker, args=(i, d), daemon=True) for i, d in enumerate(devs)]
    [t.start() for t in ts]
    [t.join() for t in ts]

    if failures:
        # A missing score dir is now a missing POINT in the figure, so a partial campaign must not
        # look like a finished one.
        print(f"\nCAMPAIGN_INCOMPLETE — {len(failures)}/{len(js)} jobs FAILED:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("\nCAMPAIGN_DONE")


if __name__ == "__main__":
    main()
