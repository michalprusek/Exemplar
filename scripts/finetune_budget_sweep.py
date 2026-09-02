#!/usr/bin/env python
"""Calibrate the fine-tuned-specialist training budget by measuring it, not by choosing it.

The fine-tuned baselines need ONE budget, applied identically to every dataset and every K, for the
same reason our own method is not allowed per-dataset tuning. The question is what that number should
be, and "each library's documented default" is not an answer: none of these libraries documents a
recipe for eight images, and at K=8 their defaults differ by four orders of magnitude in work done
(see ``library_budget`` in ``specialist_finetune_bench.py``).

So this sweeps the budget and reports score against it. We then take the plateau. The point is not
the precise number, it is that "why did you train StarDist for N epochs?" stops being an argument we
have to win and becomes a curve a reviewer can read. If a baseline is still climbing at the largest
budget, that is a finding too, and it means the campaign budget must go up.

Calibration only. It runs at K=8 on a reduced test slice, because it selects a hyper-parameter and
must never be quoted as a result. Its score_dir is deliberately separate from the campaign tree.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from queue import Queue

ROOT = "/disk1/prusek/active-segmenter"
ENVS = {"cellpose_ft": "/disk2/prusek/cellpose4_env/bin/python",
        "stardist_ft": "/disk2/prusek/stardist_env/bin/python",
        "microsam_ft": "/disk2/prusek/microsam_env/bin/python"}

# Datasets that are NOT in the campaign panel, so nothing measured here is ever reported.
#
# This matters more than which datasets they are. Calibrating on dsb2018/spheroidj, as this first
# did, reads the budget off a curve computed on the very test images the paper reports -- selecting
# a hyper-parameter on test, which is the exact failure the rest of this harness exists to prevent,
# and it would have been ours rather than a baseline's. Choosing sets that appear nowhere in the
# paper removes the objection entirely.
#
# Both are semantic rather than instance, because the registry has no instance dataset outside the
# reported panel. That is acceptable: the question here is where TRAINING saturates, and foreground
# quality answers it. Vessel and membrane sets are excluded on purpose, since a specialist scores
# near zero there at any budget and so carries no information about saturation.
DATASETS = ["spheroid", "rozpad"]
BUDGETS = [25, 50, 100, 200, 400]
POOL = {"ctc_u373": 15, "isbi2012em": 16, "fisbe": 16}


def jobs(backends, seeds, test):
    out = []
    for b in backends:
        for ds in DATASETS:
            for e in BUDGETS:
                # micro-SAM's cost is linear in epochs x K and it is ~30x the others per step, so the
                # top budget is skipped for it rather than silently dominating the whole sweep.
                # micro-SAM costs ~1.5 s per iteration and library_budget gives it epochs x K
                # iterations, so e=200 at K=8 is ~40 min PER SEED. Calibration must not cost more
                # than the campaign it calibrates, so its curve stops at 100 and the larger budgets
                # are read off the two cheap backends, which saturate at the same place or earlier.
                if b == "microsam_ft" and e > 100:
                    continue
                out.append(dict(backend=b, ds=ds, epochs=e, seeds=seeds, test=test,
                                cost=e * (30 if b == "microsam_ft" else 1)))
    out.sort(key=lambda j: -j["cost"])
    return out


def cmd_for(j, sweep_dir):
    sd = os.path.join(sweep_dir, f"{j['backend']}_e{j['epochs']}_{j['ds']}")
    return (f"{ENVS[j['backend']]} {ROOT}/scripts/specialist_finetune_bench.py "
            f"--backend {j['backend']} --datasets {j['ds']} --support 8 "
            f"--pool {POOL.get(j['ds'], 20)} --test {j['test']} --seeds {j['seeds']} "
            f"--epochs {j['epochs']} --score_dir {sd} --fg-scoring")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", default="0,1")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--test", type=int, default=10, help="reduced: this calibrates, it never reports")
    ap.add_argument("--sweep_dir", default="/disk1/prusek/ft_budget_sweep")
    ap.add_argument("--backends", default="cellpose_ft,stardist_ft,microsam_ft")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    unknown = [b for b in backends if b not in ENVS]
    if unknown:
        raise SystemExit(f"unknown backend(s) {unknown}; known: {sorted(ENVS)}")

    js = jobs(backends, args.seeds, args.test)
    devs = [d.strip() for d in args.workers.split(",")]
    print(f"{len(js)} sweep jobs over budgets {BUDGETS} x {DATASETS} x {backends}", flush=True)
    if args.dry_run:
        for j in js:
            print(f"  {j['backend']:<13} e={j['epochs']:<4} {j['ds']}")
        return

    os.makedirs(args.sweep_dir, exist_ok=True)
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
            tag = f"{j['backend']}_e{j['epochs']}_{j['ds']}"
            log = os.path.join(args.sweep_dir, f"{tag}.log")
            c = f"CUDA_VISIBLE_DEVICES={dev} PYTHONPATH={ROOT} {cmd_for(j, args.sweep_dir)}"
            t0 = time.time()
            r = subprocess.run(c, shell=True, stdout=open(log, "w"), stderr=subprocess.STDOUT)
            with lock:
                ok = r.returncode == 0
                print(f"  [gpu{dev}] {'ok  ' if ok else 'FAIL'} {tag} ({time.time()-t0:.0f}s)", flush=True)
                if not ok:
                    failures.append(tag)
            q.task_done()

    ts = [threading.Thread(target=worker, args=(d,), daemon=True) for d in devs]
    [t.start() for t in ts]
    [t.join() for t in ts]

    # Collect into a budget x backend table. A FAILED cell is reported as such and never silently
    # read as a low score, which would look exactly like saturation.
    table = {}
    for j in js:
        sd = os.path.join(args.sweep_dir, f"{j['backend']}_e{j['epochs']}_{j['ds']}")
        fp = os.path.join(sd, f"{j['backend']}__{j['ds']}.json")
        if not os.path.exists(fp):
            table[(j["backend"], j["ds"], j["epochs"])] = None
            continue
        with open(fp) as f:
            rec = json.load(f)
        n = int(rec["test_per_seed"])
        vals = [sum(rec["per_image"][i * n:(i + 1) * n]) / n
                for i in range(len(rec["per_image"]) // n)]
        table[(j["backend"], j["ds"], j["epochs"])] = sum(vals) / len(vals)

    print("\n===== SCORE vs FINE-TUNING BUDGET (support-epochs, K=8) =====", flush=True)
    for b in backends:
        for ds in DATASETS:
            cells = []
            for e in BUDGETS:
                v = table.get((b, ds, e), "n/a")
                cells.append(f"{e}:{'FAIL' if v is None else (v if isinstance(v, str) else f'{v:.4f}')}")
            print(f"{b:<14} {ds:<11} " + "  ".join(f"{c:>12}" for c in cells), flush=True)
    print("\nPick the smallest budget at which each backend has plateaued, use it for EVERY dataset\n"
          "and every K in the campaign, and report this table so the choice is auditable.", flush=True)
    with open(os.path.join(args.sweep_dir, "summary.json"), "w") as f:
        json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in table.items()}, f, indent=2)

    if failures:
        print(f"\nSWEEP_INCOMPLETE — {len(failures)} failed: {failures}", flush=True)
        sys.exit(1)
    print("\nSWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
