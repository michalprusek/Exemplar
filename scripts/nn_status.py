#!/usr/bin/env python3
"""Progress of the nnU-Net K-scaling band, at SEED granularity. Run on tulen.

WHY SEEDS AND NOT CELLS. The band is 21 cells but 210 trainings, and a cell only appears in a
directory listing once all ten of its seeds are in. Counting files therefore reports 0/21 for the
first several hours of every cell and cannot distinguish "running normally" from "wedged since
midnight". The per-seed flush added on 2026-07-31 writes the record after every seed, so the seed
count is both the true progress and the liveness signal: if it has not moved between two daily
checks, something is stuck regardless of what nvidia-smi says.

The ETA is deliberately derived from THIS band's own observed seed rate rather than from the
234 GPU-h estimate the driver prints, because that estimate is a cost model (a linear fit in field
size) and the thing we actually want to know is whether reality is tracking it.

  python3 scripts/nn_status.py            # human-readable
  python3 scripts/nn_status.py --json     # for a caller that wants to diff two checks
"""
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.environ.get("ASG_SEM_TREE", "/disk1/prusek/active-segmenter/results/final10")
LOG = os.environ.get("ASG_GAP_LOG", "/disk1/prusek/active-segmenter/gap_nn.log")
SEEDS = 10
# ctc_u373 has a 15-image pool, so K=16 cannot be drawn: the band is 11 + 10, not 11 + 11.
CELLS = {"nnunet_k4": ["spheroidj", "rozpad", "dsb2018", "monuseg", "ctc_u373", "bbbc010",
                       "bacteria", "drive", "hrf", "isbi2012em", "fisbe"],
         "nnunet_k16": ["spheroidj", "rozpad", "dsb2018", "monuseg", "bbbc010",
                        "bacteria", "drive", "hrf", "isbi2012em", "fisbe"]}


def seeds_done(band, ds):
    """Seeds present in the record, and when it was last written."""
    fs = glob.glob(os.path.join(ROOT, band, f"*__{ds}.json"))
    if not fs:
        return 0, None
    if len(fs) > 1:
        sys.exit(f"{band}/{ds}: {len(fs)} records match; a cell must resolve to exactly one")
    try:
        return len(json.load(open(fs[0])).get("seeds", [])), os.path.getmtime(fs[0])
    except json.JSONDecodeError:
        return 0, os.path.getmtime(fs[0])      # caught mid-flush; next check will read it


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


# `pgrep -f run_final_gap` MATCHES ITS OWN SHELL, because the pattern is on that shell's command
# line. That made this script report "driver ALIVE | 0 nnUNetv2_train procs" for a driver killed by a
# machine reboot -- the one state the check exists to catch. `pgrep -f` on a bracketed pattern cannot
# match the literal it is looking for, and the training count is corroborating evidence: a live driver
# always has trainings.
_drv = sh("pgrep -f 'run_final_ga[p]'")
_trn = int(sh("pgrep -c nnUNetv2_train") or 0)
state = {"checked_at": time.time(), "bands": {},
         "driver_alive": bool(_drv), "trainings": _trn}
total_done = total_want = 0
newest = 0.0

for band, names in CELLS.items():
    cells = {}
    for ds in names:
        n, mt = seeds_done(band, ds)
        cells[ds] = n
        total_done += n
        total_want += SEEDS
        if mt:
            newest = max(newest, mt)
    state["bands"][band] = cells

state["seeds_done"] = total_done
state["seeds_total"] = total_want
state["last_write_age_h"] = round((time.time() - newest) / 3600, 1) if newest else None
# A cell that dies takes its worker with it; the driver keeps running, so process liveness alone
# is not evidence of progress. Surface the failure lines instead of trusting the process table.
state["errors"] = [l for l in sh(f"grep -iE 'traceback|error|fail|died' {LOG} | tail -8").splitlines() if l]
state["log_tail"] = sh(f"tail -4 {LOG}").splitlines()

if "--json" in sys.argv:
    print(json.dumps(state, indent=1))
    sys.exit(0)

if state["driver_alive"] and _trn == 0:
    print("WARNING: driver process present but ZERO trainings — it is starting up, or wedged.\n")
print(f"driver {'ALIVE' if state['driver_alive'] else 'GONE'} | "
      f"{state['trainings']} nnUNetv2_train procs | "
      f"{total_done}/{total_want} seeds ({100 * total_done / total_want:.0f}%)")
if state["last_write_age_h"] is not None:
    print(f"last record write: {state['last_write_age_h']} h ago")
for band, cells in state["bands"].items():
    done = [d for d, n in cells.items() if n >= SEEDS]
    part = {d: n for d, n in cells.items() if 0 < n < SEEDS}
    todo = [d for d, n in cells.items() if n == 0]
    print(f"\n{band}: {len(done)}/{len(cells)} cells complete")
    if done:
        print(f"  done:    {' '.join(done)}")
    if part:
        print(f"  partial: {' '.join(f'{d}={n}/{SEEDS}' for d, n in part.items())}")
    if todo:
        print(f"  waiting: {' '.join(todo)}")
if state["errors"]:
    print("\nERRORS in the driver log:")
    for l in state["errors"]:
        print(f"  {l}")
print("\nlog tail:")
for l in state["log_tail"]:
    print(f"  {l}")
