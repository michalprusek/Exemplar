# Copied from AutoSeg scripts/c119/timing_common.py (4cd0cd9) by tools/sync_paper.py; default paths adapted, internal notes rewritten in comments and docstrings.
"""Fit and inference time of every Table 1 method on the RTX A5000 -- the shared protocol.

Table 1's last row is "fit time / inference time" for every method over all eleven datasets, measured on the idle A5000
(an idle CUDA context of another user's process is allowed when it shows no SM activity during the window).

Protocol, the same for every method (each family script imports this module; it needs numpy and the repo only):
  * the paper's split: `load_dataset(PANEL[ds], pool, 10000, seed=0)`, pool 20 (CTC-U373 15, ISBI2012-EM and FISBE 16);
  * ONE support draw, seed 0: `default_rng(0).choice(len(pool), 8, replace=False)`, as every bench draws seed 0;
  * fit: the method's own fitting call on those 8 masks, at the training length of its Table 1 column;
  * inference: the first N_INFER images of the fixed test order, each timed from the image array in memory to the
    binary mask at native resolution (GPU synchronised before and after), after ONE untimed warm-up prediction on a
    SUPPORT image (never a test image); model construction, weight loading and data loading stay outside both clocks;
  * one JSON record per (method, dataset), written atomically and NEVER over an existing file (a clean record was once
    lost to an `rm -f` before a re-run).
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
REPO = os.path.dirname(SCRIPTS)
for p in (SCRIPTS, REPO):                       # this tree first, whatever else is on the path
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

DATASETS = ["spheroidj", "rozpad", "dsb2018", "monuseg", "ctc_u373", "bbbc010", "bacteria",
            "drive", "hrf", "isbi2012em", "fisbe"]
POOL = {"ctc_u373": 15, "isbi2012em": 16, "fisbe": 16}       # the pools of every arm of the paper
K = 8
SEED = 0
N_INFER = 5


# From here to base_record/finish_record: the campaign half, imported by the research code's timing scripts (not in this
# repository). split() and metric() import active_segmenter, the research package, so this half cannot run here;
# timing_table.py uses only DATASETS and record_path.
def split(ds):
    """(pool, test) exactly as the paper's arms load them."""
    from active_segmenter.eval.registry import PANEL, load_dataset
    if ds not in PANEL:
        raise SystemExit(f"unknown dataset {ds!r}")
    return load_dataset(PANEL[ds], POOL.get(ds, 20), 10000, seed=0)


def metric(ds):
    from active_segmenter.eval.registry import PANEL
    return PANEL[ds].metric


def support(pool, k=K, seed=SEED):
    """(indices, shots) of the seed-0 draw, shots as (image, mask array) like every bench passes them."""
    idx = np.random.default_rng(seed).choice(len(pool), k, replace=False)
    return [int(i) for i in idx], [(pool[i][0], np.asarray(pool[i][1])) for i in idx]


def infer_indices(test, n=N_INFER):
    return list(range(min(n, len(test))))


def side_px(img):
    return int(max(np.asarray(img).shape[:2]))


def gpu_name():
    """Name of the card this process sees as device 0 (CUDA_VISIBLE_DEVICES applied), from nvidia-smi."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", vis],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        out = ""
    return out or "unknown"


def require_a5000(smoke):
    name = gpu_name()
    if "A5000" not in name and not smoke:
        raise SystemExit(f"device is {name!r}, not the RTX A5000: refusing (use --smoke for a functional test)")
    return name


def loadavg():
    try:
        return open("/proc/loadavg").read().split()[:3]
    except OSError:
        return None


def make_sync():
    """A callable that waits for the GPU when torch with CUDA is importable; a no-op otherwise (TensorFlow callers
    synchronise by fetching the result to the host, which their predict calls do)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.synchronize
    except ImportError:
        pass
    return lambda: None


def time_call(fn, sync):
    sync()
    t0 = time.perf_counter()
    r = fn()
    sync()
    return time.perf_counter() - t0, r


def check_mask(mask, img, ds, i):
    """The timed output must be a binary mask at native resolution, or the clock measured something else."""
    m = np.asarray(mask)
    h, w = np.asarray(img).shape[:2]
    if m.shape[:2] != (h, w):
        raise RuntimeError(f"{ds} test {i}: mask {m.shape} is not at native size {(h, w)}")
    return float((m > 0).mean())


def record_path(out_dir, method, ds):
    safe = method.replace(" ", "_").replace("/", "_")
    return os.path.join(out_dir, f"{safe}__{ds}.json")


def refuse_existing(out_dir, method, ds):
    p = record_path(out_dir, method, ds)
    if os.path.exists(p):
        raise SystemExit(f"{p} exists: a timing record is never overwritten (move it aside by hand)")
    return p


def write_record(out_dir, method, ds, rec):
    """Atomic write; refuses an existing file (checked again here, not only at start)."""
    p = refuse_existing(out_dir, method, ds)
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(rec, f, indent=1, allow_nan=False)      # a NaN timing must fail here, not reach the table
        os.link(tmp, p)                                       # fails if the record appeared meanwhile: never overwritten
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return p


def base_record(method, ds, idx, pool, test, device, versions, extra=None):
    rec = dict(method=method, dataset=ds, k=K, seed=SEED, support_idx=idx, pool=len(pool), n_test=len(test),
               side_px=side_px(test[0][0]), device=device, host=socket.gethostname(), pid=os.getpid(),
               cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), argv=sys.argv,
               date_start=time.strftime("%Y-%m-%d %H:%M:%S"), loadavg_start=loadavg(), versions=versions,
               protocol="the Table 1 timing protocol (scripts/c119/timing_common.py)")
    rec.update(extra or {})
    return rec


def finish_record(rec, fit_s, fit_kind, infer_s, infer_idx, fg, fit_detail=None, warmup=None):
    rec.update(fit_s=None if fit_s is None else round(float(fit_s), 3), fit_kind=fit_kind, fit_detail=fit_detail,
               infer_idx=infer_idx, infer_s=[round(float(t), 4) for t in infer_s],
               infer_mean_s=round(float(np.mean(infer_s)), 4), infer_fg_fraction=[round(x, 4) for x in fg],
               warmup=warmup, date_end=time.strftime("%Y-%m-%d %H:%M:%S"), loadavg_end=loadavg())
    return rec


def common_args(ap):
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--out", required=True, help="directory of the records (one JSON per method and dataset)")
    ap.add_argument("--n_infer", type=int, default=N_INFER)
    ap.add_argument("--smoke", action="store_true", help="functional test: any GPU, shortened training where the "
                    "family script supports it; records are marked smoke=True and must never reach the paper")
    return ap


def datasets_arg(s):
    names = [d.strip() for d in s.split(",") if d.strip()]
    bad = [d for d in names if d not in DATASETS]
    if bad:
        raise SystemExit(f"unknown dataset(s) {bad}")
    return names
