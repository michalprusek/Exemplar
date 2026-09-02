"""Benchmark harness: run AL-vs-random and write reproducible artifacts.

Writes to ``<out_dir>/<name>/``: ``curve.csv`` (per-round fg IoU + instance AP for
both arms), ``curve.png`` (the AL-vs-random curves), ``config.json`` (the run
config), and ``git_sha.txt`` (provenance). Every number is reproducible from these.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from active_segmenter.loop.orchestrator import run_al_vs_random  # noqa: E402


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run(cfg, pool, test, name, rounds=10, cold_k=3, out_dir="results",
        encoder=None, refiner=None, seed=0) -> dict:
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    res = run_al_vs_random(cfg, pool, test, rounds, cold_k,
                           encoder=encoder, refiner=refiner, seed=seed)
    al, rnd = res["al"], res["random"]

    with open(os.path.join(d, "curve.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n_annotated", "al_fg_iou", "al_ap", "random_fg_iou", "random_ap"])
        for a, b in zip(al, rnd):
            w.writerow([a.n_annotated, round(a.fg_iou, 4), round(a.instance_ap, 4),
                        round(b.fg_iou, 4), round(b.instance_ap, 4)])

    _plot(d, name, al, rnd)
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(cfg.to_dict(), fh, indent=2)
    with open(os.path.join(d, "git_sha.txt"), "w") as fh:
        fh.write(_git_sha() + "\n")
    return {"al": al, "random": rnd, "dir": d,
            "csv": os.path.join(d, "curve.csv"), "png": os.path.join(d, "curve.png")}


def _plot(d, name, al, rnd):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ns = [r.n_annotated for r in al]
    ax[0].plot(ns, [r.fg_iou for r in al], "-o", label="AL")
    ax[0].plot([r.n_annotated for r in rnd], [r.fg_iou for r in rnd], "-s", label="random")
    ax[0].set_title(f"{name} — foreground IoU")
    ax[1].plot(ns, [r.instance_ap for r in al], "-o", label="AL")
    ax[1].plot([r.n_annotated for r in rnd], [r.instance_ap for r in rnd], "-s", label="random")
    ax[1].set_title(f"{name} — instance AP")
    for a in ax:
        a.set_xlabel("# annotated images (oracle reveals)")
        a.legend()
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "curve.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
