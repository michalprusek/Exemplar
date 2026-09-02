"""Consolidate every number the ISBI paper needs, with seed-level error bars, from the score JSONs on tulen.
per_image is a flat list of len(seeds)*test_per_seed; per-seed dataset mean = chunk mean. Prints:
 (1) ablation means per dataset for best / cgate / film / cgate_film,
 (2) K=8 best_v2 per-dataset mean +/- std (native metric),
 (3) specialist-table AP for ours (monuseg, ctc_u373, dsb2018),
 (4) K-scaling over the fixed common-5 subset present at all K, per method, mean +/- std across seeds."""
import glob
import json
import os

import numpy as np

ROOT = "/disk1/prusek/active-segmenter/results"
COMMON5 = ["spheroidj", "dsb2018", "monuseg", "drive", "hrf"]


def per_seed_means(path):
    d = json.load(open(path))
    pi = np.asarray(d["per_image"], float)
    ns = len(d["seeds"]); t = d["test_per_seed"]
    if ns * t != len(pi):                            # FAIL-LOUD: malformed/partial file, do not guess a mean
        raise ValueError(f"{path}: per_image len {len(pi)} != {ns}*{t}; malformed score file")
    return d["metric"], pi.reshape(ns, t).mean(1)


def ds_of(path):
    return os.path.basename(path).split("__")[1].rsplit(".json", 1)[0]


def collect(dirpath):
    """dataset -> (metric, per-seed-mean array). Errors on a duplicate dataset in one dir (ambiguous)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(dirpath, "*.json"))):  # deterministic
        ds = ds_of(f)
        if ds in out:
            raise ValueError(f"duplicate {ds} files in {dirpath}; ambiguous which to use")
        out[ds] = per_seed_means(f)
    return out


def fmt(arr):
    return f"{arr.mean():.3f}+/-{arr.std(ddof=1):.3f}"  # sample SD, consistent with the paper tables


print("=" * 70)
print("(1) ABLATION  (per-dataset seed-mean, native metric)")
configs = {"best": "scores_fact/best", "cgate": "scores_fact/cgate",
           "film": "scores_fact/film", "cgate_film(best_v2)": "scores_fact/cgate_film"}
tab = {}
for name, d in configs.items():
    tab[name] = collect(os.path.join(ROOT, d))
allds = sorted({ds for c in tab.values() for ds in c})
hdr = f"{'dataset':14} " + " ".join(f"{n:22}" for n in configs)
print(hdr)
for ds in allds:
    row = f"{ds:14} "
    for n in configs:
        if ds in tab[n]:
            met, arr = tab[n][ds]
            row += f"{met}:{arr.mean():.3f}".ljust(23)
        else:
            row += "-".ljust(23)
    print(row)
# overall mean over the 6 datasets common to all configs
common = sorted(set.intersection(*[set(c) for c in tab.values()]))
print(f"\n  overall mean over {common}:")
for n in configs:
    vals = [tab[n][ds][1].mean() for ds in common]
    print(f"    {n:22}: {np.mean(vals):.3f}")

print("=" * 70)
print("(2) K=8 best_v2 per-dataset  mean +/- std  (native metric)")
k8 = collect(os.path.join(ROOT, "scores_fact/cgate_film"))
hrfk8 = collect(os.path.join(ROOT, "scores_hrfk8"))
k8.update(hrfk8)
for ds in ["spheroidj", "dsb2018", "monuseg", "ctc_u373", "drive", "hrf", "microtubules"]:
    if ds in k8:
        met, arr = k8[ds]
        print(f"  {ds:14} {met:8} {fmt(arr)}")
    else:
        print(f"  {ds:14} (missing)")

print("=" * 70)
print("(3) SPECIALIST-TABLE AP for ours (mean +/- std)")
for ds in ["monuseg", "ctc_u373", "dsb2018"]:
    if ds in k8:
        met, arr = k8[ds]
        print(f"  {ds:14} {met:8} {fmt(arr)}")

print("=" * 70)
print("(4) K-SCALING over fixed common-5 = %s (each dataset's primary metric)" % COMMON5)
OURS = {1: ["scores_kscale/k1_v2"], 4: ["scores_kscale/k4_v2"],
        8: ["scores_fact/cgate_film", "scores_hrfk8"],
        16: ["scores_kscale/k16_v2", "scores_kscale/k16_v2nb"]}
BASE = {
    "tyche": {k: [f"scores_basekscale/k{k}_tyche"] for k in (1, 4, 8, 16)},
    "universeg": {k: [f"scores_basekscale/k{k}_universeg"] for k in (1, 4, 8, 16)},
    "insid3": {k: [f"scores_basekscale/k{k}_insid3guided"] for k in (1, 4, 8, 16)},
    "matcher": {1: ["scores_basekscale/k1_matcher"]},
}


def kscale_agg(dirs):
    """merge dirs, then for each seed average the common-5 dataset-means -> (agg mean, agg std, coverage).
    RIGOR: requires the FULL common-5 (like make_final_kscale) and equal seed counts; never a partial subset."""
    merged = {}
    for d in dirs:
        for ds, val in collect(os.path.join(ROOT, d)).items():
            merged.setdefault(ds, val)               # first dir wins (prioritised), deterministic
    present = [ds for ds in COMMON5 if ds in merged]
    if len(present) < len(COMMON5):                  # do NOT average a partial subset
        return None
    arrs = [merged[ds][1] for ds in present]
    if len({len(a) for a in arrs}) != 1:             # FAIL-LOUD: no silent seed truncation
        raise ValueError(f"unequal seed counts in kscale_agg: {list(zip(present, [len(a) for a in arrs]))}")
    M = np.stack(arrs)                               # (datasets, seeds) — equal length
    per_seed_agg = M.mean(0)                         # (seeds,)
    return per_seed_agg.mean(), per_seed_agg.std(ddof=1), present


print("\n  OURS (best_v2):")
for k in (1, 4, 8, 16):
    r = kscale_agg(OURS[k])
    if r:
        print(f"    K={k:2}: {r[0]:.3f}+/-{r[1]:.3f}  (n_ds={len(r[2])}: {r[2]})")
for meth, ks in BASE.items():
    print(f"\n  {meth}:")
    for k in sorted(ks):
        r = kscale_agg(ks[k])
        if r:
            print(f"    K={k:2}: {r[0]:.3f}+/-{r[1]:.3f}  (n_ds={len(r[2])}: {r[2]})")
