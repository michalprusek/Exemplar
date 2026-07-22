#!/usr/bin/env python
"""Low-budget (K=1..5) COVERAGE-vs-RANDOM acquisition study on HOMOGENEOUS data.

The deployment question (user, 2026-07-12): even when a biologist uploads a *homogeneous*
dataset and wants to train from only K=1,2,3,4,5 labels, does ACTIVE selection of those K
images add value over a RANDOM K images? Our earlier finding was "EGL ~= random on
saturated data" — but EGL is an UNCERTAINTY/gradient method, which is a *high-budget* tool
and needs a trained head (undefined at K=1). The right low-budget tool is COVERAGE/TYPICALITY
(TypiClust / ProbCover / k-center) on the frozen DINOv3 CLS features: model-free, works from
K=1, and its value is not (only) a better MEAN but a lower VARIANCE and a better WORST-CASE —
random can draw a redundant / atypical / unlucky image and waste the label; coverage cannot.

Design (faithful to deployment, no bootstrap needed):
- The pool IS the whole uploaded dataset (fixed = all available support images).
- Coverage selection is DETERMINISTIC on that pool -> one reliable pick (seeded methods run a
  few seeds to show their *tight* spread). Random is a gamble -> sampled many times -> a wide
  distribution. We compare the coverage pick against the random DISTRIBUTION.
- K=1 is enumerated EXACTLY: every singleton is scored, so random-K1's full distribution, the
  ORACLE (best single image) and the WORST single image are exact, and we see precisely where
  the "typical"/"central" pick lands relative to them.
- Scoring uses the FROZEN correspondence segmenter (Ctx.test_iou) — fast and maximally
  sensitive to support selection. --headtrials>0 optionally confirms on the deployment
  segmenter (head_fusion) for a few (K, method) cells.

Reports per (dataset, K): random mean/std/min/max, each coverage method's mean/std/min, the
gap to oracle, and — the headline — P(a random draw beats the coverage pick), i.e. how often
the gamble pays off vs. reliably picking well.

Run on tulen (GPU1 = A5000; GPU0 is the user's train job; cache on /disk2, disk1 is full):
  CUDA_VISIBLE_DEVICES=1 ~/dinov3_env/bin/python scripts/lowk_coverage_study.py \
    --datasets spheroid,rozpad --avail 40 --test 24 \
    --cache /disk2/prusek/asg_cache_lowk --out /disk2/prusek/lowk_results.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from active_segmenter.acquire import coldstart
from active_segmenter.config import EncoderConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval.registry import PANEL, load_dataset


# ---------------------------------------------------------------- selection (model-free)
def _unit(cls: np.ndarray) -> np.ndarray:
    """L2-normalize CLS embeddings so euclidean distances are comparable across datasets
    (distance in [0, 2]; cosine geometry). Coverage/typicality operate on these."""
    return cls / np.maximum(np.linalg.norm(cls, axis=1, keepdims=True), 1e-12)


def sel_random(cls_pool: np.ndarray, k: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return list(rng.choice(len(cls_pool), size=min(k, len(cls_pool)), replace=False))


def sel_typiclust(cls_pool: np.ndarray, k: int, seed: int) -> list[int]:
    # cluster into k, pick the most TYPICAL (highest local density) point per cluster
    return coldstart.typiclust(cls_pool, k, seed=seed)


def sel_probcover(cls_pool: np.ndarray, k: int, seed: int) -> list[int]:
    """Greedy ball-coverage of the pool. Radius is ADAPTIVE — the 40th percentile of pairwise
    distances in THIS pool — so it works regardless of how tight/spread the upload is (the
    fixed 0.5 in coldstart.probcover is scale-dependent)."""
    n = len(cls_pool)
    if k >= n:
        return list(range(n))
    d = np.linalg.norm(cls_pool[:, None, :] - cls_pool[None, :, :], axis=2)
    iu = np.triu_indices(n, 1)
    radius = float(np.percentile(d[iu], 40)) if len(iu[0]) else 0.5
    covered = np.zeros(n, bool)
    picks: list[int] = []
    rng = np.random.default_rng(seed)
    for _ in range(k):
        gain = ((d <= radius) & ~covered[None, :]).sum(1).astype(float)
        gain[picks] = -1
        picks.append(int(np.argmax(gain)) if gain.max() > 0
                     else int(rng.choice([i for i in range(n) if i not in picks])))
        covered |= d[picks[-1]] <= radius
    return picks


def sel_kcenter(cls_pool: np.ndarray, k: int, seed: int = 0) -> list[int]:
    """k-center greedy (farthest-point), seeded at the MEDOID so K=1 = the central/typical
    image and larger K spread to cover the manifold. Deterministic."""
    n = len(cls_pool)
    if k >= n:
        return list(range(n))
    d = np.linalg.norm(cls_pool[:, None, :] - cls_pool[None, :, :], axis=2)
    first = int(np.argmin(d.sum(1)))              # medoid = most central point
    sel = [first]
    mind = d[first].copy()
    while len(sel) < k:
        nxt = int(np.argmax(mind))
        sel.append(nxt)
        mind = np.minimum(mind, d[nxt])
    return sel


COVERAGE = {"typiclust": sel_typiclust, "probcover": sel_probcover, "kcenter": sel_kcenter}
SEEDED = {"typiclust", "probcover"}               # kcenter is deterministic


# ---------------------------------------------------------------- study
def build_ctx(spec, args, dev):
    from scripts.al_testbed import Ctx

    support, test = load_dataset(spec, args.avail, args.test, seed=0)
    if len(support) < 6 or len(test) < 3:
        raise RuntimeError(f"too few images (support={len(support)} test={len(test)})")
    cfg = RunConfig(device="auto", cache_dir=args.cache,
                    encoder=EncoderConfig(model_id=args.model, resolution=args.res))
    enc = CachedEncoder(cfg, dev, args.cache)
    trf = [enc.extract(im) for im, _ in support]
    tef = [enc.extract(im) for im, _ in test]
    cls = np.stack([enc.extract_cls(im) for im, _ in support])
    ctx = Ctx(trf, support, tef, test, cls, dev, metric=spec.metric)
    return ctx, cfg, len(support), len(test)


def run_dataset(name, spec, args, dev):
    ctx, cfg, nsup, ntest = build_ctx(spec, args, dev)
    clsU = _unit(np.asarray(ctx.cls, np.float32))     # pool = ALL available support images
    pool = list(range(nsup))

    def score(sel_local):                             # local pool idx -> frozen fg-IoU
        return float(ctx.test_iou(ctx.bank([pool[i] for i in sel_local])))

    head_be = None
    if args.headtrials > 0:
        from scripts.al_testbed import make_backend
        head_be = make_backend(args.segmenter_head, cfg, dev)

    def score_head(sel_local):
        ctx.set_backend(head_be)
        return float(ctx.test_iou_backend([pool[i] for i in sel_local]))

    out = {"dataset": name, "note": spec.note, "metric": "iou(frozen-corr)",
           "n_support": nsup, "n_test": ntest, "ks": {}}
    print(f"\n########## {name} ({spec.note}) support={nsup} test={ntest} ##########", flush=True)

    for k in args.ks:
        rec = {}
        # ---- K=1: exact enumeration of every singleton (random dist / oracle / worst are exact)
        if k == 1:
            singles = np.array([score([i]) for i in range(nsup)])
            rec["random"] = _dist_stats(singles)
            rec["oracle"] = float(singles.max())
            rec["worst"] = float(singles.min())
            rand_draws = singles                      # random-K1 == uniform over singletons
        else:
            rand_draws = np.array([score(sel_random(clsU, k, s)) for s in range(args.trand)])
            rec["random"] = _dist_stats(rand_draws)
            rec["oracle"] = None
            rec["worst"] = float(rand_draws.min())

        # ---- coverage methods
        for mname, fn in COVERAGE.items():
            seeds = range(args.tcov) if mname in SEEDED else range(1)
            picks = [fn(clsU, k, s) for s in seeds]
            scores = np.array([score(p) for p in picks])
            st = _dist_stats(scores)
            st["beats_random_frac"] = float(np.mean(rand_draws < scores.mean()))
            st["delta_mean_vs_rand"] = float(scores.mean() - rand_draws.mean())
            st["delta_worst_vs_rand"] = float(scores.min() - rand_draws.min())
            rec[mname] = st

        # ---- optional deployment-segmenter (head_fusion) confirmation on a few cells
        if head_be is not None and k in args.head_ks:
            hr = np.array([score_head(sel_random(clsU, k, s)) for s in range(args.headtrials)])
            ht = score_head(sel_typiclust(clsU, k, 0))
            rec["head_random"] = _dist_stats(hr)
            rec["head_typiclust"] = float(ht)
            rec["head_typiclust_beats_random_frac"] = float(np.mean(hr < ht))

        out["ks"][k] = rec
        _print_k(k, rec)

    return out


def _dist_stats(a: np.ndarray) -> dict:
    a = np.asarray(a, float)
    return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()),
            "max": float(a.max()), "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)), "n": int(a.size)}


def _print_k(k, rec):
    r = rec["random"]
    orc = f" oracle={rec['oracle']:.3f}" if rec.get("oracle") is not None else ""
    print(f"\n  --- K={k} ---{orc}", flush=True)
    print(f"    {'random':>10}: mean {r['mean']:.3f}  std {r['std']:.3f}  "
          f"min {r['min']:.3f}  max {r['max']:.3f}  (n={r['n']})", flush=True)
    for m in COVERAGE:
        s = rec[m]
        print(f"    {m:>10}: mean {s['mean']:.3f}  std {s['std']:.3f}  min {s['min']:.3f}   "
              f"Dmean {s['delta_mean_vs_rand']:+.3f}  Dworst {s['delta_worst_vs_rand']:+.3f}  "
              f"beats {s['beats_random_frac']*100:.0f}% of random", flush=True)
    if "head_typiclust" in rec:
        hr = rec["head_random"]
        print(f"    [head] random mean {hr['mean']:.3f} std {hr['std']:.3f} | "
              f"typiclust {rec['head_typiclust']:.3f} "
              f"(beats {rec['head_typiclust_beats_random_frac']*100:.0f}% of random)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="spheroid,rozpad,dsb2018")
    ap.add_argument("--avail", type=int, default=40, help="pool size = all available support imgs")
    ap.add_argument("--test", type=int, default=24)
    ap.add_argument("--ks", default="1,2,3,4,5")
    ap.add_argument("--trand", type=int, default=40, help="random draws per K (K>=2)")
    ap.add_argument("--tcov", type=int, default=8, help="seeds for seeded coverage methods")
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--headtrials", type=int, default=0,
                    help="deployment-segmenter confirmation trials (0=off; slow, trains a head)")
    ap.add_argument("--head_ks", default="1,3,5")
    ap.add_argument("--segmenter_head", default="head_fusion_al")
    ap.add_argument("--cache", default="/disk2/prusek/asg_cache_lowk")
    ap.add_argument("--out", default="/disk2/prusek/lowk_results.json")
    args = ap.parse_args()
    args.ks = [int(x) for x in args.ks.split(",")]
    args.head_ks = {int(x) for x in args.head_ks.split(",")}

    dev = RunConfig(device="auto").device_resolved()
    names = [n for n in args.datasets.split(",") if n in PANEL]
    print(f"device={dev} datasets={names} avail={args.avail} test={args.test} "
          f"ks={args.ks} trand={args.trand} tcov={args.tcov} res={args.res} "
          f"headtrials={args.headtrials}", flush=True)

    results = []
    for name in names:
        try:
            results.append(run_dataset(name, PANEL[name], args, dev))
        except Exception as e:
            print(f"\n[{name}] SKIP — {repr(e)[:160]}", flush=True)

    _summary(results)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}\nLOWK_DONE", flush=True)


def _summary(results):
    print("\n\n===== SUMMARY: coverage vs random at low K (frozen fg-IoU) =====", flush=True)
    print("Dmean = coverage_mean - random_mean ; Dworst = coverage_min - random_min ; "
          "beats% = P(coverage pick > a random draw)")
    for res in results:
        print(f"\n[{res['dataset']}] ({res['note']})", flush=True)
        print(f"  {'K':>2} {'rand_mean':>9} {'rand_std':>8} "
              f"{'typ_Dmean':>9} {'typ_Dworst':>10} {'typ_beats%':>10} "
              f"{'kc_Dmean':>9} {'kc_beats%':>9}", flush=True)
        for k, rec in res["ks"].items():
            t, c = rec["typiclust"], rec["kcenter"]
            print(f"  {k:>2} {rec['random']['mean']:>9.3f} {rec['random']['std']:>8.3f} "
                  f"{t['delta_mean_vs_rand']:>+9.3f} {t['delta_worst_vs_rand']:>+10.3f} "
                  f"{t['beats_random_frac']*100:>9.0f}% "
                  f"{c['delta_mean_vs_rand']:>+9.3f} {c['beats_random_frac']*100:>8.0f}%",
                  flush=True)


if __name__ == "__main__":
    main()
