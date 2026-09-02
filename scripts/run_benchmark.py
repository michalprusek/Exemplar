#!/usr/bin/env python
"""Full P1 benchmark: AL-vs-random on DSB2018 (or synthetic overlap) with a chosen
refiner. Writes results/<name>/ artifacts (curve.csv, curve.png, config, git sha).

Run on tulen:
  ~/dinov3_env/bin/python scripts/run_benchmark.py --config configs/dsb2018.yaml \
      --dataset dsb2018 --refine sam --pool 60 --test 40 --rounds 8 --cold-k 3 \
      --data /disk1/prusek/dsb2018 --cache /disk1/prusek/asg_cache
"""
import argparse

from active_segmenter.config import RunConfig
from active_segmenter.eval import harness
from active_segmenter.eval.datasets import load_dsb2018, make_synthetic_overlap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dsb2018.yaml")
    ap.add_argument("--dataset", choices=["dsb2018", "overlap"], default="dsb2018")
    ap.add_argument("--refine", choices=["identity", "sam"], default=None)
    ap.add_argument("--data", default="/tmp/dsb2018")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--pool", type=int, default=60)
    ap.add_argument("--test", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--cold-k", type=int, default=3)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = RunConfig.from_yaml(args.config)
    if args.refine:
        cfg.refine.kind = args.refine
    if args.cache:
        cfg.cache_dir = args.cache

    if args.dataset == "dsb2018":
        pool = load_dsb2018(args.data, "train", args.pool)
        test = load_dsb2018(args.data, "test", args.test)
        name = f"dsb2018_{cfg.refine.kind}_res{cfg.encoder.resolution}"
    else:
        # overlap GT is a mask-list; adapt to (image, label_map) is lossy, so we keep
        # the overlap set for the dedicated overlap eval (scripts/overlap_eval.py).
        raise SystemExit("use scripts/overlap_eval.py for the overlap benchmark")

    print(f"device={cfg.device_resolved()} dataset={name} pool={len(pool)} test={len(test)} "
          f"rounds={args.rounds} cold_k={args.cold_k} strategy={cfg.acquire.strategy}", flush=True)
    from active_segmenter.encoder.cached import CachedEncoder
    from active_segmenter.refine import build_refiner

    dev = cfg.device_resolved()
    encoder = CachedEncoder(cfg, dev, cfg.cache_dir)
    refiner = build_refiner(cfg.refine, dev)
    report = harness.run(cfg, pool, test, name=name, rounds=args.rounds, cold_k=args.cold_k,
                         out_dir=args.out, encoder=encoder, refiner=refiner)
    al, rnd = report["al"], report["random"]
    print(f"\n{'n':>4} {'AL_IoU':>7} {'AL_AP':>7} {'rnd_IoU':>8} {'rnd_AP':>7}")
    for a, b in zip(al, rnd):
        print(f"{a.n_annotated:>4} {a.fg_iou:>7.3f} {a.instance_ap:>7.3f} "
              f"{b.fg_iou:>8.3f} {b.instance_ap:>7.3f}")
    print(f"\nartifacts -> {report['dir']}")
    print("DONE")


if __name__ == "__main__":
    main()
