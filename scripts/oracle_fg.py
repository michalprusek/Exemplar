"""Oracle-foreground diagnostic (traceable version of the paper's 0.886 claim): feed the GROUND-TRUTH
foreground into best_v2's training-free affinity-watershed instance decoder on MoNuSeg and score instance
average precision. This isolates the decoder from the foreground-prediction quality: it is the paper's
'given a perfect foreground mask, the decoder reaches AP 0.886, exceeding every specialist' number.
Same fixed-pool protocol as the benchmark (pool 20, test 24, 6 seeds, K=8, res 672)."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from active_segmenter.config import EncoderConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval.registry import PANEL, load_dataset
from active_segmenter.eval.scoring import primary_key, score_prediction
from active_segmenter.segment.base import LabeledExample
from active_segmenter.segment.head_fusion_backend import _affinity_watershed_instances, _ridge_map
from scripts.al_testbed import make_backend

MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
CACHE = "/disk1/prusek/asg_cache_oracle"


def main():
    dev = RunConfig(device="auto").device_resolved()
    cfg = RunConfig(device="auto", cache_dir=CACHE, encoder=EncoderConfig(model_id=MODEL, resolution=672))
    enc = CachedEncoder(cfg, dev, CACHE)
    pk = primary_key("instance_ap")                       # 'ap'
    pool, test = load_dataset(PANEL["monuseg"], 20, 24, seed=0)   # multi-draw fixed pool (load ONCE)
    aps = []
    for seed in range(6):
        be = make_backend("head_fusion_best_cgate_film_nobank", cfg, dev, enc=enc)
        sub = list(np.random.default_rng(seed).choice(len(pool), 8, replace=False))
        be.fit([LabeledExample(pool[i][0], enc.extract(pool[i][0]), np.asarray(pool[i][1])) for i in sub])
        per = []
        for im, gt in test:
            gt = np.asarray(gt)
            fg = gt > 0                                   # ORACLE: perfect foreground (union of GT instances)
            fgrid = enc.extract(im)
            ridge = _ridge_map(be._channel(im))
            inst = [m.mask for m in _affinity_watershed_instances(
                fg, ridge, fgrid, 1, r_star=be._inst_r, merge_cos=be._inst_merge_cos)]
            per.append(float(score_prediction("instance_ap", fg, gt, inst)[pk]))
        m = float(np.mean(per))
        aps.append(m)
        print(f"  seed {seed}: oracle-fg AP = {m:.3f}", flush=True)
    aps = np.array(aps)
    print(f"\nORACLE-FG MoNuSeg instance-AP (GT foreground -> affinity decoder): "
          f"{aps.mean():.3f} +/- {aps.std(ddof=1):.3f}  [paper text claims 0.886; best specialist 0.405]")


if __name__ == "__main__":
    main()
