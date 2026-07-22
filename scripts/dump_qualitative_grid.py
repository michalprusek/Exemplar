"""Dump best_v2 predictions for the qualitative GRID figure (Fig. 2).

For each dataset it fits best_v2 (head_fusion_best_cgate_film_nobank) on K=8 support masks
(rng seed 0, exactly as the paper protocol), then for a few candidate test images saves the
NATIVE-resolution input image, the ground-truth label map, the predicted foreground mask, and
a predicted per-instance id map to one ``.npz`` per (dataset, index). The figure itself is then
composed offline (no GPU) by ``scripts/compose_qualitative_grid.py`` so the layout/vector styling
can be iterated without re-running inference.

Isolation: uses its OWN writable feature cache (``QUAL_CACHE``), never the campaign's, so it is
safe to run concurrently with the benchmark campaign (no shared-writable-cache race). Pin it to a
specific GPU with ``CUDA_VISIBLE_DEVICES``.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from active_segmenter.config import EncoderConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval.registry import PANEL, load_dataset
from active_segmenter.segment.base import LabeledExample
from scripts.al_testbed import make_backend

MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
CACHE = os.environ.get("QUAL_CACHE", "/disk1/prusek/asg_cache_qual_grid")
OUT = os.environ.get("QUAL_OUT", "/disk1/prusek/qual_grid_dump")
# one entry per dataset; INDICES are candidate test images (compose picks the nicest 6-8).
# Decay (rozpad) replaces HRF in the panel: DRIVE already covers vessels, and a decay-spheroid
# example shows the method on the morphology our prior bank was built for. Override with QUAL_DATASETS.
DATASETS = os.environ.get(
    "QUAL_DATASETS", "spheroidj,dsb2018,monuseg,ctc_u373,drive,rozpad").split(",")
INDICES = [int(x) for x in os.environ.get("QUAL_INDICES", "0,1,2").split(",")]
POOL, TEST, K, SEED = 20, 24, 8, 0


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    dev = RunConfig(device="auto").device_resolved()
    cfg = RunConfig(device="auto", cache_dir=CACHE,
                    encoder=EncoderConfig(model_id=MODEL, resolution=672))
    enc = CachedEncoder(cfg, dev, CACHE)
    print(f"device={dev} cache={CACHE} out={OUT} indices={INDICES}", flush=True)

    for ds in DATASETS:
        spec = PANEL[ds]
        pool, test = load_dataset(spec, POOL, TEST, seed=SEED)
        be = make_backend("head_fusion_best_cgate_film_nobank", cfg, dev, enc=enc)
        sub = list(np.random.default_rng(SEED).choice(len(pool), K, replace=False))
        be.fit([LabeledExample(pool[i][0], enc.extract(pool[i][0]), np.asarray(pool[i][1]))
                for i in sub])
        for idx in INDICES:
            if idx >= len(test):
                print(f"[{ds}] idx {idx} >= n_test {len(test)}, skip", flush=True)
                continue
            im, gt = test[idx]
            im = np.asarray(im)
            gt = np.asarray(gt)
            feat = enc.extract(im)                       # cached; computed once, reused
            fg = np.asarray(be.foreground(im, feat)) > 0
            # Instance decode is best-effort: on SEMANTIC datasets (binary support) the affinity
            # decoder is uncalibrated and predict() raises — the figure only needs the foreground
            # overlay, so fall back to zeros there instead of failing the whole dump.
            pred_inst = np.zeros(im.shape[:2], np.int32)
            n_inst = -1
            try:
                insts = be.predict(im, feat)
                for i, inst in enumerate(insts, 1):
                    pred_inst[np.asarray(inst.mask) > 0] = i
                n_inst = len(insts)
            except Exception as e:                        # noqa: BLE001 — decoder inactive (semantic)
                print(f"[{ds} idx={idx}] predict() inactive: {e}", flush=True)
            out = os.path.join(OUT, f"{ds}_{idx}.npz")
            np.savez_compressed(
                out, image=im, gt=gt.astype(np.int32), pred_fg=fg.astype(np.uint8),
                pred_inst=pred_inst, kind=str(spec.kind), metric=str(spec.metric), ds=ds)
            print(f"[{ds} idx={idx}] im={im.shape} gt_objs={len(np.unique(gt)) - 1} "
                  f"pred_fg_frac={fg.mean():.3f} pred_insts={n_inst} -> {out}", flush=True)
    print("DONE dump ->", OUT, flush=True)


if __name__ == "__main__":
    main()
