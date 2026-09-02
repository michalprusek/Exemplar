"""Does `_detect_n_classes` build a MULTI-CLASS head at K=1 on any reported dataset?

WHY THIS EXISTS. The guard in `_detect_n_classes` rejects instance ids as classes by requiring the
nonzero id SET to be IDENTICAL across all support images. At K=1 there is only one support image, so
that test is vacuously true: a single instance-labelled mask whose ids all fall at or below `cap=8`
is read as "max_id semantic classes". The head's final 1x1 then becomes 99 -> n+1 rather than
99 -> 1, which changes the reported 358,168-parameter count and, worse, changes the loss -- the same
class of bug that once produced a fake -0.048 regression on ctc_u373.

The paper reports K=1 numbers (0.711 in the abstract), so this has to be checked before submission.
It cannot be answered from the score records: they do not store the resolved class count.

THE DRAW MUST MATCH THE HARNESS EXACTLY. sota_final.py:319 uses
``np.random.default_rng(seed).choice(len(support_pool), K, replace=False)`` over a pool built by
``load_dataset(spec, pool, test, seed=0)``. Any other RNG call inspects different images and proves
nothing about the reported runs.

RUN ON THE MACHINE THAT HOLDS THE DATA (tulen):

    cd /disk1/prusek/active-segmenter
    ASG_DATA_ROOT=/disk1/prusek PYTHONPATH=$PWD python scripts/check_detect_n_classes_k1.py

Exit code 1 and a loud report if any reported draw resolves to more than one class.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from active_segmenter.eval.registry import PANEL, load_dataset
from active_segmenter.segment.head_fusion_backend import _detect_n_classes

# Binary-GT datasets have max id 1 and are structurally immune; only instance-labelled ones can trip
# this. Pool sizes mirror the reported runs (CLAUDE.md: 20, except 15/16 where the annotation caps it).
CANDIDATES = ["fisbe", "bbbc010", "ctc_u373", "dsb2018", "monuseg", "bacteria"]
POOL = {"ctc_u373": 15, "isbi2012em": 16, "fisbe": 16}
SEEDS = range(10)
KS = (1, 4)          # K=4 too: four masks could in principle share one id set


class _Shim:
    """Only `.label_map` is read by `_detect_n_classes`, so the encoder is not needed."""

    def __init__(self, label_map):
        self.label_map = label_map


bad = []
for ds in CANDIDATES:
    if ds not in PANEL:
        print(f"{ds:<12} NOT IN PANEL -- skipped")
        continue
    pool, _ = load_dataset(PANEL[ds], POOL.get(ds, 20), 10000, seed=0)
    for K in KS:
        hits = []
        for seed in SEEDS:
            sub = np.random.default_rng(seed).choice(len(pool), K, replace=False)
            support = [_Shim(np.asarray(pool[int(i)][1])) for i in sub]
            n = _detect_n_classes(support)
            if n > 1:
                ids = sorted(int(v) for v in np.unique(support[0].label_map) if v > 0)
                hits.append((seed, n, ids[:10]))
        print(f"{ds:<12} K={K:<2d} over {len(list(SEEDS))} seeds -> "
              + (f"MULTI-CLASS on {len(hits)}: {hits}" if hits else "binary"))
        if hits:
            bad.append((ds, K, hits))

print()
if bad:
    print("FAIL: _detect_n_classes returns >1 on " + ", ".join(f"{d} K={k}" for d, k, _ in bad))
    print("  Both the 358,168-parameter count and those cells' scores are affected.")
    print("  Fix: require len(support) > 1 before the set-identity branch. Then re-run those cells.")
    sys.exit(1)
print("PASS: every reported draw resolves to a binary head.")
print("  358,168, the K=1 panel mean and the abstract's 0.711 all stand.")
