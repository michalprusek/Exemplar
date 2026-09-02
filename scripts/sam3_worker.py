#!/usr/bin/env python
"""Isolated SAM 3 text-PCS worker (batched).

Reads ``in.npz`` (``text`` concept, ``n``, and ``img_0..img_{n-1}`` query images), loads SAM 3
ONCE, segments every image for the concept, writes ``out.npz`` (``masks_0..masks_{n-1}``, each
``[Ni, H, W]`` bool). Runs under the SAM 3 env ONLY (``~/sam3_env``, transformers 5.13.x) — never
the DINOv3 4.57 env; invoked across the subprocess boundary by ``scripts/sam3_bench.py``.
"""
import sys

import numpy as np


def main(inp, outp):
    from sam3_pcs_shim import segment_pcs, segment_pcs_boxes

    d = np.load(inp, allow_pickle=True)
    n = int(d["n"])
    thr = float(d["threshold"]) if "threshold" in d else 0.3
    mode = str(d["mode"]) if "mode" in d else "text"
    out = {}
    for i in range(n):
        if mode == "exemplar":
            boxes = d[f"boxes_{i}"]  # [K,4] XYXY absolute
            masks = segment_pcs_boxes(d[f"img_{i}"], list(boxes), device="cuda", threshold=thr)
        else:
            masks = segment_pcs(d[f"img_{i}"], str(d["text"]), device="cuda", threshold=thr)
        out[f"masks_{i}"] = np.asarray(masks, bool)
    np.savez(outp, **out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
