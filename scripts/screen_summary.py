"""Summarize a two-method fast-screen: per-dataset mean delta + paired Wilcoxon on aligned per-image
scores (per_image is seed-major and identical seeds→draws across methods, so element-wise paired).
Usage: python scripts/screen_summary.py <ours_score_dir> <baseline_score_dir>"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon


def _load(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        j = json.load(open(f))
        out[j["dataset"]] = j
    return out


def main(mdir, bdir):
    A, B = _load(mdir), _load(bdir)
    both = sorted(set(A) & set(B))
    print(f"{'dataset':13} {'metric':10} {'base':>7} {'ours':>7} {'delta':>8} {'p':>8}  verdict")
    reg = gain = 0
    for ds in both:
        a, b = A[ds], B[ds]
        pa, pb = np.array(a["per_image"], float), np.array(b["per_image"], float)
        ma, mb = float(pa.mean()), float(pb.mean())
        d = ma - mb
        p = "n/a"
        if len(pa) == len(pb) and len(pa) > 1 and np.any(pa != pb):
            try:
                p = f"{float(wilcoxon(pa, pb).pvalue):.4f}"
            except Exception:
                p = "n/a"
        v = "GAIN" if d > 0.01 else ("REGRESS" if d < -0.005 else "tied")
        reg += v == "REGRESS"; gain += v == "GAIN"
        print(f"{ds:13} {a['metric']:10} {mb:7.3f} {ma:7.3f} {d:+8.3f} {p:>8}  {v}")
    print(f"\nSCREEN: {gain} gain / {reg} regress / {len(both)-gain-reg} tied. "
          f"GO iff >=1 target GAIN and 0 control REGRESS.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
