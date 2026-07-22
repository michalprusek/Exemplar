"""C-RADIOv4 mode probe (2026-07-12): which teacher mode best serves our in-context correspondence?

Loads C-RADIOv4-H (adaptors dino_v3_7b / sam3 / siglip2-g + backbone), extracts each mode's spatial
feature grid, and scores our frozen correspondence (support→query cosine matching → fg-IoU) PER MODE
across datasets. Tells us: does sam3-mode beat dino on boundaries, siglip on OOD, vs our DINOv3
baseline — i.e. is C-RADIO a free upgrade and are the modes complementary (→ adaptive blend worth it).
Runs in the DINOv3 env (C-RADIO loads there). GPU script.
"""
from __future__ import annotations

import argparse

import numpy as np

_MODES = ["backbone", "dino_v3_7b", "sam3", "siglip2-g"]


def load_cradio(dev):
    import torch

    return torch.hub.load("NVlabs/RADIO", "radio_model", version="c-radio_v4-h",
                          adaptor_names=["dino_v3_7b", "sam3", "siglip2-g"],
                          skip_validation=True).eval().to(dev)


def extract_modes(model, image, dev, res, patch=16):
    import torch

    a = np.asarray(image).astype(np.float32)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    a = a[..., :3]
    a = (a - a.min()) / (np.ptp(a) + 1e-6)
    t = torch.from_numpy(a).permute(2, 0, 1)[None].to(dev)
    h, w = model.get_nearest_supported_resolution(res, res)
    t = torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    with torch.no_grad():
        out = model(t)
    gh, gw = h // patch, w // patch
    feats = {}
    for k in _MODES:
        _, f = out[k]                                   # [1, L, D]
        d = f.shape[-1]
        grid = f[0].reshape(gh, gw, d).float().cpu().numpy()
        grid = grid / np.maximum(np.linalg.norm(grid, axis=2, keepdims=True), 1e-6)
        feats[k] = grid.astype(np.float32)
    return feats


def score_mode(sup_grids, sup_labels, test_grids, test_labels, dev):
    from active_segmenter.config import MatchConfig
    from active_segmenter.eval import metrics
    from active_segmenter.membank.bank import MemoryBank
    from active_segmenter.propose import correspondence as corr
    from active_segmenter.segment.base import foreground_from_score

    mc = MatchConfig()
    bank = MemoryBank()
    for grid, lab in zip(sup_grids, sup_labels):
        fg = (np.asarray(lab) > 0).astype(int)
        bank.add_from_annotation(grid, fg, {1: 1} if fg.any() else {}, 0)
    ious = []
    for grid, lab in zip(test_grids, test_labels):
        s = corr.score_map(grid, bank, 1, mc, device=dev)
        fgm = foreground_from_score(s, np.asarray(lab).shape[:2])
        ious.append(metrics.foreground_iou(fgm, lab))
    return float(np.mean(ious))


def main():
    from active_segmenter.config import RunConfig
    from active_segmenter.eval.registry import PANEL, load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="rozpad,dsb2018,hrf,kvasir")
    ap.add_argument("--support", type=int, default=8)
    ap.add_argument("--test", type=int, default=8)
    ap.add_argument("--res", type=int, default=672)
    args = ap.parse_args()

    dev = RunConfig(device="auto").device_resolved()
    model = load_cradio(dev)
    print(f"C-RADIOv4-H loaded on {dev}; correspondence fg-IoU per MODE (res~{args.res})", flush=True)
    print(f"{'dataset':>10} " + " ".join(f"{m:>12}" for m in _MODES), flush=True)
    for ds in args.datasets.split(","):
        try:
            spec = PANEL[ds]
            support, test = load_dataset(spec, args.support, args.test, seed=0)
            sup_feats = [extract_modes(model, im, dev, args.res) for im, _ in support]
            tst_feats = [extract_modes(model, im, dev, args.res) for im, _ in test]
            sup_lab = [np.asarray(l) for _, l in support]
            tst_lab = [np.asarray(l) for _, l in test]
            row = {}
            for m in _MODES:
                row[m] = score_mode([f[m] for f in sup_feats], sup_lab,
                                    [f[m] for f in tst_feats], tst_lab, dev)
            print(f"{ds:>10} " + " ".join(f"{row[m]:>12.3f}" for m in _MODES), flush=True)
        except Exception as e:
            print(f"{ds:>10} SKIP ({type(e).__name__}: {str(e)[:60]})", flush=True)
    print("CRADIO_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
