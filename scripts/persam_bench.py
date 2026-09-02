#!/usr/bin/env python
"""Standalone PerSAM / PerSAM-F few-shot baseline on the AutoSeg panel.

PerSAM (Zhang et al., "Personalize Segment Anything Model with One Shot", ICLR'24)
is a *training-free* personalization of SAM: from ONE reference (image, mask) it builds
a target prototype (masked-average feature), turns the per-pixel cosine-similarity map
into a positive/negative point prior + target-guided attention, and runs SAM's decoder
with cascaded box refinement. PerSAM-F adds a 10-second fit of TWO scalar weights that
blend SAM's 3 multi-mask outputs, resolving the part-vs-whole scale ambiguity.

This script is SELF-CONTAINED: it imports nothing from the AutoSeg testbed except the
read-only protocol helpers (registry + scoring, which only need numpy/skimage/PIL). It
reuses the upstream `per_segment_anything` fork (the target-guided-attention SAM) from a
clone of https://github.com/ZrrSkywalker/Personalize-SAM.

--- HONEST K=8 ADAPTATION (documented per the task) ------------------------------------
PerSAM is 1-shot by construction. Our protocol hands 8 labelled support shots. We adapt
by building a MULTI-SHOT PROTOTYPE: we masked-average-pool SAM features over the
foreground of ALL 8 support images and average them into a single target prototype
(PerSAM uses mean; PerSAM-F uses max/2+mean/2, per the paper). This uses every support
label, adds ZERO test-time cost (the prototype is one vector regardless of K), and is the
natural multi-shot generalization of PerSAM's single-vector prototype. PerSAM-F's 2 mask
weights are fit jointly over all 8 support shots (each shot contributes its dice+focal
loss). We did NOT ensemble 8 full inference passes per test image (8x slower, and the
prototype-average is the cleaner, standard multi-shot form).

--- SINGLE-OBJECT CAVEAT (honest) ------------------------------------------------------
PerSAM segments ONE object per image (the single best-similarity location). Our blob
datasets (spheroid/rozpad) and especially dsb2018 (dense nuclei) have many objects per
image. PerSAM therefore UNDER-segments multi-object foreground by design; on dsb2018 the
single predicted region is split into connected components for the instance-AP metric,
which is expected to score low. This is a faithful property of the method, not a bug in
the integration.

--- ANCHOR REPRODUCED 2026-07-19 (this claim is now measured, not asserted) -------------
Upstream `persam.py` was run on PerSeg (40 objects) in this environment, with this SAM
checkpoint and this `per_segment_anything` fork, and scored by upstream `eval_miou.py`:

    mIoU 89.32   vs   89.3 published (Zhang et al., ICLR'24, Table 1)

An earlier version of this docstring claimed such a validation without one existing; it was
corrected to say so, and the run has now actually been done. What it establishes is the
environment, the checkpoint and the fork -- the wrapper below reuses all three and its
hyper-parameters were separately verified line-by-line against upstream `persam_f.py`.

Two independent checks agree that the LOW panel numbers are the method's ceiling rather
than a broken integration:
  * spheroidj (1.0 objects/image) scores 0.837, dsb2018 (55.1 objects/image) scores 0.050;
    a broken integration would fail on both.
  * on dsb2018 a perfect segmentation of ONE median instance against the whole foreground
    is IoU 0.0406. PerSAM measures 0.0498-0.0564, i.e. slightly ABOVE that ceiling because
    SAM sometimes captures a small cluster rather than a single nucleus.
Report the low numbers WITH this explanation; quoting 0.05 alone reads as a baseline we
broke, when it is in fact the method operating at the limit of what it is built to do.

CAVEAT still open: our multi-shot prototype (K=8) scores BELOW K=1 on spheroidj (0.764 vs
0.837), and PerSAM-F scores below PerSAM there (0.783 vs 0.837) although published PerSeg
has PerSAM-F far ahead (95.3 vs 89.3). Both are consistent with over-fitting two scalars on
few shots, but that is a hypothesis; the PerSAM-F anchor has not been run.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


# ------------------------------------------------------------------ image/mask helpers
def to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """Any array (gray/rgb/rgba, uint8/uint16/float) -> HxWx3 uint8 RGB in [0,255]."""
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    if a.ndim == 3 and a.shape[2] == 3:
        rgb = a
    elif a.ndim == 2:
        rgb = np.repeat(a[..., None], 3, axis=2)
    else:  # single-channel HxWx1 or odd -> take first channel
        rgb = np.repeat(np.asarray(a).reshape(a.shape[0], a.shape[1], -1)[..., :1], 3, axis=2)
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32)
        lo, hi = float(rgb.min()), float(rgb.max())
        rgb = np.zeros_like(rgb) if hi <= lo else (rgb - lo) * (255.0 / (hi - lo))
        rgb = rgb.astype(np.uint8)
    return np.ascontiguousarray(rgb)


def mask_to_rgb_uint8(binmask: np.ndarray) -> np.ndarray:
    m = (np.asarray(binmask) > 0).astype(np.uint8) * 255
    return np.repeat(m[..., None], 3, axis=2)


# ------------------------------------------------------------------ PerSAM core (fork API)
class MaskWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)


def _point_selection(mask_sim: torch.Tensor, topk: int = 1):
    """Verbatim from upstream persam.py: returns pos (x,y) + neg (x,y) prompts."""
    w, h = mask_sim.shape
    top = mask_sim.flatten(0).topk(topk)[1]
    top_x = (top // h).unsqueeze(0)
    top_y = (top - top_x * h)
    top_xy = torch.cat((top_y, top_x), dim=0).permute(1, 0).cpu().numpy()
    top_label = np.array([1] * topk)
    last = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last // h).unsqueeze(0)
    last_y = (last - last_x * h)
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0).cpu().numpy()
    last_label = np.array([0] * topk)
    return top_xy, top_label, last_xy, last_label


def _restore(predictor, shot):
    predictor.features = shot["features"]
    predictor.input_size = shot["input_size"]
    predictor.original_size = shot["original_size"]
    predictor.is_image_set = True


def _sim_map(predictor, target_feat: torch.Tensor) -> torch.Tensor:
    """Cosine-sim of the currently-set image's features vs the prototype -> [H,W] map."""
    test_feat = predictor.features.squeeze()          # [C,64,64]
    C, h, w = test_feat.shape
    tf = test_feat / test_feat.norm(dim=0, keepdim=True)
    tf = tf.reshape(C, h * w)
    sim = target_feat @ tf                            # [1, h*w]
    sim = sim.reshape(1, 1, h, w)
    sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
    sim = predictor.model.postprocess_masks(
        sim, input_size=predictor.input_size, original_size=predictor.original_size).squeeze()
    return sim


def dice_loss(inputs, targets):
    inputs = inputs.sigmoid().flatten(1)
    num = 2 * (inputs * targets).sum(-1)
    den = inputs.sum(-1) + targets.sum(-1)
    return (1 - (num + 1) / (den + 1)).sum()


def focal_loss(inputs, targets, alpha=0.25, gamma=2):
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    return loss.mean(1).sum()


def encode_support(predictor, support, device):
    """Encode every support shot once; cache features + fg-pooled feats + gt for reuse."""
    shots = []
    for img, lbl in support:
        binm = np.asarray(lbl) > 0
        if not binm.any():
            continue
        rgb = to_rgb_uint8(img)
        ref_mask_t = predictor.set_image(rgb, mask_to_rgb_uint8(binm))     # [1,3,1024,1024]
        feat = predictor.features                                          # [1,256,64,64]
        ref_feat = feat.squeeze().permute(1, 2, 0)                         # [64,64,256]
        rm = F.interpolate(ref_mask_t, size=ref_feat.shape[:2], mode="bilinear").squeeze()[0]
        fg = ref_feat[rm > 0]                                              # [Nfg,256]
        if fg.numel() == 0:
            continue
        shots.append(dict(
            features=feat.clone(), input_size=predictor.input_size,
            original_size=predictor.original_size, fg=fg.clone(),
            gt=torch.as_tensor(binm, device=device).float().reshape(1, -1)))
    return shots


def build_prototypes(shots):
    """Multi-shot prototypes: PerSAM = mean; PerSAM-F = max/2+mean/2 (both over all shots)."""
    allfg = torch.cat([s["fg"] for s in shots], dim=0)     # [N,256]
    mean = allfg.mean(0)
    mx = allfg.max(0)[0]
    emb = mean.unsqueeze(0)                                 # [1,256]
    persam = dict(
        target_feat=(emb / emb.norm(dim=-1, keepdim=True)),        # [1,256]
        target_embedding=emb.unsqueeze(0))                         # [1,1,256]
    pf = (mx / 2 + mean / 2).unsqueeze(0)
    persam_f_feat = pf / pf.norm(dim=-1, keepdim=True)             # [1,256]
    return persam, persam_f_feat


def train_mask_weights(predictor, shots, persam_f_feat, device, epochs, lr):
    """Fit the 2 blend weights over ALL support shots (decoder outputs are fixed -> cache)."""
    cached = []
    for s in shots:
        _restore(predictor, s)
        sim = _sim_map(predictor, persam_f_feat)
        top_xy, top_label, _, _ = _point_selection(sim, topk=1)
        _, _, _, logits_high = predictor.predict(
            point_coords=top_xy, point_labels=top_label, multimask_output=True)
        cached.append((logits_high.flatten(1).detach(), s["gt"]))          # ([3,HW],[1,HW])
    mw = MaskWeights().to(device)
    opt = torch.optim.AdamW(mw.parameters(), lr=lr, eps=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        total = 0.0
        for lh, gt in cached:
            w = torch.cat((1 - mw.weights.sum(0).unsqueeze(0), mw.weights), dim=0)  # [3,1]
            logit = (lh * w).sum(0).unsqueeze(0)                            # [1,HW]
            total = total + dice_loss(logit, gt) + focal_loss(logit, gt)
        opt.zero_grad(); (total / len(cached)).backward(); opt.step(); sched.step()
    mw.eval()
    with torch.no_grad():
        w = torch.cat((1 - mw.weights.sum(0).unsqueeze(0), mw.weights), dim=0)
    return w.detach()


# ------------------------------------------------------------------ per-image inference
def _bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()])


def predict_persam(predictor, rgb, persam):
    predictor.set_image(rgb)
    sim = _sim_map(predictor, persam["target_feat"])
    top_xy, top_label, last_xy, last_label = _point_selection(sim, topk=1)
    xy = np.concatenate([top_xy, last_xy], axis=0)
    lab = np.concatenate([top_label, last_label], axis=0)
    s = (sim - sim.mean()) / torch.std(sim)
    s = F.interpolate(s.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
    attn_sim = s.sigmoid_().unsqueeze(0).flatten(3)
    masks, scores, logits, _ = predictor.predict(
        point_coords=xy, point_labels=lab, multimask_output=False,
        attn_sim=attn_sim, target_embedding=persam["target_embedding"])
    best = 0
    if not masks[best].any():
        return masks[best]
    masks, scores, logits, _ = predictor.predict(
        point_coords=xy, point_labels=lab,
        mask_input=logits[best:best + 1, :, :], multimask_output=True)
    best = int(np.argmax(scores))
    if not masks[best].any():
        return masks[best]
    masks, scores, logits, _ = predictor.predict(
        point_coords=xy, point_labels=lab, box=_bbox(masks[best])[None, :],
        mask_input=logits[best:best + 1, :, :], multimask_output=True)
    return masks[int(np.argmax(scores))]


def predict_persam_f(predictor, rgb, persam_f_feat, weights):
    predictor.set_image(rgb)
    sim = _sim_map(predictor, persam_f_feat)
    top_xy, top_label, _, _ = _point_selection(sim, topk=1)
    masks, scores, logits, logits_high = predictor.predict(
        point_coords=top_xy, point_labels=top_label, multimask_output=True)
    logit_high = (logits_high * weights.unsqueeze(-1)).sum(0)
    mask = (logit_high > 0).detach().cpu().numpy()
    w_np = weights.cpu().numpy()
    logit = (logits * w_np[..., None]).sum(0)
    if not mask.any():
        return mask
    masks, scores, logits, _ = predictor.predict(
        point_coords=top_xy, point_labels=top_label, box=_bbox(mask)[None, :],
        mask_input=logit[None, :, :], multimask_output=True)
    best = int(np.argmax(scores))
    if not masks[best].any():
        return masks[best]
    masks, scores, logits, _ = predictor.predict(
        point_coords=top_xy, point_labels=top_label, box=_bbox(masks[best])[None, :],
        mask_input=logits[best:best + 1, :, :], multimask_output=True)
    return masks[int(np.argmax(scores))]


# ------------------------------------------------------------------ scoring glue
def score_one(score_prediction, key, metric, fg_mask, label_map, measure_label):
    """metric-appropriate score; dsb2018 -> connected components as instances."""
    instances = None
    if metric == "instance_ap":
        lab = measure_label(fg_mask)
        instances = [lab == i for i in range(1, int(lab.max()) + 1)]
    res = score_prediction(metric, np.asarray(fg_mask, bool), label_map, instances)
    return float(res[key])


def run_dataset(name, spec, predictor, args, api, device):
    load_dataset, score_prediction, primary_key, measure_label, split_fingerprint, \
        effective_metric = api
    # --fg-scoring routes through the SAME shared expression every other campaign script uses, and
    # it drives BOTH the number computed below and the `metric` field of the record written from
    # `key`. Deriving them separately is how a column ends up scored one way and labelled another,
    # which stats() answers with a silent METRIC MISMATCH skip rather than an error.
    eff = effective_metric(spec.metric, args.fg_scoring)
    key = primary_key(eff)
    per_seed = {"persam": [], "persam_f": []}
    # Per-image scores kept as [seed][image] instead of collapsed to a per-seed mean, so the run can
    # emit a sota_final score record and enter the PAIRED statistics. The mean is what the table
    # prints; the paired Wilcoxon needs the individual images, and they were being thrown away.
    per_seed_images = {"persam": [], "persam_f": []}
    n_test = args.test if args.smoke_test <= 0 else args.smoke_test
    # SAME multi-draw fixed-pool protocol as scripts/sota_final.py: load the support POOL and the test
    # split ONCE at seed 0, then subsample K support masks per seed. Passing `support=args.support,
    # seed=seed` here (the previous behaviour) silently scored a DIFFERENT test set than our method on
    # every download-kind dataset, because the flat-directory loader slices
    # `permutation(seed)[support : support+test]` — so both the seed AND the support size move the test
    # slice. Per-image scores from the two protocols are then not comparable, and a paired test over
    # them is meaningless even though the image COUNT matches.
    support_pool, test = load_dataset(spec, args.pool, args.test, seed=0)
    test = test[:n_test]
    # AFTER the truncation: the fingerprint is a claim about the images actually scored, and a
    # smoke run's short split must not be able to masquerade as the full one.
    split_fp = split_fingerprint(test)
    for seed in args.seeds:
        idx = np.random.default_rng(seed).choice(len(support_pool), args.support, replace=False)
        support = [support_pool[i] for i in idx]
        shots = encode_support(predictor, support, device)
        if not shots:
            raise RuntimeError("all support masks empty after encoding")
        persam, persam_f_feat = build_prototypes(shots)
        weights = None
        if args.mode in ("persam_f", "both"):
            weights = train_mask_weights(predictor, shots, persam_f_feat, device,
                                         args.train_epoch, args.lr)
        s_scores, sf_scores = [], []
        for img, lbl in test:
            rgb = to_rgb_uint8(img)
            if args.mode in ("persam", "both"):
                try:
                    m = predict_persam(predictor, rgb, persam)
                except Exception as e:   # an empty mask here scores 0.0 and DEFLATES this baseline
                    raise RuntimeError(f"persam failed on a test image; refusing to score it as an "
                                       f"empty mask (that would silently understate this baseline)") from e
                s_scores.append(score_one(score_prediction, key, eff, m, lbl, measure_label))
            if args.mode in ("persam_f", "both"):
                try:
                    m = predict_persam_f(predictor, rgb, persam_f_feat, weights)
                except Exception as e:   # an empty mask here scores 0.0 and DEFLATES this baseline
                    raise RuntimeError(f"persam_f failed on a test image; refusing to score it as an "
                                       f"empty mask (that would silently understate this baseline)") from e
                sf_scores.append(score_one(score_prediction, key, eff, m, lbl, measure_label))
        if s_scores:
            per_seed["persam"].append(float(np.mean(s_scores)))
            per_seed_images["persam"].append(list(s_scores))
        if sf_scores:
            per_seed["persam_f"].append(float(np.mean(sf_scores)))
            per_seed_images["persam_f"].append(list(sf_scores))
        msg = [f"seed{seed}"]
        if s_scores:
            msg.append(f"persam={np.mean(s_scores):.4f}")
        if sf_scores:
            msg.append(f"persam_f={np.mean(sf_scores):.4f}")
        print("    " + " ".join(msg), flush=True)
    return key, per_seed, per_seed_images, split_fp


def write_records(args, name, key, per_seed_images, split_fp, write_score_record):
    """Emit one sota_final score record per PerSAM variant that actually produced scores.

    The seeds here are the SAME draws sota_final makes (``default_rng(seed).choice`` over the same
    fixed pool), and the test split is the same fixed one, so these records pair directly against
    our method's — which is the whole point of writing them.
    """
    for method, rows in per_seed_images.items():
        if not rows:                    # variant not selected by --mode
            continue
        path = write_score_record(
            args.score_dir, method=method, dataset=name, metric=key, per_seed_images=rows,
            seeds=args.seeds, split_fp=split_fp,
            # fg_scoring is recorded even though `metric` usually implies it: on a clDice dataset
            # the convention keeps clDice either way, so the record would otherwise carry no trace
            # of which scoring convention produced it.
            protocol=dict(pool=args.pool, test=args.test, support=args.support, split_seed=0,
                          sam_type=args.sam_type, fg_scoring=bool(args.fg_scoring)))
        print(f"    wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persam-repo", default="/disk2/prusek/Personalize-SAM")
    ap.add_argument("--autoseg-repo", default="/disk1/prusek/active-segmenter")
    ap.add_argument("--ckpt", default="/disk2/prusek/sam_vit_h_4b8939.pth")
    ap.add_argument("--sam-type", default="vit_h")
    ap.add_argument("--datasets", default="spheroid,spheroidj,dsb2018,rozpad,kvasir,hrf")
    ap.add_argument("--seeds", default="0,1,2,3,4,5")
    ap.add_argument("--support", type=int, default=8)
    ap.add_argument("--pool", type=int, default=20,
                    help="support POOL size; must match sota_final's --pool so the test split (which the "
                         "flat-dir loader slices AFTER the pool) is identical to our method's")
    ap.add_argument("--test", type=int, default=24)
    ap.add_argument("--mode", default="both", choices=["persam", "persam_f", "both"])
    ap.add_argument("--train-epoch", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--smoke-test", type=int, default=0,
                    help=">0 caps #test images per seed (smoke run)")
    ap.add_argument("--fg-scoring", action="store_true",
                    help="score the FOREGROUND (fg_iou) on every non-clDice dataset instead of its "
                         "native metric, the campaign-wide convention that lets one table row name "
                         "one metric. Same semantics and flag name as specialist_finetune_bench.py; "
                         "the record's `metric` field follows, so the record still pairs.")
    ap.add_argument("--score_dir", default="",
                    help="write per-image score records here (results/scores) so PerSAM/PerSAM-F "
                         "enter `sota_final.py stats`'s PAIRED significance instead of being "
                         "read off this script's stdout table. Empty = table only.")
    args = ap.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(",") if s != ""]

    sys.path.insert(0, args.persam_repo)      # per_segment_anything (target-guided-attn SAM)
    sys.path.insert(0, args.autoseg_repo)     # registry + scoring (numpy/skimage only)
    from per_segment_anything import sam_model_registry, SamPredictor
    from active_segmenter.eval.registry import PANEL, load_dataset
    from active_segmenter.eval.score_record import split_fingerprint, write_score_record
    from active_segmenter.eval.scoring import effective_metric, score_prediction, primary_key
    from skimage.measure import label as measure_label

    if args.score_dir and args.smoke_test > 0:
        # A smoke run scores a TRUNCATED split. Its record would carry a different split_fp and a
        # smaller test_per_seed, so stats() would skip it — but only after it had been sitting in
        # the score dir looking like a real result, and only if nobody had meanwhile deleted the
        # real one. Refuse at the single gate the writes are behind rather than write-then-hope.
        print("[persam_bench] REFUSING to write score records: --smoke-test caps the test split, "
              "and a truncated split must never land beside full-split records. Re-run without "
              "--smoke-test to produce pairable records; this run still prints its table.",
              flush=True)
        args.score_dir = ""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[persam_bench] device={device} ckpt={args.ckpt} mode={args.mode} "
          f"seeds={args.seeds} smoke_test={args.smoke_test} fg_scoring={args.fg_scoring}",
          flush=True)
    sam = sam_model_registry[args.sam_type](checkpoint=args.ckpt).to(device)
    sam.eval()
    predictor = SamPredictor(sam)
    api = (load_dataset, score_prediction, primary_key, measure_label, split_fingerprint,
           effective_metric)

    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    results = {}
    for name in names:
        if name not in PANEL:
            print(f"\n== {name} ==  SKIP: not in PANEL", flush=True)
            continue
        spec = PANEL[name]
        print(f"\n== {name} ({spec.metric}) ==", flush=True)
        try:
            key, per_seed, per_seed_images, split_fp = run_dataset(name, spec, predictor, args,
                                                                   api, device)
        except Exception as e:
            print(f"  SKIP {name}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            continue
        results[name] = (key, per_seed)
        # Deliberately OUTSIDE the SKIP handler: a scoring failure is a dataset we could not
        # measure, but a record that violates the contract is a bug in this script, and swallowing
        # it into a SKIP line would hide it behind a legitimate-looking skip.
        if args.score_dir:
            write_records(args, name, key, per_seed_images, split_fp, write_score_record)

    print("\n================ PERSAM PANEL RESULTS ================", flush=True)
    print(f"{'dataset':<11}{'metric':<14}{'PerSAM':<20}{'PerSAM-F':<20}", flush=True)
    for name in names:
        if name not in results:
            print(f"{name:<11}{'-':<14}{'SKIP':<20}{'SKIP':<20}", flush=True)
            continue
        key, per_seed = results[name]

        def fmt(vals):
            if not vals:
                return "-"
            return f"{np.mean(vals):.4f}±{np.std(vals):.4f}"
        print(f"{name:<11}{key:<14}{fmt(per_seed['persam']):<20}{fmt(per_seed['persam_f']):<20}",
              flush=True)
    print("PERSAM_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
