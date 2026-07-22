"""N1 — Support-derived FiLM: proof-of-concept (isolated; does NOT touch the main pipeline).

Question: does conditioning a SHARED segmentation head on the K-shot SUPPORT embedding via FiLM (feature-wise
γ,β) beat a per-task head at K≈8, and — crucially — does the CONDITIONING generalise to an UNSEEN dataset?
That is the genuine architectural novelty (self-configuring → self-CONDITIONING) — FiLM is only meaningful if
META-trained across support sets (per-task it degenerates to a learned affine), so we meta-train episodically
over the panel and evaluate leave-one-dataset-out.

Design (kept simple + honest — frozen DINOv3 features, a 1×1 head):
  features:  z = DINOv3 grid feature [D,g,g] (CachedEncoder, resolution RES).
  support embedding e(S): mean over the K support of [global-avg-pooled z (D)] ⊕ [mean fg-fraction] (D+1).
  FiLMGen:   MLP(e) → (γ,β) ∈ R^D each; modulated features z' = (1+γ)⊙z + β (residual FiLM, γ,β start ~0).
  head:      shared Conv1×1(D→1) on z' → per-pixel logit, bilinear to mask res.
  META-TRAIN: episodes sampled from TRAIN datasets; each episode = (support K, query Q); compute e(S), FiLM,
              segment Q, Dice+BCE loss on Q; Adam over {FiLMGen, head} (both SHARED, no per-task weights).
  BASELINES per held-out dataset: (a) per-task 1×1 head trained on its K support (the current few-shot recipe,
              no meta-learning, no FiLM); (b) meta head WITHOUT FiLM (shared head only). FiLM must beat both.
  EVAL: leave-one-dataset-out — meta-train on the other datasets, K-shot condition + segment the held-out.

Run (independent-agent-VERIFIED first, per CLAUDE.md): on tulen, GPU, PANEL_DL_ROOT set. `--datasets` picks a
small subset for the smoke; `--holdout` the LODO test set; `--episodes` meta-training steps.
"""
from __future__ import annotations

import argparse
import numpy as np


def _load_feats(names, res, cache, dev):
    """→ {name: [(feat[g,g,D] float32, mask[H,W] bool)]} for support-pool + a held-out query pool."""
    from active_segmenter.config import EncoderConfig, RunConfig
    from active_segmenter.encoder.cached import CachedEncoder
    from active_segmenter.eval.registry import PANEL, load_dataset

    cfg = RunConfig(device="auto", cache_dir=cache, encoder=EncoderConfig(resolution=res))
    enc = CachedEncoder(cfg, dev, cache)
    out = {}
    for n in names:
        pool, test = load_dataset(PANEL[n], 20, 24, seed=0)
        items = [(np.asarray(enc.extract(im), np.float32), np.asarray(m) > 0) for im, m in pool + test]
        out[n] = items
    return out


def _embed(feats_support, masks_support):
    """support embedding e(S): mean over support of [global-avg-pooled feature ⊕ fg-fraction]."""
    parts = []
    for z, m in zip(feats_support, masks_support):
        gap = z.reshape(-1, z.shape[-1]).mean(0)                      # [D] global-avg-pool
        parts.append(np.concatenate([gap, [float(m.mean())]]))       # ⊕ fg-fraction
    return np.mean(parts, 0).astype(np.float32)                      # [D+1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="spheroid,dsb2018,rozpad,kvasir,hrf,drive")
    ap.add_argument("--holdout", default="hrf")
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--cache", default="/disk1/prusek/asg_cache_film")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = args.datasets.split(",")
    assert args.holdout in names
    data = _load_feats(names, args.res, args.cache, dev)
    D = data[names[0]][0][0].shape[-1]
    train_ds = [n for n in names if n != args.holdout]
    rng = np.random.default_rng(0)

    def batch(items, idxs):
        """stack a set of (feat,mask) into GPU tensors: X[b,D,g,g], Y[b,1,g,g] (mask at feature res)."""
        X = torch.from_numpy(np.stack([items[i][0].transpose(2, 0, 1) for i in idxs])).to(dev)
        ys = []
        for i in idxs:
            g = items[i][0].shape[0]
            m = torch.from_numpy(items[i][1].astype(np.float32))[None, None]
            ys.append((F.interpolate(m, size=(g, g), mode="bilinear", align_corners=False) > 0.5).float())
        return X, torch.cat(ys).to(dev)

    def seg_loss(logit, Y):
        p = logit.sigmoid()
        dice = 1 - (2 * (p * Y).sum() + 1) / (p.sum() + Y.sum() + 1)
        return dice + F.binary_cross_entropy_with_logits(logit, Y)

    class Net(nn.Module):
        def __init__(self, film):
            super().__init__()
            self.film = film
            self.head = nn.Conv2d(D, 1, 1)
            if film:
                self.gen = nn.Sequential(nn.Linear(D + 1, 128), nn.GELU(), nn.Linear(128, 2 * D))
                nn.init.zeros_(self.gen[-1].weight); nn.init.zeros_(self.gen[-1].bias)  # start = identity

        def forward(self, X, e=None):
            z = X
            if self.film and e is not None:
                gb = self.gen(e)
                g, b = gb[:D], gb[D:]
                z = (1 + g[None, :, None, None]) * z + b[None, :, None, None]           # residual FiLM
            return self.head(z)

    def train_meta(film, seed=0):
        torch.manual_seed(seed)                       # SAME init for FiLM/no-FiLM (head created first → same),
        lrng = np.random.default_rng(seed)            # and SAME episode stream → the gap is FiLM alone
        net = Net(film).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        net.train()
        for _ in range(args.episodes):
            ds = train_ds[lrng.integers(len(train_ds))]
            items = data[ds]
            idx = lrng.permutation(len(items))
            sup, qry = idx[:args.k], idx[args.k:args.k + 8]
            e = None
            if film:
                e = torch.from_numpy(_embed([items[i][0] for i in sup], [items[i][1] for i in sup])).to(dev)
            opt.zero_grad()
            loss = 0.0
            for qi in qry:                                             # per-query forward (varying grid size)
                Xq, Yq = batch(items, [qi])
                loss = loss + seg_loss(net(Xq, e), Yq)
            (loss / len(qry)).backward()
            opt.step()
        return net

    def per_task_baseline(items, sup, qry):
        """current recipe: train a fresh 1×1 head on the K support, eval on query (no meta, no FiLM)."""
        head = nn.Conv2d(D, 1, 1).to(dev)
        opt = torch.optim.Adam(head.parameters(), lr=1e-2)
        head.train()
        for _ in range(60):
            opt.zero_grad(); loss = 0.0
            for si in sup:
                Xs, Ys = batch(items, [si]); loss = loss + seg_loss(head(Xs), Ys)
            (loss / len(sup)).backward(); opt.step()
        head.eval(); return lambda Xq: head(Xq)

    def iou_full(logit, mask_bool):
        """IoU at MASK resolution (upsample the coarse logit) — else thin structures vanish on the ~48² grid."""
        H, W = mask_bool.shape
        up = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)[0, 0].sigmoid()
        p = (up > 0.5).cpu().numpy()
        u = int((p | mask_bool).sum())
        return 1.0 if u == 0 else float((p & mask_bool).sum() / u)

    # meta nets trained ONCE with matched init + episode stream (film is the ONLY difference); independent
    # of the holdout support draw. per-task baseline is refit per draw (the current few-shot recipe).
    net_film = train_meta(film=True, seed=0).eval()
    net_meta = train_meta(film=False, seed=0).eval()
    items = data[args.holdout]
    accs = {"per_task": [], "meta_noFiLM": [], "meta_FiLM": []}
    ndraw = 5
    for draw in range(ndraw):                             # multi-draw K-shot (average out support-draw noise)
        idx = np.random.default_rng(100 + draw).permutation(len(items))
        sup, qry = idx[:args.k], idx[args.k:args.k + 24]
        e_t = torch.from_numpy(_embed([items[i][0] for i in sup], [items[i][1] for i in sup])).to(dev)
        base = per_task_baseline(items, sup, qry)         # refit a fresh head on THIS draw's support
        with torch.no_grad():
            for qi in qry:
                Xq = torch.from_numpy(items[qi][0].transpose(2, 0, 1)[None]).to(dev)
                mk = items[qi][1]
                accs["per_task"].append(iou_full(base(Xq), mk))
                accs["meta_noFiLM"].append(iou_full(net_meta(Xq), mk))
                accs["meta_FiLM"].append(iou_full(net_film(Xq, e_t), mk))
    for k, v in accs.items():
        print(f"  {args.holdout:11s} {k:12s} fg-IoU {np.mean(v):.3f} (n={len(v)}, {ndraw} draws)")
    print("FILM_POC_DONE")


if __name__ == "__main__":
    main()
