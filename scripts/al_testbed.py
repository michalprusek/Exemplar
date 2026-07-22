#!/usr/bin/env python
"""Reusable active-learning acquisition test-bed.

Answers "does acquisition X actually pick the image that maximises the model's
improvement?" for ANY scorer, benchmarked against the greedy-oracle upper bound and
a random control, with multi-seed error bars. New acquisition functions register in
SCORERS — a scorer maps (pool, bank_idxs, ctx) -> {pool_index: score} (higher first).

Reports, per method:
- learning curve (mean +/- std test fg IoU over seeds),
- mean gap to the greedy oracle (regret) and % of the random->oracle gap closed,
- mean Spearman correlation between the score and the ACTUAL per-candidate test gain
  (the direct "is the ranking informative?" number; >0 good, ~0 == random, <0 harmful).

Run on tulen:
  ~/dinov3_env/bin/python scripts/al_testbed.py --cache /disk1/prusek/asg_cache \
      --data /disk1/prusek/dsb2018 --pool 30 --test 30 --rounds 6 --seeds 3 \
      --methods random,uncertainty,epig
"""
import argparse

import numpy as np
from scipy.stats import spearmanr
from skimage.transform import resize

from active_segmenter.config import ClusterConfig, EncoderConfig, MatchConfig, RunConfig
from active_segmenter.encoder.cached import CachedEncoder
from active_segmenter.eval import metrics
from active_segmenter.eval.datasets import load_dsb2018, load_fewshot, make_heterogeneous
from active_segmenter.membank.bank import MemoryBank
from active_segmenter.propose import correspondence as corr
from active_segmenter.acquire import (
    badge,
    coldstart,
    fg_coverage,
    smec,
    submodular,
    support_loo,
    transductive,
)
from active_segmenter.acquire.uncertainty import ambiguous_fraction
from active_segmenter.segment.base import LabeledExample
from active_segmenter.segment.correspondence_backend import CorrespondenceBackend

MC = MatchConfig(topk=5, bidirectional=False)


def make_backend(name: str, cfg, dev: str, refine: str = "none", enc=None, support_k: int = 0):
    """Construct a SegmenterBackend by name. Backends beyond the baseline are imported
    lazily so the heavy deps (torch head, CRF, SAM3 subprocess) only load when raced.
    ``refine`` (none|point|mask|mask_box|amodal) wraps the backend in a SAM :class:`RefiningBackend`
    so the trained head can be SAM-refined and raced head-to-head against INSID3 on instance-AP."""
    if name == "correspondence":
        be = CorrespondenceBackend(MC, ClusterConfig(), device=dev)
    elif name == "head":
        from active_segmenter.segment.head_backend import TrainableHeadBackend
        # warm_start=False: the testbed reuses one backend across seeds, so train the head
        # from scratch on each labeled set — the honest per-budget evaluation (no leakage).
        be = TrainableHeadBackend(device=dev, warm_start=False)
    elif name == "insid3":
        from active_segmenter.segment.insid3_backend import Insid3FrozenBackend
        be = Insid3FrozenBackend(MC, device=dev)
    elif name == "universeg":                              # ICCV'23 few-shot medical SOTA baseline
        from active_segmenter.segment.universeg_backend import UniverSegBackend
        be = UniverSegBackend(device=dev)
    elif name == "tyche":                                  # CVPR'24 stochastic few-shot in-context baseline
        from active_segmenter.segment.tyche_backend import TycheBackend
        be = TycheBackend(device=dev)
    elif name == "matcher":                                # ICLR'24 training-free one-shot DINOv2+SAM matcher
        from active_segmenter.segment.matcher_backend import MatcherBackend
        be = MatcherBackend(device=dev)
    elif name == "seggpt":                                 # ICCV'23 general in-context painting foundation model
        from active_segmenter.segment.seggpt_backend import SegGptBackend
        # max_support MUST follow K. Its default of 8 silently truncated the support with
        # `support[:8]`, so a K=16 run measured 8 shots while the score record faithfully wrote
        # support=16 — a mislabelled point in the K-scaling figure, and undetectable downstream
        # because nothing else disagreed. SegGPT already beats us on some datasets, so
        # under-feeding it is not a harmless bug.
        be = SegGptBackend(device=dev, max_support=support_k or 8)
    elif name == "sam3":
        from active_segmenter.segment.sam3_backend import Sam3PcsBackend
        be = Sam3PcsBackend(device=dev)
    elif name in ("hyperbank", "hyperbank_fusion"):
        from active_segmenter.segment.hyperbank_backend import HyperBankBackend
        be = HyperBankBackend(device=dev, fusion=(name == "hyperbank_fusion"))
    elif name in ("head_fusion", "head_fusion_tc", "head_fusion_tile", "head_fusion_cc",
                  "head_fusion_gf", "head_fusion_v2", "head_fusion_bnd", "head_fusion_al",
                  "head_fusion_blob", "head_fusion_hi", "head_fusion_thin", "head_fusion_sf",
                  "head_fusion_sf_dys", "head_fusion_sf_gid", "head_fusion_sf_gil",
                  "head_fusion_sf_car", "head_fusion_uni", "head_fusion_uni_adapt",
                  "head_fusion_adaptive", "head_fusion_adaptive_dist", "head_fusion_adaptive_thin",
                  "head_fusion_adaptive_forcenat", "head_fusion_adaptive_color",
                  "head_fusion_adaptive_affinity", "head_fusion_adaptive_trace",
                  "head_fusion_adaptive_tc") or name.startswith("head_fusion_best"):
        #   _color=colour bank; _affinity/_trace = SAM-free decoders; _tc = unfrozen bank; head_fusion_best =
        #   UNIFIED (colour + affinity decoder + tubularity-gated bank-unfreeze), fully SAM-free = folded-in
        #   current best. COMPOSABLE lever suffixes: head_fusion_best[_<lever>]... where lever ∈
        #   {cgate,corr,film,bank}; ANY combination works (e.g. head_fusion_best_cgate_film) — these are the
        #   factorial candidates. cgate=competitive per-pixel group gate; corr=correspondence prior; film=
        #   support-conditioned FiLM hypernetwork; bank=expanded curated classical bank (roadmap #8).
        cgate = corrp = film = bank_extra = nobank = mproto = bdou_lev = prec_lev = nocls = False
        noloss = nocolor = rgbfeat = bankselect = scaleconf = False   # self-config ablation negations + raw-RGB / bank-select / scale-config A/B (default OFF)
        dino_scale = "both"; proj_dim = 32
        if name.startswith("head_fusion_best_"):
            toks = name[len("head_fusion_best_"):].split("_")
            allowed = ("cgate", "corr", "film", "bank", "nobank", "mproto", "bdou", "prec", "nocls",
                       "coarseonly", "fineonly", "noloss", "nocolor", "rgbfeat", "bankselect", "scaleconf")
            bad = [t for t in toks if t not in allowed and not (t.startswith("pd") and t[2:].isdigit())]
            if bad:
                raise ValueError(f"unknown head_fusion_best lever token(s) {bad} in '{name}'")
            cgate = "cgate" in toks; corrp = "corr" in toks
            film = "film" in toks; bank_extra = "bank" in toks
            nocls = "nocls" in toks     # ABLATION: DINO features only — zero the classical prior bank
            noloss = "noloss" in toks   # SELF-CONFIG ABLATION: disable the adaptive-loss constructor (fixed loss weights)
            nocolor = "nocolor" in toks # SELF-CONFIG ABLATION: disable colour/stain channel selection + CLAHE (grayscale)
            rgbfeat = "rgbfeat" in toks # A/B: append raw R,G,B as native features (simplification test vs colour selection)
            bankselect = "bankselect" in toks # LEVER: Fisher-select classical bank channels from support (unify colour+bank selection)
            scaleconf = "scaleconf" in toks   # LEVER: cluster support DT radii → set bank Frangi/Sauvola/LoG scales per centroid
            nobank = "nobank" in toks   # DISABLE tubularity-gated bank-unfreeze (fixes its hours-long latency)
            # ABLATION tokens: coarseonly/fineonly = single DINO scale; pd<N> = DINO channel embedding width
            dino_scale = "coarse" if "coarseonly" in toks else "fine" if "fineonly" in toks else "both"
            proj_dim = next((int(t[2:]) for t in toks if t.startswith("pd") and t[2:].isdigit()), 32)
            mproto = "mproto" in toks   # multi-prototype correspondence (Lever 1, DROPPED C15): k-means + max-pool corr
            corrp = corrp or mproto     # mproto implies the corr channel (in multi-prototype mode)
            bdou_lev = "bdou" in toks   # Boundary DoU loss term (Lever 2, DROPPED C16): density-gated fg-contour sharpening
            prec_lev = "prec" in toks   # precision Tversky (Lever 3): dense-compact FP-penalty, curbs interior over-prediction
            name = "head_fusion_best"   # reuse best's full config; the levers are the only additions
        from active_segmenter.segment.head_fusion_backend import HeadFusionBackend
        v2 = name in ("head_fusion_v2", "head_fusion_bnd")  # W2 boundary head; v2 also W1+clDice
        al = name == "head_fusion_al"  # fast config for the AL learning-curve loop (many refits)
        hi = name == "head_fusion_hi"      # W1: hi-res classical (tile + fine scales + contrast)
        thin = name == "head_fusion_thin"  # thin-structure: tile + fine scales + clDice (HRF vessels)
        # scale-fusion variants: coarse⊕fine DINOv3 concat, with a chosen upsampler (bilinear default)
        sf_up = {"head_fusion_sf": None, "head_fusion_sf_dys": "dysample",
                 "head_fusion_sf_gid": "guided", "head_fusion_sf_gil": "guided_lite",
                 "head_fusion_sf_car": "carafe", "head_fusion_uni": "guided",
                 "head_fusion_uni_adapt": "guided", "head_fusion_adaptive": "guided",
                 "head_fusion_adaptive_dist": "guided", "head_fusion_adaptive_thin": "guided",
                 "head_fusion_adaptive_forcenat": "guided", "head_fusion_adaptive_color": "guided",
                 "head_fusion_adaptive_affinity": "guided", "head_fusion_adaptive_trace": "guided",
                 "head_fusion_adaptive_tc": "guided", "head_fusion_best": "guided"}
        sf = name in sf_up
        # _adaptive = the FULL adaptive-loss constructor: the loss is built per support image from its mask
        # morphology (Dice anchor + gated focal/tversky/clDice/boundary/instance; near-0 terms skipped).
        # _dist = the instance stage is a DT-REGRESSION head (center-distance → seeded watershed, StarDist/
        # micro-SAM-AIS pattern) instead of the boundary-BCE head — smoother + few-shot-friendlier.
        adapt_loss = name in ("head_fusion_adaptive", "head_fusion_adaptive_dist",
                              "head_fusion_adaptive_thin", "head_fusion_adaptive_forcenat",
                              "head_fusion_adaptive_color", "head_fusion_adaptive_affinity",
                              "head_fusion_adaptive_trace", "head_fusion_adaptive_tc", "head_fusion_best") and not noloss
        dist = name == "head_fusion_adaptive_dist"
        # UNIVERSAL recipe: sf_gid + ALL always-on morphology terms (topology + touching-separation loss,
        # CLAHE, finer scales) → one recipe for blobs/contours/filaments; overlap via `--refine amodal`.
        # _adapt = the clDice topology term is PER-IMAGE adaptive (weight ∝ annotated-shape thinness →
        # self-gates to ~0 on blobs, full on filaments), fixing the fixed-superset blob regression.
        uni = name in ("head_fusion_uni", "head_fusion_uni_adapt", "head_fusion_adaptive",
                       "head_fusion_adaptive_dist", "head_fusion_adaptive_thin",
                       "head_fusion_adaptive_forcenat", "head_fusion_adaptive_color",
                       "head_fusion_adaptive_affinity", "head_fusion_adaptive_trace",
                       "head_fusion_adaptive_tc", "head_fusion_best")
        inst = ("trace" if name == "head_fusion_adaptive_trace"            # SAM-free crossing-filament tracer
                else "affinity" if name in ("head_fusion_adaptive_affinity", "head_fusion_best")  # SAM-free decoder
                else "blob" if name in (("head_fusion_blob",) + tuple(sf_up))  # scale-adaptive blob markers
                else "cc" if name == "head_fusion_cc" else "watershed")
        be = HeadFusionBackend(
            device=dev,
            trainable_classical=(name in ("head_fusion_tc", "head_fusion_adaptive_tc")),
            epochs=30 if al else 60,
            max_side=768 if al else (1024 if name == "head_fusion_tc" else 1536),
            tile_classical=(name in ("head_fusion_tile", "head_fusion_hi", "head_fusion_thin")),  # A
            instance_mode=inst,                            # B: marker watershed / blob markers
            guided_fuse=(name == "head_fusion_gf"),        # C: guided-fusion block
            boundary_head=((v2 or uni) and not dist),      # W2: boundary head (dist head replaces it)
            dist_head=dist,                                # DT-regression instance head → seeded watershed
            contrast_norm=((name in ("head_fusion_v2", "head_fusion_hi", "head_fusion_thin") or uni) and not nocolor),  # CLAHE (nocolor ablation disables it entirely, not just its adaptive strength)
            cldice=(name == "head_fusion_v2" or thin or uni),  # thin-structure soft-clDice loss
            cldice_adaptive=(name == "head_fusion_uni_adapt"),   # per-image weight ∝ (1−solidity)
            clahe_adaptive=(name in ("head_fusion_uni_adapt", "head_fusion_adaptive",
                                     "head_fusion_adaptive_thin", "head_fusion_adaptive_forcenat",
                                     "head_fusion_adaptive_color", "head_fusion_adaptive_affinity",
                              "head_fusion_adaptive_trace", "head_fusion_adaptive_tc",
                              "head_fusion_best")) and not nocolor,  # CLAHE
            color_adaptive=(name in ("head_fusion_adaptive_color", "head_fusion_best")) and not nocolor,  # colour/stain bank input
            bank_unfreeze_adaptive=(name == "head_fusion_best" and not nobank),  # tubularity-gated Frangi/Sauvola unfreeze (nobank disables — kills its latency)
            competitive_gate=cgate,                                  # per-pixel competitive group gate in the head
            corr_prior=corrp,                                        # support fg/bg correspondence-prior channel
            n_proto=(4 if mproto else 1),                            # Lever 1: k-means multi-prototype corr (mproto)
            boundary_dou=bdou_lev,                                   # Lever 2: density-gated Boundary DoU loss (bdou)
            prec_loss=prec_lev,                                      # Lever 3: dense-compact precision Tversky (prec)
            dino_only=nocls,                                         # ABLATION: zero classical bank → DINO-only head
            dino_scale=dino_scale,                                   # ABLATION: both / coarse-only / fine-only DINO
            rgb_feat=rgbfeat,                                        # A/B: append raw RGB as native features
            bankselect=bankselect,                                   # LEVER: Fisher-select classical bank channels
            scaleconf=scaleconf,                                     # LEVER: support-clustered bank scales

            proj_dim=proj_dim,                                       # ABLATION: DINO channel embedding width (pd<N>)
            film=film,                                               # support-conditioned FiLM (γ,β) hypernetwork
            bank_extra=bank_extra,                                   # roadmap #8: expanded curated classical bank
            adaptive_loss=adapt_loss,                            # full morphology-driven loss constructor
            thin_adaptive=(name in ("head_fusion_adaptive_thin", "head_fusion_adaptive_forcenat")),
            # forcenat = native gate FORCED ON for ALL support (thin_ref=-1, thin_max_side=∞) → the LODO
            # native factorial's "native ON" arm; the normal _thin alias keeps the morphology gate.
            #
            # The normal arm passes None, NOT the old fitted 1500: None means "derive the bound from the
            # head's own training cap", which is the whole point of removing the fitted constant. Naming
            # the number here would have quietly reinstated it for every benchmark, since make_backend is
            # the single path all of them go through.
            thin_ref=(-1.0 if name == "head_fusion_adaptive_forcenat" else 0.4),
            thin_max_side=(10**9 if name == "head_fusion_adaptive_forcenat" else None),
            fine_scales=(hi or thin or name == "head_fusion_uni"),  # v2 uni_adapt DROPS fine_scales (net−)
            scale_fusion=sf,                               # continuous learned scale-fusion
            encoder=enc,                                   # raw encoder for the fine native branch
            upsampler=sf_up.get(name),                     # learned upsampler (None=bilinear)
        )
    else:
        raise ValueError(f"unknown segmenter backend: {name}")
    if refine != "none":
        from active_segmenter.config import RefineConfig
        from active_segmenter.refine import build_refiner
        from active_segmenter.segment.refining_backend import RefiningBackend
        amodal = refine == "amodal"
        mode = {"point": "point", "mask": "mask", "mask_box": "mask_box", "amodal": "mask"}[refine]
        rcfg = RefineConfig(kind="sam", prompt_mode=mode, amodal=amodal, sam_negatives=not amodal)
        be = RefiningBackend(be, build_refiner(rcfg, dev))
    return be


class Ctx:
    """Everything a scorer might need, precomputed per run."""
    def __init__(self, trf, train, tef, test, cls, dev, metric="iou"):
        self.trf, self.train, self.tef, self.test = trf, train, tef, test
        self.cls, self.dev = cls, dev
        self._backend = None
        self.metric = metric      # the dataset's designated metric (registry DatasetSpec.metric)

    def set_backend(self, be):
        self._backend = be

    def _support(self, idxs):
        return [LabeledExample(self.train[i][0], self.trf[i],
                               (np.asarray(self.train[i][1]) > 0).astype(int)) for i in idxs]

    def test_iou_backend(self, idxs):
        """Score the test set with the attached SegmenterBackend using the DATASET'S metric
        (fg-IoU / clDice / instance-AP). The segmenter is what the race varies; acquisition
        is held fixed by the caller."""
        from active_segmenter.eval.scoring import primary_key, score_prediction

        be = self._backend
        be.fit(self._support(idxs))
        pk = primary_key(self.metric)
        out = []
        for j in range(len(self.test)):
            im, l = self.test[j]
            fg = be.foreground(im, self.tef[j])   # tef[j] is the cached feat_grid for test[j]
            instances = None
            if self.metric == "instance_ap":
                try:
                    instances = [m.mask for m in be.predict(im, self.tef[j])]
                except Exception:
                    instances = []
            out.append(score_prediction(self.metric, fg, l, instances)[pk])
        return float(np.mean(out))

    def bank(self, idxs):
        b = MemoryBank()
        for i in idxs:
            b.add_from_annotation(self.trf[i], (np.asarray(self.train[i][1]) > 0).astype(int), {1: 1}, 0)
        return b

    def test_iou(self, bank):
        out = []
        for f, (im, l) in zip(self.tef, self.test):
            s = corr.score_map(f, bank, 1, MC, device=self.dev)
            pf = resize((s > 0).astype(np.float32), np.asarray(l).shape, order=0,
                        mode="edge", anti_aliasing=False) > 0.5
            out.append(metrics.foreground_iou(pf, l))
        return float(np.mean(out))

    def uncertainty(self, i, bank):
        return ambiguous_fraction(corr.score_map(self.trf[i], bank, 1, MC, device=self.dev),
                                  MC.fg_bg_margin_eps)

    def typicality(self):
        if not hasattr(self, "_typ"):
            self._typ = transductive.typicality(self.cls, k=20)
        return self._typ


# ---- scorers: (pool, bank_idxs, ctx) -> {i: score}, higher = pick first ----
def s_random(pool, idxs, ctx, rng):
    return {i: float(rng.random()) for i in pool}


def s_uncertainty(pool, idxs, ctx, rng):
    bank = ctx.bank(idxs)
    return {i: ctx.uncertainty(i, bank) for i in pool}


def s_epig(pool, idxs, ctx, rng):
    bank = ctx.bank(idxs)
    cls = ctx.cls
    return {i: ctx.uncertainty(i, bank) * float(np.mean(cls[i] @ cls[pool].T)) for i in pool}


def s_typiclust(pool, idxs, ctx, rng):
    # transductive coverage of the uploaded dataset; model-free, uses only embeddings
    return transductive.typiclust_rank_scores(ctx.cls, idxs, pool, ctx.typicality(), seed=0)


def s_proxy_eer(pool, idxs, ctx, rng):
    # exact one-step Expected-Error-Reduction look-ahead via the free bank retrain
    return transductive.proxy_eer_scores(pool, ctx.bank(idxs), ctx.trf, ctx.tef, MC, ctx.dev)


def s_smec(pool, idxs, ctx, rng):
    # Support-Marginal Error Coverage: label-free error field from disagreement of the
    # frozen segmenter across support SUBSETS, covered via facility location in CLS space.
    bank_cache = {}

    def predict_fn(support, target):
        key = tuple(support)
        bank = bank_cache.get(key)
        if bank is None:
            bank = ctx.bank(list(support))
            bank_cache[key] = bank
        return corr.score_map(ctx.trf[target], bank, 1, MC, device=ctx.dev) > 0

    seed = int(rng.integers(1 << 30))
    return smec.smec_scores(pool, idxs, ctx.cls, predict_fn,
                            n_committee=4, subset_frac=0.6, seed=seed, min_support=2)


def s_hybrid(pool, idxs, ctx, rng):
    # union of the two complementary model-aware signals: SMEC (disagreement-coverage)
    # + proxy-EER (confidence-sharpening), fused by z-score over the pool.
    smec_sc = s_smec(pool, idxs, ctx, rng)
    eer_sc = s_proxy_eer(pool, idxs, ctx, rng)
    return smec.zscore_fuse([smec_sc, eer_sc])


def _grid_gt(ctx, sidx):
    """Known GT foreground on the feature grid for support image ``sidx`` (non-square safe)."""
    lab = np.asarray(ctx.train[sidx][1]) > 0
    gh, gw = ctx.trf[sidx].shape[:2]
    return resize(lab.astype(np.float32), (gh, gw), order=0, mode="edge",
                  anti_aliasing=False) > 0.5


def s_geoloop(pool, idxs, ctx, rng):
    """Frozen-loop acquisition: foreground-patch coverage (on-sphere novelty of a
    candidate's fg patches vs the bank's fg patches) fused with support leave-one-out
    GT-grounded error (propagated to the pool). One geometry serves both signals; see
    active_segmenter/acquire/{fg_coverage,support_loo}.py."""
    bank = ctx.bank(idxs)
    bank_fg = bank.fg(1)                                   # [Nb, D] bank foreground patches
    # foreground-candidate patches per pool image = grid patches the bank labels fg
    cand_fg = {}
    for i in pool:
        s = corr.score_map(ctx.trf[i], bank, 1, MC, device=ctx.dev)
        feat = ctx.trf[i]
        m = s > 0
        cand_fg[i] = feat[m] if m.any() else np.zeros((0, feat.shape[-1]), np.float32)
    cov = fg_coverage.fg_coverage_scores(cand_fg, bank_fg)

    def predict_fn(others, sidx):
        return corr.score_map(ctx.trf[sidx], ctx.bank(list(others)), 1, MC, device=ctx.dev) > 0

    errs = support_loo.support_loo_errors(idxs, predict_fn, lambda s: _grid_gt(ctx, s))
    loo = support_loo.propagate_errors_to_pool(pool, errs, ctx.cls)
    return smec.zscore_fuse([cov, loo])


def _badge_embeddings(pool, idxs, ctx):
    """Fit the head backend on the current labels, then read its BADGE gradient embedding for
    each pool candidate. Requires a TrainableHeadBackend attached (weight-coupled acquisition
    only makes sense with a trainable segmenter)."""
    import inspect

    be = getattr(ctx, "_backend", None)
    if be is None or not hasattr(be, "grad_embedding"):
        raise RuntimeError("badge/egl acquisition needs --segmenter head (a trainable backend)")
    be.fit(ctx._support(idxs))
    # head_fusion's grad_embedding is (image, feat_grid) — classical needs the image; plain head is (feat_grid)
    img_aware = len(inspect.signature(be.grad_embedding).parameters) >= 2
    if img_aware:
        return {i: be.grad_embedding(ctx.train[i][0], ctx.trf[i]) for i in pool}
    return {i: be.grad_embedding(ctx.trf[i]) for i in pool}


def s_egl(pool, idxs, ctx, rng):
    """Expected-gradient-length: ||head-gradient|| per candidate (weight-coupled uncertainty)."""
    return badge.egl_scores(_badge_embeddings(pool, idxs, ctx))


def s_badge(pool, idxs, ctx, rng):
    """BADGE as a per-item score: k-means++ selection ORDER over head-gradient embeddings turned
    into descending scores (so argmax/batch both work). The batch path (_select_batch) calls
    badge_select directly; this scalarisation lets the sequential loop use it too."""
    emb = _badge_embeddings(pool, idxs, ctx)
    order = badge.badge_select(emb, k=len(pool), seed=int(rng.integers(1 << 30)))
    rank = {i: len(order) - r for r, i in enumerate(order)}   # first-picked = highest score
    return {i: float(rank.get(i, 0)) for i in pool}


SCORERS = {
    "random": s_random,
    "uncertainty": s_uncertainty,
    "epig": s_epig,
    "typiclust": s_typiclust,
    "proxy_eer": s_proxy_eer,
    "smec": s_smec,
    "hybrid": s_hybrid,
    "geoloop": s_geoloop,
    "egl": s_egl,
    "badge": s_badge,
}


def oracle_gains(pool, idxs, ctx, base):
    b0 = None
    out = {}
    for i in pool:
        out[i] = ctx.test_iou(ctx.bank(idxs + [i])) - base
    return out


def _base_iou(ctx, idxs):
    """Evaluate the current labeled set: through the attached SegmenterBackend if one is
    set (the segmenter race), else through the default correspondence bank (acquisition
    runs, unchanged)."""
    if getattr(ctx, "_backend", None) is not None:
        return ctx.test_iou_backend(idxs)
    return ctx.test_iou(ctx.bank(idxs))


def _select_batch(scorer, pool, idxs, ctx, rng, batch, method=None):
    """Pick the next label(s). batch<=1 = argmax of the scorer (sequential, most adaptive).
    batch>1 = a diverse batch. BADGE uses k-means++ over the head's gradient embeddings
    (its native selection); other methods use submodular facility location over the pool."""
    if method == "badge" and batch > 1:
        emb = _badge_embeddings(pool, idxs, ctx)
        return badge.badge_select(emb, batch, seed=int(rng.integers(1 << 30)))
    sc = scorer(pool, idxs, ctx, rng)
    if batch <= 1:
        return [max(pool, key=lambda i: sc[i])]
    order = list(pool)
    scores = np.array([sc[i] for i in order], np.float32)
    w = scores - scores.min()                                  # shift to non-negative weight
    cls = np.asarray(ctx.cls, np.float32)[order]
    unit = cls / np.maximum(np.linalg.norm(cls, axis=1, keepdims=True), 1e-12)
    sim = np.maximum(0.0, unit @ unit.T)                        # [P, P] on-sphere coverage
    return [order[j] for j in submodular.facility_location_greedy(sim, w, batch)]


def run_curve(scorer, cold, pool0, ctx, rounds, rng, batch=1, method=None):
    # SEED INDEPENDENCE: the testbed reuses ONE backend across seeds (set_backend is called once), so
    # a trainable head must be reset per seed or seed s+1 warm-starts from seed s's weights and the
    # multi-seed error bars are contaminated. TrainableHeadBackend handles this via warm_start=False;
    # HeadFusionBackend only rebuilds when head is None → reset it here (matches sota_final's per-seed
    # `be.head = None`). Only the head is reset; the image-id-keyed feature/classical caches and the
    # per-dataset gate state (_thin_active/_clahe_strength) are correct to keep within one dataset.
    be = getattr(ctx, "_backend", None)
    if be is not None and hasattr(be, "head"):
        be.head = None
    idxs = list(cold)
    pool = [p for p in pool0 if p not in idxs]
    curve = []
    for r in range(rounds):
        base = _base_iou(ctx, idxs)
        curve.append((len(idxs), base))
        if r == rounds - 1 or not pool:
            break
        for p in _select_batch(scorer, pool, idxs, ctx, rng, batch, method=method):
            if p in pool:
                idxs.append(p)
                pool.remove(p)
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/asg_cache")
    ap.add_argument("--data", default="/tmp/dsb2018")
    ap.add_argument("--pool", type=int, default=30)
    ap.add_argument("--test", type=int, default=30)
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--methods", default="random,uncertainty,epig")
    ap.add_argument("--dataset", choices=["dsb2018", "heterogeneous", "fewshot"], default="dsb2018")
    ap.add_argument("--domains", type=int, default=4)
    ap.add_argument("--no-oracle", dest="oracle", action="store_false",
                    help="skip the expensive greedy-oracle upper bound")
    ap.add_argument("--segmenter", default=None,
                    choices=["correspondence", "head", "insid3", "sam3",
                             "head_fusion", "head_fusion_sf", "head_fusion_blob"],
                    help="race a SegmenterBackend for the fg-IoU curve (acquisition held fixed)")
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m",
                    help="encoder model id (e.g. facebook/dinov3-convnext-large-pretrain-lvd1689m)")
    ap.add_argument("--backbone", default="auto", choices=["auto", "vit", "convnext"])
    ap.add_argument("--stage", type=int, default=2,
                    help="ConvNeXt hidden_states stage: 1=stride4 2=stride8 3=stride16 4=stride32")
    ap.add_argument("--batch", type=int, default=1,
                    help="labels acquired per round; >1 uses submodular (1-1/e) batch selection")
    ap.add_argument("--skew", action="store_true",
                    help="skewed (long-tail) domain distribution for --dataset heterogeneous")
    args = ap.parse_args()

    cfg = RunConfig(device="auto", cache_dir=args.cache,
                    encoder=EncoderConfig(model_id=args.model, resolution=args.res,
                                          backbone=args.backbone, convnext_stage=args.stage))
    dev = cfg.device_resolved()
    if args.dataset == "heterogeneous":
        # balanced pool + balanced test across all domains: the coverage advantage is
        # coupon-collector (random misses domains; TypiClust covers them systematically)
        train = make_heterogeneous(args.pool, n_domains=args.domains, skew=args.skew, seed=100)
        test = make_heterogeneous(args.test, n_domains=args.domains, skew=False, seed=200)
    elif args.dataset == "fewshot":
        train = load_fewshot(args.data, "support", args.pool)
        test = load_fewshot(args.data, "test", args.test)
    else:
        train = load_dsb2018(args.data, "train", args.pool)
        test = load_dsb2018(args.data, "test", args.test)
    enc = CachedEncoder(cfg, dev, args.cache)
    ctx = Ctx([enc.extract(im) for im, _ in train], train,
              [enc.extract(im) for im, _ in test], test,
              np.stack([enc.extract_cls(im) for im, _ in train]), dev)
    if args.segmenter:
        ctx.set_backend(make_backend(args.segmenter, cfg, dev, enc=enc))
    pool0 = list(range(len(train)))
    methods = [m for m in args.methods.split(",") if m in SCORERS]
    print(f"device={dev} pool={len(train)} test={len(test)} rounds={args.rounds} "
          f"seeds={args.seeds} methods={methods}", flush=True)

    cold0 = coldstart.typiclust(ctx.cls, 3, seed=0)

    # per-method learning curves (mean over seeds) — the coverage advantage shows at
    # LOW budget, so print the whole curve, not just the final point.
    curves = {}
    for m in methods:
        per_seed = []
        for seed in range(args.seeds):
            cold = coldstart.typiclust(ctx.cls, 3, seed=seed)
            per_seed.append(run_curve(SCORERS[m], cold, pool0, ctx, args.rounds,
                                      np.random.default_rng(seed), batch=args.batch, method=m))
        ns = [n for n, _ in per_seed[0]]
        mean = [float(np.mean([per_seed[s][r][1] for s in range(args.seeds)])) for r in range(len(ns))]
        std = [float(np.std([per_seed[s][r][1] for s in range(args.seeds)])) for r in range(len(ns))]
        curves[m] = (ns, mean, std)

    oracle_curve = None
    if args.oracle:
        oracle_curve = run_curve(
            lambda pool, idxs, c, rng: oracle_gains(pool, idxs, c, c.test_iou(c.bank(idxs))),
            cold0, pool0, ctx, args.rounds, np.random.default_rng(0))

    # rank-correlation to the true per-candidate gain at the cold bank
    base = ctx.test_iou(ctx.bank(cold0))
    cand = [i for i in pool0 if i not in cold0]
    gains = np.array([ctx.test_iou(ctx.bank(list(cold0) + [i])) - base for i in cand])

    ns = curves[methods[0]][0]
    print("\n# learning curves (test fg IoU, mean over seeds)")
    header = f"{'n':>4} " + " ".join(f"{m:>12}" for m in methods)
    if oracle_curve is not None:
        header += f" {'oracle':>10}"
    print(header)
    for r in range(len(ns)):
        row = f"{ns[r]:>4} " + " ".join(f"{curves[m][1][r]:>12.3f}" for m in methods)
        if oracle_curve is not None:
            row += f" {oracle_curve[r][1]:>10.3f}"
        print(row)

    print("\n# final IoU + rank correlation to true per-candidate gain")
    print(f"{'method':>12} {'final_IoU':>16} {'rho_to_gain':>12}")
    for m in methods:
        sc = SCORERS[m](cand, list(cold0), ctx, np.random.default_rng(0))
        rho, _ = spearmanr([sc[i] for i in cand], gains)
        ns_m, mean_m, std_m = curves[m]
        print(f"{m:>12} {mean_m[-1]:>10.3f} +/-{std_m[-1]:>4.3f} {rho:>+12.3f}")
    print("DONE")


if __name__ == "__main__":
    main()
