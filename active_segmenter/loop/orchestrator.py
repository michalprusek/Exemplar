"""Active-learning correct-and-advance orchestrator.

Round 0 seeds the bank from a cold-start selection; each round pre-labels the test
set (correspondence -> cluster -> refine), records fg IoU + instance AP, then the
acquisition proposes the next pool image(s), the oracle reveals their labels, and
the bank grows and is curated. Always run against a random control arm via
:func:`run_al_vs_random` (shared cold start so round 0 is identical).
"""
from __future__ import annotations

import numpy as np

from active_segmenter.acquire import build_acquisition, coldstart, diversity
from active_segmenter.acquire.base import AcqContext
from active_segmenter.acquire.uncertainty import ambiguous_fraction
from active_segmenter.eval import metrics
from active_segmenter.eval.oracle import GtOracle
from active_segmenter.loop.state import RoundResult
from active_segmenter.membank.bank import MemoryBank
from active_segmenter.propose import correspondence, instances


class ALLoop:
    def __init__(self, cfg, encoder, refiner, oracle: GtOracle, seed: int = 0,
                 feat_cache: dict | None = None):
        self.cfg = cfg
        self.encoder = encoder
        self.refiner = refiner
        self.oracle = oracle
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.device = cfg.device_resolved()
        self._cache = feat_cache if feat_cache is not None else {}

    # -- feature access (in-memory cache, shared across arms) ----------------
    def _feat(self, tag, idx, image):
        key = (tag, idx)
        if key not in self._cache:
            self._cache[key] = self.encoder.extract(image)
        return self._cache[key]

    def _cls(self, idx, image):
        key = ("cls", idx)
        if key not in self._cache:
            self._cache[key] = self.encoder.extract_cls(image)
        return self._cache[key]

    # -- prelabel + metrics --------------------------------------------------
    def _predict(self, image, feat, bank) -> list:
        insts = []
        for cid, s in correspondence.prelabel(feat, bank, self.cfg.match, device=self.device).items():
            grid = instances.decompose(
                s, self.cfg.cluster, cid,
                feat_grid=feat if self.cfg.cluster.use_features else None)
            insts += instances.upsample_masks(grid, np.asarray(image).shape[:2])
        if insts:
            insts = self.refiner.refine(image, insts, feat_grid=feat)
        return insts

    def _evaluate(self, test, bank) -> tuple[float, float]:
        fg_ious, aps = [], []
        for ti, (im, lbl) in enumerate(test):
            feat = self._feat("test", ti, im)
            shape = np.asarray(lbl).shape
            insts = self._predict(im, feat, bank)
            masks = [m.mask for m in insts]
            union = np.any(masks, axis=0) if masks else np.zeros(shape, bool)
            fg_ious.append(metrics.foreground_iou(union, lbl))
            aps.append(metrics.instance_ap(masks or [np.zeros(shape, bool)], gt_labels=lbl)["ap"])
        return float(np.mean(fg_ious)), float(np.mean(aps))

    # -- bank + acquisition --------------------------------------------------
    def _add(self, bank, idx, round):
        img = self.oracle.image(idx)
        label = self.oracle.reveal(idx)
        feat = self._feat("pool", idx, img)
        binlabel = (np.asarray(label) > 0).astype(int)
        if binlabel.any():
            bank.add_from_annotation(feat, binlabel, {1: 1}, round=round)

    def _pool_uncertainty(self, pool, bank) -> np.ndarray:
        n = len(self.oracle.dataset)
        unc = np.zeros(n, np.float32)
        classes = bank.classes()
        for i in pool:
            feat = self._feat("pool", i, self.oracle.image(i))
            fr = [ambiguous_fraction(
                correspondence.score_map(feat, bank, c, self.cfg.match, device=self.device),
                self.cfg.match.fg_bg_margin_eps) for c in classes]
            unc[i] = float(np.mean(fr)) if fr else 0.0
        return unc

    def _pool_cls_array(self) -> np.ndarray:
        return np.stack([self._cls(i, self.oracle.image(i)) for i in range(len(self.oracle.dataset))])

    def _select(self, acq, pool, bank, labeled) -> list[int]:
        cls = self._pool_cls_array()
        if not hasattr(self, "_typ"):
            from active_segmenter.acquire import transductive
            self._typ = transductive.typicality(cls, k=20)
        ctx = AcqContext(uncertainty=self._pool_uncertainty(pool, bank), cls=cls, rng=self.rng,
                         labeled=list(labeled), typicality=self._typ)
        ranked = acq.rank(pool, ctx)
        batch = max(1, self.cfg.acquire.topk_batch)
        if batch == 1:
            return ranked[:1]
        top = ranked[: max(batch, 4 * batch)]
        sub = diversity.kcenter_select(cls[top], batch, chosen=[])
        return [top[j] for j in sub]

    # -- main loop -----------------------------------------------------------
    def run(self, pool_idx, test, rounds, cold_k, acquisition_name=None,
            cold_start_picks=None) -> list[RoundResult]:
        acq_name = acquisition_name or self.cfg.acquire.strategy
        acq = build_acquisition(acq_name)
        bank = MemoryBank()
        pool = list(pool_idx)

        if cold_start_picks is None:
            cls = np.stack([self._cls(i, self.oracle.image(i)) for i in pool])
            local = coldstart.typiclust(cls, cold_k, seed=self.seed)
            cold_start_picks = [pool[j] for j in local]
        for p in cold_start_picks:
            self._add(bank, p, round=0)
            if p in pool:
                pool.remove(p)

        results, history = [], []
        for r in range(rounds):
            bank.curate(self.cfg.acquire.bank_cap, seed=self.seed)
            fg_iou, ap = self._evaluate(test, bank)
            results.append(RoundResult(n_annotated=self.oracle.n_revealed, fg_iou=fg_iou,
                                       instance_ap=ap, arm=acq_name))
            history.append({"iou": fg_iou, "correction_rate": 1.0, "acq_score": 1.0})
            if r == rounds - 1 or not pool:
                break
            labeled = [i for i in pool_idx if i not in pool]
            for p in self._select(acq, pool, bank, labeled):
                self._add(bank, p, round=r + 1)
                if p in pool:
                    pool.remove(p)
        return results


def run_al_vs_random(cfg, pool, test, rounds, cold_k, encoder=None, refiner=None, seed=0):
    """Run AL and a random control arm sharing the same cold start + feature cache."""
    if encoder is None:
        from active_segmenter.encoder.dinov3 import Dinov3Encoder
        encoder = Dinov3Encoder(cfg.encoder, cfg.device_resolved())
    if refiner is None:
        from active_segmenter.refine import build_refiner
        refiner = build_refiner(cfg.refine, cfg.device_resolved())

    shared_cache: dict = {}
    cls = np.stack([_cls_cached(shared_cache, encoder, i, pool[i][0]) for i in range(len(pool))])
    cold_local = coldstart.typiclust(cls, cold_k, seed=seed)

    out = {}
    for arm, name in (("al", cfg.acquire.strategy), ("random", "random")):
        loop = ALLoop(cfg, encoder, refiner, GtOracle(pool), seed=seed, feat_cache=shared_cache)
        out[arm] = loop.run(list(range(len(pool))), test, rounds, cold_k,
                            acquisition_name=name, cold_start_picks=list(cold_local))
    return out


def _cls_cached(cache, encoder, idx, image):
    key = ("cls", idx)
    if key not in cache:
        cache[key] = encoder.extract_cls(image)
    return cache[key]
