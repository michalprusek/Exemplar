"""P0-2 — Leave-one-dataset-out generalisation of the self-configuration gates (the decisive novelty
experiment; defuses the "thresholds tuned on the eval panel" attack, M2/M5).

For each held-out dataset D: calibrate each gate's threshold on the OTHER N-1 datasets' (morphology →
lever-helps) labels, then predict D's lever settings from D's SUPPORT morphology alone, and check the
prediction against the ORACLE (what actually helps D, from the clean ablation). If the held-out
predictions match the oracle, the self-config generalises to UNSEEN morphology → not overfit.

Two gates (REPLICAS of the deployed rules — superres branches on is_instance from dataset metadata,
not support morphology; both size/downscale terms are calibrated on N-1):
  superres_on  ⇔  NOT is_instance  AND  mean_side/res <= τ_downscale
  native_on    ⇔  tubularity > τ_thin  AND  mean_side <= τ_side
Ground-truth "helps":
  superres_helps[D] = score(A2)-score(A1) > 0            (clean ladder: adaptive-loss +superres vs not)
  native_helps[D]   = score(naton)-score(natoff) > 0     (FACTORIAL results/scores_fac: native FORCED on
                      vs off on every dataset — NOT the gated ladder A3-A2, which is ~0 run-noise wherever
                      the gate correctly keeps native off, so the ladder measured noise (native LODO 0/N))

Run AFTER the clean ablation (A0..A3 on all 7) is complete. On tulen:
  PYTHONPATH=/disk1/prusek/active-segmenter PANEL_DL_ROOT=/disk1/prusek/panel_datasets \
    /home/prusek/dinov3_env/bin/python scripts/lodo_eval.py
"""
import json
import os

import numpy as np

from active_segmenter.eval.registry import PANEL, load_dataset
from active_segmenter.segment.head_fusion_backend import _tubularity

RES = 672
SCORES = "results/scores_clean"
FAC = "results/scores_fac"                       # native FACTORIAL (naton vs natoff, one cache)
METHOD = {"A0": "head_fusion_uni_adapt", "A1": "head_fusion_adaptive",
          "A2": "head_fusion_adaptive", "A3": "head_fusion_adaptive_thin"}
FACM = {"natoff": "head_fusion_adaptive", "naton": "head_fusion_adaptive_forcenat"}


def _score(tag, ds):
    fp = os.path.join(SCORES, tag, f"{METHOD[tag]}__{ds}.json")
    return float(np.mean(json.load(open(fp))["per_image"])) if os.path.exists(fp) else None


def _fac(arm, ds):
    fp = os.path.join(FAC, arm, f"{FACM[arm]}__{ds}.json")
    return float(np.mean(json.load(open(fp))["per_image"])) if os.path.exists(fp) else None


def descriptors(datasets):
    """Per-dataset support-derived features (seed-0 K=8 subsample) — no oracle helps-labels (reads the
    support GT MASKS only, to compute tubularity/size, exactly what the deployed gates see)."""
    d = {}
    for name in datasets:
        pool, _ = load_dataset(PANEL[name], 20, 24, seed=0)
        sub = list(np.random.default_rng(0).choice(len(pool), 8, replace=False))
        sides = [max(np.asarray(pool[i][0]).shape[:2]) for i in sub]
        tubs = [_tubularity(np.asarray(pool[i][1])) for i in sub]
        d[name] = dict(mean_side=float(np.mean(sides)), downscale=float(np.mean(sides)) / RES,
                       tub=float(np.mean(tubs)), is_instance=PANEL[name].metric == "instance_ap")
    return d


def _fit_threshold(pairs, want_high_helps):
    """1-D threshold on `x` separating helps(True)/hurts(False). Returns τ (midpoint of the best split).
    want_high_helps=True → helps when x<=τ (superres: helps at LOW downscale). Robust to a clean split;
    if none exists, returns the midpoint minimising errors."""
    if not pairs:
        raise ValueError("_fit_threshold: no calibration pairs for this fold (cannot learn a threshold)")
    xs = sorted({x for x, _ in pairs})
    cands = [(-1e9)] + [(a + b) / 2 for a, b in zip(xs, xs[1:])] + [1e9]
    best, err = cands[0], 1e9
    for t in cands:
        e = sum((x <= t) != h for x, h in pairs) if want_high_helps else sum((x > t) != h for x, h in pairs)
        if e < err:
            err, best = e, t
    return best


def main():
    ALL = ["spheroid", "spheroidj", "dsb2018", "rozpad", "kvasir", "hrf", "microtubules",
           "drive", "isbi2012em", "monuseg", "ctc_u373"]                       # full 11-dataset panel
    def ready(d):  # superres helps needs A1/A2 (clean ladder); native helps needs the FACTORIAL arms
        return (_score("A1", d) is not None and _score("A2", d) is not None
                and _fac("natoff", d) is not None and _fac("naton", d) is not None)
    datasets = [d for d in ALL if ready(d)]
    missing = [d for d in ALL if d not in datasets]
    if missing:  # M1: never report a partial panel as if complete
        print(f"!!! WARNING: LODO over {len(datasets)}/{len(ALL)} datasets — MISSING scores "
              f"for {missing} (need clean A1/A2 + factorial naton/natoff)")
    desc = descriptors(datasets)
    # superres helps: from the clean ladder (A2-A1). native helps: from the FACTORIAL (naton-natoff), NOT
    # the gated ladder (A3-A2) — the ladder gates native OFF for most datasets so A3-A2 is pure run-noise,
    # which made the earlier native-gate LODO measure noise (0/N). The factorial forces native on/off for ALL.
    sup_help = {d: _score("A2", d) - _score("A1", d) > 0 for d in datasets}
    nat_help = {d: _fac("naton", d) - _fac("natoff", d) > 0 for d in datasets}
    print(f"{'dataset':13s} {'downscale':>9s} {'tub':>5s} {'inst':>5s} | "
          f"{'sup_helps':>9s} {'nat_helps':>9s}")
    for d in datasets:
        print(f"{d:13s} {desc[d]['downscale']:9.2f} {desc[d]['tub']:5.2f} {str(desc[d]['is_instance']):>5s} | "
              f"{str(sup_help[d]):>9s} {str(nat_help[d]):>9s}")

    print("\n===== LODO: calibrate gate thresholds on N-1, predict held-out =====")
    sup_ok = nat_ok = 0
    for held in datasets:
        rest = [d for d in datasets if d != held]
        # superres gate: instance always OFF; else threshold on downscale (helps at low downscale)
        tau_ds = _fit_threshold([(desc[d]['downscale'], sup_help[d]) for d in rest if not desc[d]['is_instance']],
                                want_high_helps=True)
        pred_sup = (not desc[held]['is_instance']) and (desc[held]['downscale'] <= tau_ds)
        # native gate = REPLICA of the deployed rule (thin AND small): fit τ_tub AND τ_side on `rest`,
        # predict the conjunction. Omitting τ_side validated a weaker gate and let huge-thin hrf (most
        # tubular, native-hurts) drag τ_tub and flip genuinely-thin sets — so both conjuncts are required.
        tau_tub = _fit_threshold([(desc[d]['tub'], nat_help[d]) for d in rest], want_high_helps=False)
        tau_side = _fit_threshold([(desc[d]['mean_side'], nat_help[d]) for d in rest], want_high_helps=True)
        pred_nat = (desc[held]['tub'] > tau_tub) and (desc[held]['mean_side'] <= tau_side)
        s_match, n_match = pred_sup == sup_help[held], pred_nat == nat_help[held]
        sup_ok += s_match
        nat_ok += n_match
        print(f"  hold {held:13s} τ_ds={tau_ds:5.2f} τ_tub={tau_tub:4.2f} τ_side={tau_side:5.0f} | "
              f"superres pred={str(pred_sup):>5s} oracle={str(sup_help[held]):>5s} {'✓' if s_match else '✗'} | "
              f"native pred={str(pred_nat):>5s} oracle={str(nat_help[held]):>5s} {'✓' if n_match else '✗'}")
    n = len(datasets)
    print(f"\nLODO accuracy — superres gate {sup_ok}/{n}, native gate {nat_ok}/{n}")
    print("LODO_DONE")


if __name__ == "__main__":
    main()
