#!/usr/bin/env python
"""Publication-grade SOTA comparison harness — runs our method + in-harness baselines at their BEST
config, dumps PER-IMAGE scores, aggregates with external baselines' per-image dumps, and computes the
final table (mean±std over seeds) + PAIRED significance (Wilcoxon signed-rank + bootstrap 95% CI +
Cliff's δ) of our method vs EACH baseline, per dataset.

Two-stage design so heterogeneous baselines (each in its own env) all feed one stats step:
  STAGE `run`  — run an IN-HARNESS backend (uni_adapt / insid3@1024 / universeg) over the panel, writing
                 per-image scores to `results/scores/<method>__<dataset>.json`. External-env baselines
                 (persam/microsam/cellpose/stardist/sam3) write the SAME json format from their own scripts.
  STAGE `stats`— load every `results/scores/*.json`, build the mean±std table, and for each dataset run
                 paired Wilcoxon(our, baseline) over the shared per-image scores (same seeds→splits →
                 aligned), with bootstrap CI + Cliff's δ.

Score-json contract (ALL methods must emit this): {"method","dataset","metric","test_per_seed":N,
"seeds":[...], "per_image":[float,...]}  where per_image is seed-major (seed0's N images, seed1's N, ...).

  # run ours + baselines that live in dinov3_env (insid3 uses --res 1024 = its best config):
  PYTHONPATH=. python scripts/sota_final.py run --method head_fusion_uni_adapt --res 672
  PYTHONPATH=. python scripts/sota_final.py run --method insid3 --res 1024
  PYTHONPATH=. python scripts/sota_final.py run --method universeg --res 672
  # after all methods (incl external envs) have written results/scores/*.json:
  PYTHONPATH=. python scripts/sota_final.py stats --ours head_fusion_uni_adapt
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import traceback

import numpy as np

# Imported at MODULE scope on purpose. It used to be imported inside ``run_inharness`` only, which
# left the module-level ``PANEL_DATASETS`` below raising NameError at import time — i.e. neither
# ``run`` nor ``stats`` could start at all. The registry itself pulls in numpy/skimage only (no
# torch), so hoisting it costs nothing even for the light ``stats`` path.
from active_segmenter.eval.registry import PANEL  # noqa: E402

# Derived from the registry, never hand-maintained: a hand-written list silently drops any
# dataset missing from it (written to disk, absent from every table and paired test).
PANEL_DATASETS = list(PANEL)  # + 4 public (WACV P0-4)
SCORE_DIR_DEFAULT = "results/scores"


def _score_path(score_dir, method, dataset):
    return os.path.join(score_dir, f"{method}__{dataset}.json")


# The campaign writes one directory PER (method, K) — ``{tree}/{method}_k{K}/`` — plus a bare
# ``{tree}/{method}/`` for the methods that take no support at all. ``stats`` used to glob a single
# flat directory, so it could not read a campaign tree at ALL: no p-value in the paper could be
# produced from the run that generates the paper's numbers.
_K_DIR = re.compile(r"_k(\d+)$")


def _seed_rows(rec):
    """``per_image`` reshaped to a list of per-seed rows, or None if the shape is inconsistent."""
    n = int(rec.get("test_per_seed") or 0)
    flat = rec.get("per_image") or []
    if n <= 0 or not flat or len(flat) % n:
        return None
    return [flat[i * n:(i + 1) * n] for i in range(len(flat) // n)]


def _looks_support_blind(rec):
    """Does this record's OWN DATA show that the support draw had no effect?

    A method that ignores the support masks scores the fixed test split identically under every
    seed, because the seeds only re-draw the support. So support-blindness is a checkable property
    of the numbers, not a claim to be taken from a directory name.

    This matters because the off-the-shelf specialists still record ``protocol.support`` (their
    argparse default, kept for provenance), so "no support recorded" does NOT identify them. Trusting
    the directory instead would let any record dropped into a ``_k``-less directory — including a
    genuinely K-dependent in-context method — be announced as support-blind and replicated into
    every K table, with paired p-values computed against it.

    Returns None when there are fewer than two seeds, i.e. when the property cannot be checked
    either way; the caller refuses rather than guessing.
    """
    rows = _seed_rows(rec)
    if rows is None or len(rows) < 2:
        return None
    return all(r == rows[0] for r in rows)


def _label_of(sub):
    """Campaign job label from a directory name: ``insid3_guided_k8`` -> ``insid3_guided``."""
    return _K_DIR.sub("", sub)


def _discover_records(score_dir, support):
    """Yield ``(path, record, label, k)`` for every score record under ``score_dir``.

    Accepts BOTH layouts: records sitting directly in ``score_dir`` (the historical flat tree) and
    records one level down in per-(method, K) subdirectories (what ``run_campaign.py`` writes).
    Nesting stops at one level on purpose rather than using ``**``: a recursive glob would happily
    absorb an unrelated tree that happens to live below the score directory, and the campaign
    layout is exactly one level deep.

    IDENTITY IS THE DIRECTORY LABEL, not the record's ``method`` field. ``run_campaign.py`` maps
    several job labels onto one harness method: ``insid3_guided`` and ``insid3_dense`` are both
    ``--method insid3``, so their records carry an identical ``(dataset, method)`` and keying on that
    made every INSID3 pair collide. The first version of this function did exactly that and aborted
    on the real campaign tree, which is to say it never read the tree it was written to read.

    ``k`` comes from the ``_k<N>`` suffix and is cross-checked against ``protocol.support``.
    Disagreement is fatal. Note what this check can and cannot do: it catches a directory whose label
    contradicts the run's own record, but NOT a run whose measurement was silently truncated while it
    faithfully recorded the K it was ASKED for — there the record, the directory and the request all
    agree and only the measurement is wrong. That case is real (a baseline truncating its support
    list) and is caught by comparing source code across hosts, not here.
    """
    paths = sorted(glob.glob(os.path.join(score_dir, "*.json")))
    paths += sorted(glob.glob(os.path.join(score_dir, "*", "*.json")))
    for fp in paths:
        try:
            with open(fp) as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError) as e:   # a 0-byte file from a crashed run must NAME
            print(f"  [stats] skipping unreadable {fp}: {e!r}", flush=True)   # itself, not kill the stage
            continue
        if not isinstance(rec, dict) or not all(
                k in rec for k in ("dataset", "method", "per_image", "test_per_seed")):
            # Not necessarily an error: a campaign tree also holds summary/aggregate files that were
            # never meant to be score records. Say so and move on.
            print(f"  [stats] skipping non-record {fp} (missing score-record keys)", flush=True)
            continue

        rel = os.path.relpath(fp, score_dir)
        sub = os.path.dirname(rel)
        m = _K_DIR.search(sub) if sub else None
        dir_k = int(m.group(1)) if m else None
        label = _label_of(sub) if sub else rec["method"]
        rec_k = rec.get("protocol", {}).get("support")
        if dir_k is not None and rec_k is not None and int(rec_k) != dir_k:
            raise SystemExit(
                f"K MISLABELLED: {fp} sits in a directory declaring K={dir_k} but its record was "
                f"measured at support={int(rec_k)}. One of the two is wrong, and the difference is "
                f"invisible in every downstream table and figure. Fix the run before reporting.")
        yield fp, rec, label, dir_k


def _round16(x):
    return int(round(float(x) / 16.0) * 16)


def split_fingerprint(pairs) -> str:
    """Content digest of a loaded split — the identity check that a shape check cannot perform.

    Two runs can agree on image COUNT and still have scored different pictures: the flat-directory
    loader slices ``permutation(seed)[support : support+test]``, so the test slice moves when EITHER
    the seed or the support size changes (this harness pins ``pool``/``seed=0``, but a baseline script
    that passes ``support=8, seed=seed`` gets a different, partially disjoint test set on every
    download-kind dataset). Pairing per-image scores across such runs yields a confident but
    meaningless p-value, so every score file carries this digest and ``stats`` refuses to pair records
    that disagree on it.
    """
    h = hashlib.sha256()
    for image, label in pairs:
        for a in (np.ascontiguousarray(image), np.ascontiguousarray(label)):
            h.update(str(a.shape).encode())
            h.update(str(a.dtype).encode())
            h.update(a.tobytes())
    return h.hexdigest()[:16]


def _per_image_mean(rec):
    """Collapse a seed-major per-image vector to ONE value per test image.

    The test split is loaded once and is identical across seeds, so each image contributes one score
    per seed. Those repeats are correlated measurements of the same unit, not independent samples:
    testing all seed x image values would treat 6x24 scores as 144 independent pairs and shrink every
    p-value accordingly (pseudoreplication). Averaging over seeds first leaves the test images as the
    independent units, which is what a paired test assumes.
    """
    arr = np.asarray(rec["per_image"], float).reshape(-1, int(rec["test_per_seed"]))
    return arr.mean(0)


def adaptive_res_decision(fg_fracs, sides, base_res, min_obj_px, res_max=1536):
    """Self-configuring ENCODER RESOLUTION picker from the K support masks (NO test peeking).

    Failure regime targeted: SMALL objects in LARGE images (low support fg-fraction + big native
    side). At the efficient default res the heavily-downscaled object is under-resolved and the
    segmenter collapses — empirically (this session): PerSeg ``candle`` (fg ~3% of a 2048^2 image)
    scored 4% IoU @res672 vs 92% @res1024, and PerSeg mIoU rose 84.9% -> 90.2% just by raising res.
    We DETECT the regime from the aggregate support foreground fraction (robust; deliberately NOT a
    connected-component COUNT — a recorded mask-speckle negative) and RAISE the encoder resolution
    so the object is adequately resolved, while leaving normal-scale data (most microscopy: objects
    ~20-30% of the frame) at the efficient default.

    Signal: for the MEDIAN support image, the aggregate foreground's linear size in the base-res
    encoder INPUT is ``eff_obj_px = sqrt(fg_frac) * base_res`` (the native side S cancels for a
    single object — it is the resolution in the resized input that the ViT actually sees). Below
    ``min_obj_px`` the object is under-resolved. We only RAISE when there is genuine downscale
    headroom (median native side > base_res); an already-small image has no lost detail to recover,
    so upsampling it would just burn compute. Target res is the one at which the object reaches
    ``min_obj_px`` (= ``min_obj_px / sqrt(fg_frac)``), clipped to [base_res, min(res_max, native
    side)] (never upsample beyond the native pixels) and snapped to the patch multiple (16).

    Returns ``(target_res, info)``; ``info`` carries the logged decision fields.
    """
    med_fg = float(np.median(fg_fracs)) if len(fg_fracs) else 0.0
    med_S = float(np.median(sides)) if len(sides) else 0.0
    eff_obj_px = (med_fg ** 0.5) * base_res if med_fg > 0 else 0.0
    info = dict(fg_frac=med_fg, S=med_S, eff_obj_px=eff_obj_px, raised=False)
    if min_obj_px <= 0:                        # OFF -> byte-identical to fixed-res behaviour
        return int(base_res), info
    under = med_fg > 0 and eff_obj_px < min_obj_px            # object under-resolved at base_res
    headroom = med_S > base_res                              # only large (downscaled) images lost detail
    if under and headroom:
        raw = min_obj_px / (med_fg ** 0.5)     # res at which eff_obj_px == min_obj_px
        cap = min(float(res_max), med_S)       # never upsample beyond the native side
        target = _round16(min(max(raw, base_res), cap))
        target = max(int(base_res), min(target, int(res_max)))
        if target > base_res:
            info["raised"] = True
            return target, info
    return int(base_res), info


def run_inharness(args):
    """Run an in-harness backend over the panel and dump per-image scores per dataset."""
    from active_segmenter.config import EncoderConfig, RunConfig
    from active_segmenter.encoder.cached import CachedEncoder
    from active_segmenter.eval.registry import PANEL, load_dataset
    from active_segmenter.eval.scoring import primary_key, score_prediction
    from active_segmenter.segment.base import LabeledExample, reset_backend_for_new_support
    from scripts.al_testbed import make_backend

    dev = RunConfig(device="auto").device_resolved()
    os.makedirs(args.score_dir, exist_ok=True)
    requested = args.datasets.split(",") if args.datasets else PANEL_DATASETS
    requested = [d.strip() for d in requested if d.strip()]
    unknown = [d for d in requested if d not in PANEL]
    if unknown:                       # a typo used to silently reduce this to [] and still exit 0
        raise SystemExit(f"unknown dataset(s) {unknown}; known: {sorted(PANEL)}")
    datasets = requested
    if not datasets:
        raise SystemExit("no datasets selected — refusing to report a run that scored nothing")
    failures = []                                    # datasets that errored → fail loud at the end
    for name in datasets:
        spec = PANEL[name]
        # METRIC OVERRIDE: a SEMANTIC-only method (e.g. Matcher) is scored on FOREGROUND (fg_iou/cldice),
        # not the dataset's native instance-AP — scoring a semantic mask with instance-AP would unfairly
        # zero it on touching-object datasets. When set, ``--metric_override`` replaces the scoring metric
        # (and disables instance prediction) for THIS run; the score json records the override metric.
        eff_metric = getattr(args, "metric_override", "") or spec.metric
        pk = primary_key(eff_metric)
        # When the panel is scored on an overridden metric, ALSO record the dataset's own
        # metric. Otherwise instance AP is never computed anywhere in the campaign and the
        # instance-specialist comparison the plan calls for cannot be made at all.
        native_pk = primary_key(spec.metric) if eff_metric != spec.metric else ""
        try:
            # FIXED test + support POOL loaded ONCE (seed=0) → the test set is identical across seeds and
            # support-count-INDEPENDENT (fixes the download-kind test-slice shift that undersold hrf/kvasir);
            # K=8 support is randomly subsampled per seed from the fixed pool → proper few-shot variance,
            # the standard protocol. Same fixed test + same seed→draw for every method.
            support_pool, test = load_dataset(spec, args.pool, args.test, seed=0)
            # Identity of the scored images, recorded so `stats` can PROVE two methods were compared on
            # the same pictures rather than merely on the same COUNT of pictures.
            split_fp = split_fingerprint(test)
            # ADAPTIVE SUPER-RESOLUTION: superres helps mid-scale (spheroidj/kvasir/microtubules) but
            # REGRESSES exactly two settings — (a) INSTANCE segmentation (dsb: super-densifying the ViT grid
            # spawns spurious watershed markers → fragments instances → lower AP) and (b) HUGE-downscale images
            # (hrf 5.2×: coarse-grid superres can't recover heavily-downscaled vessels, and scale-fusion's
            # native branch already supplies that detail). BOTH predictors are robust and known at deployment
            # with zero label cost (the task's metric + the support image size); NB a connected-component
            # COUNT is NOT robust (mask speckle → kvasir reads 496 "instances") so it is deliberately unused.
            # Gate superres OFF for those two, ON (=factor) otherwise → the max(baseline, superres) envelope
            # (uniform-≥-baseline) from a support-only decision (self-configuring, now over the feature scale).
            sr = getattr(args, "superres", 1)
            res = args.res                     # per-dataset effective encoder resolution (adaptive_res may raise)
            if getattr(args, "adaptive_superres", 0):
                mean_side = float(np.mean([max(np.asarray(im).shape[:2]) for im, _ in support_pool]))
                bigdown = (mean_side / args.res) > args.sr_downscale     # hrf-like huge downscale
                is_instance = spec.metric == "instance_ap"              # dsb-like: superres frags markers
                sr = 1 if (bigdown or is_instance) else args.adaptive_superres
                print(f"  [{name}] adaptive_superres: mean_side={mean_side:.0f} bigdown={bigdown} "
                      f"instance={is_instance} → superres={sr}", flush=True)
            # ADAPTIVE RESOLUTION: raise the encoder resolution for small-objects-in-large-images
            # (see ``adaptive_res_decision``). The decision is derived ONLY from the K support masks
            # (no test peeking). Use a LOCAL ``res`` (never mutate ``args.res``) so a raise on one
            # dataset cannot leak into the next; ``cache_tag`` encodes resolution so per-res features
            # never collide on disk. When res is raised, force superres=1 — raising res already
            # densifies the ViT grid, so keeping superres would double the cost (and OOM at res_max).
            if getattr(args, "adaptive_res", 0):
                fg_fracs = [float((np.asarray(l) > 0).mean()) for _, l in support_pool]
                sides = [float(max(np.asarray(im).shape[:2])) for im, _ in support_pool]
                res, ainfo = adaptive_res_decision(fg_fracs, sides, args.res, args.adaptive_res,
                                                   getattr(args, "res_max", 1536))
                if ainfo["raised"]:
                    sr = 1
                print(f"  [{name}] adaptive_res: fg_frac={ainfo['fg_frac']:.3f} S={ainfo['S']:.0f} "
                      f"eff_obj_px={ainfo['eff_obj_px']:.0f} → res={res}"
                      f"{' (superres->1)' if ainfo['raised'] else ' (kept)'}", flush=True)
            up = getattr(args, "feat_upsampler", "none")
            if up not in (None, "none"):
                sr = 1                                            # upsampler REPLACES superres (mutually exclusive)
                print(f"  [{name}] feat_upsampler={up}×{getattr(args, 'upsampler_factor', 2)} "
                      f"(superres forced off)", flush=True)
            _lstr = getattr(args, "layers", "") or ""
            _layers = tuple(int(x) for x in _lstr.split(",") if x.strip() != "")   # layer-fusion set (or empty)
            cfg = RunConfig(device="auto", cache_dir=args.cache,
                            encoder=EncoderConfig(model_id=args.model, resolution=res,
                                                  superres_factor=sr, layer=getattr(args, "layer", -1),
                                                  layers=_layers, feat_upsampler=up,
                                                  feat_upsample_factor=getattr(args, "upsampler_factor", 2)))
            enc = CachedEncoder(cfg, dev, args.cache)
            per_image, per_image_native, seeds_used, ntest = [], [], [], None
            be = make_backend(args.method, cfg, dev, refine=args.refine, enc=enc,
                              support_k=args.support)
            T = [LabeledExample(im, enc.extract(im), np.asarray(l)) for im, l in test]
            for seed in range(args.seeds):
                sub = list(np.random.default_rng(seed).choice(len(support_pool), args.support, replace=False))
                P = [LabeledExample(support_pool[i][0], enc.extract(support_pool[i][0]),
                                    np.asarray(support_pool[i][1])) for i in sub]
                # Each seed is an INDEPENDENT draw, so the backend must forget everything the previous
                # draw configured (colour channel, CLAHE strength, thin gate, affinity calibration,
                # FiLM prototypes) — not just its trained head. Otherwise seeds 1..N reuse seed 0's
                # self-configuration and the reported std is not the std of the whole method.
                reset_backend_for_new_support(be)
                import torch
                torch.manual_seed(seed)
                be.fit(P)
                ntest = len(T)
                for ex in T:
                    fg = be.foreground(ex.image, ex.feat_grid)
                    # Instances are needed when EITHER the effective metric or the dataset's own
                    # metric is instance AP. Scoring the whole panel as foreground IoU takes away
                    # exactly the ability the instance specialists exist for (separating touching
                    # objects), so the native metric is recorded alongside it and the paper can
                    # report both tables from one run instead of choosing the flattering one.
                    want_inst = "instance_ap" in (eff_metric, spec.metric)
                    inst = [m.mask for m in be.predict(ex.image, ex.feat_grid)] if want_inst else None
                    per_image.append(float(score_prediction(eff_metric, fg, ex.label_map, inst)[pk]))
                    if native_pk:
                        per_image_native.append(
                            float(score_prediction(spec.metric, fg, ex.label_map, inst)[native_pk]))
                seeds_used.append(seed)
            if not per_image or ntest is None:
                raise RuntimeError(f"no per-image scores produced ({len(per_image)} scores, ntest={ntest})")
            arr = np.array(per_image).reshape(len(seeds_used), ntest)   # validate shape BEFORE writing
            out = dict(method=args.method, dataset=name, metric=pk, test_per_seed=ntest,
                       seeds=seeds_used, per_image=per_image, split_fp=split_fp,
                       # `loss_scale` (env ASG_LOSS_SCALE) ALTERS TRAINING but is not part of the method
                       # token grammar, so without it a record trained at 2.0 is byte-identical in its
                       # metadata to one trained at 1.0 -- and a stray `export` left in a shell would
                       # silently retrain a whole campaign under a non-default loss with nothing to show
                       # for it. CLAUDE.md requires "method + ALL flags" in the log; an env var that
                       # changes the model is a flag.
                       protocol=dict(pool=args.pool, test=args.test, support=args.support,
                                     split_seed=0, res=res,
                                     loss_scale=float(os.environ.get("ASG_LOSS_SCALE", "1.0"))))
            if native_pk:
                out["metric_native"] = native_pk
                out["per_image_native"] = per_image_native
            with open(_score_path(args.score_dir, args.method, name), "w") as f:
                json.dump(out, f)
            print(f"  [{name}] {args.method}: {pk} {arr.mean(1).mean():.3f}±{arr.mean(1).std():.3f} "
                  f"(n_img={len(per_image)}) → {_score_path(args.score_dir, args.method, name)}", flush=True)
        except Exception:
            # Do NOT silently drop a dataset — a vanished row makes a method look stronger than it is and
            # is indistinguishable from "never scheduled". Log the FULL traceback and fail loud at the end.
            failures.append(name)
            print(f"  [{name}] {args.method} FAILED:\n{traceback.format_exc()}", flush=True)
    if failures:
        print(f"SOTA_RUN_INCOMPLETE — {len(failures)}/{len(datasets)} dataset(s) FAILED: {failures}",
              flush=True)
        sys.exit(1)
    print("SOTA_RUN_DONE")


def _paired_rank_biserial(a, b):
    """Matched-pairs rank-biserial correlation — the PAIRED effect size that belongs next to a paired
    Wilcoxon. The previous all-pairs Cliff's delta compared every a_i against every b_j, discarding the
    pairing the test relies on and generally understating the effect."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]                                   # Wilcoxon drops ties; the effect size must match
    if d.size == 0:
        return 0.0
    from scipy.stats import rankdata
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())




def _bootstrap_ci(delta, iters=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(delta)
    bs = [rng.choice(d, len(d), replace=True).mean() for _ in range(iters)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _pairing_problem(ours_rec, base_rec):
    """Why these two records must NOT be paired, or ``None`` if pairing is sound.

    Each check catches a failure the others cannot. A metric mismatch makes the comparison meaningless
    even when the shapes agree (foreground IoU against instance AP is not a contest, and
    ``--metric_override`` makes that easy to produce by accident). A split mismatch makes it meaningless
    even when the metric agrees. The shape check — all this harness used to do — passes happily in both
    cases and still prints a confident p-value.
    """
    if ours_rec.get("metric") != base_rec.get("metric"):
        return f"METRIC MISMATCH ({ours_rec.get('metric')} vs {base_rec.get('metric')})"
    if ours_rec.get("test_per_seed") != base_rec.get("test_per_seed"):
        return (f"SHAPE MISMATCH (test_per_seed {ours_rec.get('test_per_seed')} vs "
                f"{base_rec.get('test_per_seed')})")
    fa, fb = ours_rec.get("split_fp"), base_rec.get("split_fp")
    if fa and fb and fa != fb:
        return (f"DIFFERENT TEST IMAGES (split {fa} vs {fb}) — check the baseline's load_dataset "
                f"protocol: {ours_rec.get('protocol')} vs {base_rec.get('protocol')}")
    return None


def stats(args):
    from scipy.stats import wilcoxon

    support = getattr(args, "support", None)
    data = {}                                       # dataset -> label -> record
    origin = {}                                     # (dataset, label) -> path, for collision reports
    k_free = set()                                  # labels with no K axis, reported below
    n_seen = n_bad = 0
    other_k = {}                                    # label -> set of K it DID run at, for accounting
    for fp, r, label, dir_k in _discover_records(args.score_dir, support):
        n_seen += 1
        # Validate the reshape HERE rather than letting numpy raise deep inside the table loop, where
        # an uncaught ValueError killed the whole stage before printing a single row and never named
        # the offending file. A length that is not a whole number of seeds means a crashed or
        # truncated run, so the record is unusable either way — name it and drop it. This runs BEFORE
        # the record can be admitted anywhere, including into `k_free`: a method whose only record is
        # truncated used to still be announced as K-independent, promising a column never printed.
        n_img, n_flat = int(r["test_per_seed"]), len(r["per_image"])
        if n_img <= 0 or n_flat == 0 or n_flat % n_img:
            print(f"  [stats] skipping malformed {fp}: {n_flat} per-image scores is not a whole "
                  f"number of seeds × test_per_seed={n_img} (crashed or truncated run)", flush=True)
            n_bad += 1
            continue

        if dir_k is None and support is not None:
            # A directory with no K claims to be support-blind. VERIFY it from the record's own
            # numbers instead of believing the directory: a support-blind run scores the fixed test
            # split identically under every seed. Without this, any K-dependent record dropped into
            # a `_k`-less directory joins every K family and gets paired p-values computed against it.
            blind = _looks_support_blind(r)
            if blind is None:
                raise SystemExit(
                    f"{fp} is in a directory with no _k suffix, so it claims to ignore the support, "
                    f"but it has fewer than two seeds and the claim cannot be checked. Refusing: a "
                    f"K-dependent record admitted here would be replicated into EVERY K table.")
            if not blind:
                raise SystemExit(
                    f"{fp} is in a directory with no _k suffix (so it would be reported at every K), "
                    f"but its per-image scores DIFFER across seeds — the support draw changed the "
                    f"result, so it is not support-blind. Move it to a '{label}_k<N>' directory.")
            k_free.add(label)
        elif support is not None and dir_k != support:
            # Filtering the family is right; discarding it unaccounted is not. `methods` is derived
            # from what survives, so a baseline with no directory at this K gets no column at all --
            # not even a dash -- and the reader cannot tell it was omitted. Under the campaign this
            # is the NORMAL case (Matcher runs at K in {1,4}, PerSAM at {1,8}), and a genuine
            # failure looks identical to it. Remember what we dropped and say so.
            other_k.setdefault(label, set()).add(dir_k)
            continue

        prev = data.setdefault(r["dataset"], {}).get(label)
        if prev is not None:
            # Two files claiming the same (dataset, label) silently overwrote each other — the
            # documented route by which a wrong number reaches the paper. Refuse rather than pick.
            hint = ("" if support is not None else
                    "\n  Most likely cause: this is a campaign tree holding several K per method, "
                    "and no K was selected.\n  Pass --support K to pick one family; a table mixing "
                    "K would print an 8-shot and a 16-shot\n  number in the same column with no "
                    "way for the reader to tell them apart.")
            raise SystemExit(f"duplicate record for ({r['dataset']}, {label}) in "
                             f"{args.score_dir}: {fp} collides with {origin[(r['dataset'], label)]}."
                             f"{hint}")
        data[r["dataset"]][label] = r
        origin[(r["dataset"], label)] = fp
    if not data:
        # Refuse rather than print an empty table. A run that scored nothing and a run whose every
        # record was rejected both produce a header with no rows underneath, which reads as "these
        # methods were compared and tied" to anyone skimming the output — and exits 0, so no
        # wrapper notices either. The causes need opposite fixes, so they are distinguished.
        if other_k:
            why = (f"{n_seen} record(s) exist but none at support={support}; the tree has "
                   + ", ".join(f"{m} at K={sorted(ks)}" for m, ks in sorted(other_k.items()))
                   + " — the PATH is fine, the K family is empty")
        elif n_bad:
            why = f"{n_bad} of {n_seen} record file(s) were rejected as malformed — see the lines above"
        else:
            why = "no record files were found at all — check the path"
        raise SystemExit(f"nothing to report from {args.score_dir}"
                         + (f" at support={support}" if support is not None else "")
                         + f": {why}.")
    scope = f" — K={support}" if support is not None else ""
    print(f"\n===== SOTA FINAL{scope} — mean±std over seeds (each dataset's designated metric) =====",
          flush=True)
    if n_bad:
        print(f"  !! {n_bad} of {n_seen} record file(s) were REJECTED as malformed; this table is "
              f"incomplete", flush=True)
    # Report ONLY methods that are genuinely absent from the requested family. Every record
    # filtered by K was recorded above, including records of methods that DO have a cell at this K
    # (a method run at K=1,4,8,16 contributes three filtered records), so reporting the raw set
    # announced methods as missing while their column was right there -- a warning that is wrong is
    # worse than none, because it teaches the reader to ignore the line.
    present = {m for ds in data.values() for m in ds}
    absent = {m: ks for m, ks in other_k.items() if m not in present}
    if absent:
        # Absent-at-this-K is stated, not left as a missing column. Otherwise a by-design absence
        # (Matcher has no K=8) and a real failure (microSAM-FT died at K=16) look identical.
        print("  not in this K family (no column below): "
              + ", ".join(f"{m} ran at K={sorted(ks)}" for m, ks in sorted(absent.items())),
              flush=True)
    if k_free:
        # Named explicitly: these columns are the SAME numbers in every K table, so a reader
        # comparing across K must know not to read them as a K-scaling trend.
        print(f"  K-independent (support ignored, verified identical across seeds): "
              f"{', '.join(sorted(k_free))}", flush=True)
    methods = sorted({m for ds in data.values() for m in ds})

    # Resolve --ours against BOTH the campaign job label ("ours") and the harness method name
    # ("head_fusion_best_cgate_film_nobank"), because both are natural to type and the campaign
    # directory carries the former while the record body carries the latter.
    ours = args.ours
    if ours not in methods:
        by_method = sorted({lb for ds in data.values() for lb, r in ds.items()
                            if r.get("method") == args.ours})
        if len(by_method) == 1:
            ours = by_method[0]
            print(f"  (--ours {args.ours!r} resolved to job label {ours!r})", flush=True)
        elif len(by_method) > 1:
            raise SystemExit(f"--ours {args.ours!r} is ambiguous: it is the harness method behind "
                             f"several job labels {by_method}. Pass the label you mean.")
        else:
            # REFUSE. This used to fall through and print the full significance banner, the column
            # header, no rows, and "Holm-Bonferroni over the 0 tests" — then exit 0. An empty
            # significance table reads as "compared and tied", and no wrapper notices a zero exit.
            # The default --ours is a stale method name, so this was the DEFAULT invocation's output.
            raise SystemExit(
                f"--ours {args.ours!r} matches no record: not a job label and not a method name in "
                f"this tree.\n  Available: {', '.join(methods)}\n  For the campaign tree pass "
                f"--ours ours (the job label) or the exact method string.")

    # A record whose dataset is not a registry key is dropped by every loop below, silently: the
    # "derived from the registry" guarantee only runs forward (no registry dataset is forgotten),
    # not backward (no record is orphaned). A renamed dataset orphans its records with no message.
    orphan = sorted(set(data) - set(PANEL_DATASETS))
    if orphan:
        print(f"  !! ORPHANED records for dataset(s) not in the registry, absent from every row "
              f"and every test: {', '.join(orphan)}", flush=True)

    # Column headers are truncated to 16 characters, and these method names differ in their TAILS:
    # `head_fusion_best_cgate_film` and `head_fusion_best_cgate_film_nobank` both render as
    # `head_fusion_best`. Two distinct methods under one header is a table nobody can read
    # correctly. Disambiguate ONLY the names that actually collide, so the common case keeps its
    # familiar plain truncation and downstream readers of this output are not disturbed.
    def _headers(names):
        trunc = {n: n[:16] for n in names}
        clash = {t for t in trunc.values() if list(trunc.values()).count(t) > 1}
        out = {}
        for n in names:
            out[n] = (n[:7] + ".." + n[-7:]) if trunc[n] in clash else trunc[n]
        return out

    alias = _headers(methods)
    renamed = {n: a for n, a in alias.items() if a != n[:16]}
    if renamed:
        print("  column aliases (names that collide when truncated): "
              + ", ".join(f"{a} = {n}" for n, a in renamed.items()), flush=True)
    print(f"{'dataset':>13} {'metric':>10} " + " ".join(f"{alias[m]:>16}" for m in methods))
    for ds in PANEL_DATASETS:
        if ds not in data:
            continue
        # The column header can only name ONE metric, so a row mixing metrics (easy to produce with
        # --metric_override) would silently present incomparable numbers side by side. Name the offenders.
        seen_metrics = {m: r.get("metric") for m, r in data[ds].items()}
        seen_splits = {r.get("split_fp") for r in data[ds].values() if r.get("split_fp")}
        row = f"{ds:>13} {next(iter(data[ds].values()))['metric']:>10} "
        for m in methods:
            if m in data[ds]:
                r = data[ds][m]
                arr = np.array(r["per_image"]).reshape(-1, r["test_per_seed"])
                row += f"{arr.mean(1).mean():>10.3f}±{arr.mean(1).std():<5.3f}"
            else:
                row += f"{'—':>16}"
        print(row, flush=True)
        if len(set(seen_metrics.values())) > 1:
            print(f"{'':>13} !! MIXED METRICS in this row — NOT comparable: {seen_metrics}", flush=True)
        if len(seen_splits) > 1:
            print(f"{'':>13} !! MIXED TEST SPLITS in this row — methods scored different images: "
                  f"{ {m: r.get('split_fp') for m, r in data[ds].items()} }", flush=True)

    print(f"\n===== PAIRED SIGNIFICANCE — {ours} vs each baseline =====", flush=True)
    print("Unit of analysis = ONE TEST IMAGE (scores averaged over seeds first). The seeds re-draw the\n"
          "support, not the test split, so seed repeats of an image are correlated measurements of the\n"
          "same unit; testing them as independent pairs would inflate every p-value.", flush=True)
    print(f"{'dataset':>13} {'baseline':>16} {'n_img':>6} {'Δmean':>8} {'wilcoxon_p':>11} "
          f"{'p_holm':>11} {'boot95CI':>20} {'rank-bis':>9}")
    _rows = []                       # buffered so Holm can adjust across the whole family before printing
    for ds in PANEL_DATASETS:
        if ds not in data:
            continue
        if ours not in data[ds]:
            # Named, not skipped: our own method missing on a dataset that HAS baselines means the
            # whole row is untested, which an absent row does not convey.
            print(f"{ds:>13} {'':>16}   NO ROWS — {ours!r} has no record on this dataset, so none "
                  f"of its {len(data[ds])} baseline(s) could be tested", flush=True)
            continue
        ours_rec = data[ds][ours]
        for m in methods:
            if m == ours or m not in data[ds]:
                continue
            base_rec = data[ds][m]
            problem = _pairing_problem(ours_rec, base_rec)
            if problem:
                print(f"{ds:>13} {m[:16]:>16}   SKIPPED — {problem}", flush=True)
                continue
            if not (ours_rec.get("split_fp") and base_rec.get("split_fp")):
                # REFUSE, do not warn-and-pair. This branch used to print a single advisory line and
                # then fall through to the Wilcoxon, so exactly the records the fingerprint exists to
                # catch — the ones predating it, whose test slice may genuinely differ — were paired,
                # bootstrapped and Holm-starred like verified ones. A missing fingerprint is a missing
                # guarantee, and a paper cannot cite a p-value whose two sides may score different
                # pictures. Re-run the offending record; do not relax this.
                missing = [n for n, r in ((ours, ours_rec), (m, base_rec)) if not r.get("split_fp")]
                print(f"{ds:>13} {m[:16]:>16}   SKIPPED — split identity UNVERIFIABLE "
                      f"(no split_fp on: {', '.join(missing)}); re-run that record", flush=True)
                continue
            a, b = _per_image_mean(ours_rec), _per_image_mean(base_rec)
            try:
                p = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
            except ValueError as e:
                # A test that could not be computed is still a comparison that was ATTEMPTED, and
                # it must not shrink the correction family: dropping it lowers n_tests and makes
                # every surviving p_holm SMALLER, i.e. it flatters us. Surfaced as nan, counted in
                # the family below, and named here so it is not mistaken for a missing row.
                print(f"{ds:>13} {m[:16]:>16}   wilcoxon could not be computed ({e}); counted in "
                      f"the Holm family but reported as nan", flush=True)
                p = float("nan")
            d = a - b
            lo, hi = _bootstrap_ci(d)
            # Buffered: Holm needs the whole family before any row can be printed.
            _rows.append((ds, m, len(a), float(d.mean()), p, lo, hi, _paired_rank_biserial(a, b)))

    # F15 — family-wise correction. Every (dataset x baseline) cell is one test; a table of ~30 raw
    # p-values will contain "significant" cells by chance alone. Holm-Bonferroni: step down over the
    # sorted p-values with multiplier (n_tests - rank), enforcing monotonicity, and clip at 1.
    finite = [i for i, r in enumerate(_rows) if np.isfinite(r[4])]
    p_adj = {i: float("nan") for i in range(len(_rows))}
    order = sorted(finite, key=lambda i: _rows[i][4])
    # The family size is EVERY comparison attempted, not only those that produced a number. Using
    # len(order) instead let an uncomputable test shrink the correction and make every surviving
    # adjusted p smaller, which is the anti-conservative direction and favours our own method.
    n_t = len(_rows)
    n_nan = n_t - len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n_t - rank) * _rows[i][4])   # monotone non-decreasing
        p_adj[i] = min(1.0, running)
    for i, (ds, m, n_a, dm, p, lo, hi, rb) in enumerate(_rows):
        star = "*" if np.isfinite(p_adj[i]) and p_adj[i] < 0.05 else " "
        print(f"{ds:>13} {alias.get(m, m)[:16]:>16} {n_a:>6} {dm:>+8.3f} {p:>11.2e} {p_adj[i]:>10.2e}{star} "
              f"{f'[{lo:+.3f},{hi:+.3f}]':>20} {rb:>+8.3f}", flush=True)
    print(f"\n  p_holm = Holm-Bonferroni over the {n_t} comparisons in this table"
          + (f" ({n_nan} of them uncomputable, counted in the family)" if n_nan else "")
          + "; * = adjusted p < 0.05.")
    print("  Report the ADJUSTED column: the raw one is descriptive only.")
    print("SOTA_STATS_DONE")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    r = sub.add_parser("run")
    r.add_argument("--method", required=True)
    r.add_argument("--datasets", default="")
    r.add_argument("--support", type=int, default=8)
    r.add_argument("--pool", type=int, default=20)
    r.add_argument("--test", type=int, default=24)
    r.add_argument("--seeds", type=int, default=6)
    r.add_argument("--res", type=int, default=672)
    r.add_argument("--superres", type=int, default=1,
                   help="training-free feature super-resolution factor (shift-merge); >1 densifies the "
                        "ViT patch grid (proven thin-structure lever). Enters the encoder cache_tag.")
    r.add_argument("--adaptive_superres", type=int, default=0,
                   help="if >0, pick superres per-dataset from support morphology: this factor by default, "
                        "1 (off) for dense-tiny-instance or huge-downscale support (where superres regresses).")
    r.add_argument("--sr_downscale", type=float, default=4.0,
                   help="mean(support image max-side)/res above which superres is gated OFF (huge downscale, "
                        "e.g. hrf 5.2×). Instance-metric datasets are gated OFF regardless.")
    r.add_argument("--adaptive_res", type=int, default=0,
                   help="if >0, this VALUE is min_obj_px: self-configure the ENCODER RESOLUTION from the K "
                        "support masks. RAISE res (up to --res_max) when the aggregate support foreground is "
                        "small AND the native image is large (small-objects-in-large-images regime — e.g. "
                        "PerSeg candle, fg tiny in a 2048px image: 4pct IoU @672 vs 92pct @1024); normal-scale "
                        "data stays at --res. 0 = OFF (byte-identical to fixed --res). When res is raised, "
                        "superres is forced to 1 (raising res already densifies the grid).")
    r.add_argument("--res_max", type=int, default=1536,
                   help="memory cap: adaptive_res never raises the encoder resolution above this.")
    r.add_argument("--layer", type=int, default=-1,
                   help="DINOv3 hidden-state layer for features (-1=last; ~12 = mid, better on filaments).")
    r.add_argument("--layers", default="",
                   help="LAYER-FUSION: comma-separated DINOv3 blocks to concat (e.g. '-1,12'); head learns the "
                        "per-layer weighting. Empty = single --layer. Enters the encoder cache_tag.")
    r.add_argument("--feat_upsampler", default="none", choices=["none", "anyup", "jafar"],
                   help="learned frozen-feature upsampler for the coarse grid (label-free alternative to "
                        "--superres; MUTUALLY EXCLUSIVE — forces superres off). Enters the cache_tag.")
    r.add_argument("--upsampler_factor", type=int, default=2,
                   help="densification factor for --feat_upsampler (2 = /8 grid, 4 = /4 'F4' rung).")
    r.add_argument("--refine", default="none")
    r.add_argument("--metric_override", default="",
                   help="score with this metric instead of each dataset's native one (e.g. fg_iou for a "
                        "semantic-only baseline like Matcher on instance-AP datasets); also disables the "
                        "instance-prediction path. Empty = use the dataset's designated metric.")
    r.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    r.add_argument("--cache", default="/disk1/prusek/asg_cache_panel")
    r.add_argument("--score_dir", default=SCORE_DIR_DEFAULT)
    s = sub.add_parser("stats")
    s.add_argument("--ours", default="head_fusion_uni_adapt")
    s.add_argument("--score_dir", default=SCORE_DIR_DEFAULT,
                   help="flat directory of score records, OR a campaign tree of per-(method, K) "
                        "subdirectories such as results/final10/. Both layouts are read.")
    s.add_argument("--support", type=int, default=None,
                   help="report only this K, read from the _k<N> subdirectory names. Required when "
                        "the tree holds several K per method; methods with no K subdirectory (the "
                        "off-the-shelf specialists, which ignore the support) appear in every K.")
    args = ap.parse_args()
    (run_inharness if args.stage == "run" else stats)(args)


if __name__ == "__main__":
    main()
