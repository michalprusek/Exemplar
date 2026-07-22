"""The score-record contract that admits an out-of-env baseline into the paired statistics.

``scripts/sota_final.py stats`` compares our method against a baseline by loading
``<score_dir>/<method>__<dataset>.json`` from BOTH sides and running a paired Wilcoxon over their
per-image scores. A baseline that only prints a mean to stdout can be regex-scraped into the
table but can never be paired, so its column carries no significance test at all — which is where
the PerSAM / PerSAM-F / micro-SAM / Cellpose / StarDist rows stood.

The writer lives here, once, instead of in each baseline script, for two reasons:

* ``split_fingerprint`` must agree BYTE-FOR-BYTE with ``sota_final``'s or every comparison is
  refused as DIFFERENT TEST IMAGES. Three independent copies of a hash are three chances to
  drift, and the symptom (a silently skipped comparison) is indistinguishable from a genuine
  protocol mismatch, so nobody would go looking for a hash bug.
* the SEED-MAJOR ordering of ``per_image`` cannot be checked by inspection downstream: ``stats``
  reshapes the flat vector to ``(n_seeds, test_per_seed)`` and averages down the seed axis, so an
  image-major vector has the identical length, reshapes without error, and produces a confident
  WRONG p-value. Taking the scores as a nested ``[seed][image]`` list and flattening them HERE
  makes the ordering structural — no caller is in a position to get it wrong.

Only ``json``/``os``/``hashlib``/numpy and the read-only registry are imported, so all three
baseline scripts can use it from their own environments (PerSAM's SAM fork, micro-SAM's mamba
env, Cellpose's torch env, StarDist's tensorflow env) through the same ``sys.path`` entry they
already add to reach ``active_segmenter.eval.registry`` and ``.scoring``.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from active_segmenter.eval.registry import PANEL
from active_segmenter.eval.scoring import PRIMARY

# Derived from the metric routing, never hand-maintained. ``metric`` in a record must be the
# PRIMARY KEY (``fg_iou``/``cldice``/``ap``) — the key ``score_prediction`` returns the number
# under — and NOT the DatasetSpec tag (``iou``/``instance_ap``) that selected it. The two are
# easy to confuse because both are called "metric", and getting it wrong does not raise: every
# comparison against our method is simply skipped with METRIC MISMATCH, leaving the baseline in
# the table with no test behind it, which is the exact defect this module exists to fix.
VALID_METRICS = frozenset(PRIMARY.values())


def split_fingerprint(pairs) -> str:
    """Content digest of a loaded split — VERBATIM the algorithm in ``scripts/sota_final.py``.

    Two runs can agree on image COUNT and still have scored different pictures: the flat-directory
    loader slices ``permutation(seed)[support : support+test]``, so the test slice moves when
    EITHER the seed or the support size changes. ``stats`` refuses to pair records that disagree
    on this digest, so it must be computed the same way on both sides — same hash, same field
    order, same truncation — or the baseline is silently dropped from the significance table.

    Compute it on the FINAL test list, i.e. after any ``[:n]`` truncation, since the digest is a
    claim about the images actually scored.
    """
    h = hashlib.sha256()
    for image, label in pairs:
        for a in (np.ascontiguousarray(image), np.ascontiguousarray(label)):
            h.update(str(a.shape).encode())
            h.update(str(a.dtype).encode())
            h.update(a.tobytes())
    return h.hexdigest()[:16]


def score_record_path(score_dir: str, method: str, dataset: str) -> str:
    """One file per (method, dataset) — the layout ``stats`` globs. Mirrors ``_score_path``."""
    return os.path.join(score_dir, f"{method}__{dataset}.json")


def write_score_record(score_dir: str, *, method: str, dataset: str, metric: str,
                       per_seed_images, seeds, split_fp: str, protocol: dict,
                       note: str | None = None) -> str:
    """Write one ``stats``-conforming score record and return its path.

    ``per_seed_images`` is a nested ``[seed][image]`` list — NOT a flat vector. Callers hand over
    the scores in the shape they naturally produced them (one inner list per support draw, in
    test-split order) and the seed-major flattening happens here, so the one ordering that
    ``stats`` cannot detect the violation of is decided in a single place.

    Every check below refuses rather than repairs: a malformed record that reaches the score dir
    is worse than no record, because the table prints it and the reader cannot tell.
    """
    if dataset not in PANEL:
        raise ValueError(f"dataset {dataset!r} is not a key of active_segmenter.eval.registry."
                         f"PANEL; stats() indexes its table by that key, so the record would be "
                         f"written and then never read. Known: {sorted(PANEL)}")
    if metric not in VALID_METRICS:
        hint = (f" — {metric!r} is a DatasetSpec.metric tag; pass primary_key({metric!r}) = "
                f"{PRIMARY[metric]!r}" if metric in PRIMARY else "")
        raise ValueError(f"metric {metric!r} is not one of {sorted(VALID_METRICS)}{hint}. "
                         f"stats() SKIPS every comparison whose two sides disagree on `metric`, "
                         f"so the wrong tag here costs the significance test silently.")
    seeds = [int(s) for s in seeds]
    rows = [[float(v) for v in row] for row in per_seed_images]
    if len(rows) != len(seeds):
        raise ValueError(f"{len(rows)} seed rows but {len(seeds)} seeds declared for "
                         f"{method}/{dataset}: a seed died mid-run and its scores are missing")
    if not rows or not rows[0]:
        raise ValueError(f"refusing to write {method}/{dataset} with no per-image scores")
    test_per_seed = len(rows[0])
    ragged = [seeds[i] for i, r in enumerate(rows) if len(r) != test_per_seed]
    if ragged:
        raise ValueError(f"seed(s) {ragged} scored a different number of images than seed "
                         f"{seeds[0]} ({test_per_seed}) for {method}/{dataset} — a partial seed "
                         f"cannot be reshaped, and padding it would fabricate scores")
    # SEED-MAJOR, the ordering ``_per_image_mean`` assumes: all of seed0's images in test-split
    # order, then all of seed1's, ... Interleaving by image instead yields a vector of identical
    # length that reshapes cleanly and averages the WRONG axis.
    per_image = [v for row in rows for v in row]
    # The invariant ``stats`` reshapes on, restated at the write boundary. The rectangularity
    # check above should make this unreachable; if it ever is reachable the record must not exist.
    if len(per_image) != len(seeds) * test_per_seed:
        raise ValueError(f"{len(per_image)} per-image scores != {len(seeds)} seeds x "
                         f"{test_per_seed} images for {method}/{dataset}")
    if not isinstance(split_fp, str) or not split_fp:
        raise ValueError(f"{method}/{dataset} needs a split_fp: stats() refuses to pair a record "
                         f"whose test-split identity is unverifiable, so an empty one is a "
                         f"guaranteed SKIP rather than a missing nicety")
    record = dict(method=method, dataset=dataset, metric=metric, test_per_seed=test_per_seed,
                  seeds=seeds, per_image=per_image, split_fp=split_fp, protocol=dict(protocol))
    if note:
        record["note"] = note
    os.makedirs(score_dir, exist_ok=True)
    path = score_record_path(score_dir, method, dataset)
    with open(path, "w") as f:
        json.dump(record, f)
    return path
