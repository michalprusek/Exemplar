"""Task-appropriate metric routing.

User directive: *every benchmark is evaluated with the metric designed for it*. A dataset's
:class:`~active_segmenter.eval.registry.DatasetSpec` carries a ``metric`` tag; this module maps
a tag to the actual scoring of a backend prediction on one image, so the panel / testbed never
hard-codes fg-IoU for a dataset whose structure calls for instance-AP or clDice.

- ``iou``          — semantic foreground overlap (blobs: spheroids, polyps).
- ``cldice``       — centerline Dice for tubular/thin structures (vessels, filaments).
- ``instance_ap``  — AP@[.5:.95] over per-instance masks (nuclei with per-instance GT).
"""
from __future__ import annotations

import numpy as np

from active_segmenter.eval import metrics

PRIMARY = {"iou": "fg_iou", "cldice": "cldice", "instance_ap": "ap"}


def score_prediction(metric: str, fg_mask, label_map, instances=None) -> dict:
    """Score one image. ``fg_mask`` = native bool foreground; ``label_map`` = GT (binary or
    per-instance); ``instances`` = optional list of predicted per-instance bool masks (only
    the ``instance_ap`` metric needs them). Returns a dict of metric-name -> value; the
    dataset's primary metric key is ``PRIMARY[metric]``."""
    gt_bin = np.asarray(label_map) > 0
    out = {"fg_iou": metrics.foreground_iou(fg_mask, label_map),
           "bf": metrics.boundary_f1(fg_mask, gt_bin)}
    if metric == "cldice":
        out["cldice"] = metrics.cldice(fg_mask, gt_bin)
    elif metric == "instance_ap":
        preds = instances if instances is not None else []
        out["ap"] = metrics.instance_ap(preds, gt_labels=label_map)["ap"] if preds else 0.0
    return out


def primary_key(metric: str) -> str:
    return PRIMARY.get(metric, "fg_iou")


def effective_metric(metric: str, fg_scoring: bool) -> str:
    """The metric a FOREGROUND-CONSISTENT campaign actually scores ``metric`` with.

    A cross-method table row can name only ONE metric, so the campaign scores every method the same
    way: clDice datasets keep clDice (foreground IoU on a one-pixel-wide vessel measures almost
    nothing), everything else is scored on the foreground so that a semantic-only baseline is not
    zeroed by instance-AP on touching objects.

    The expression lives here, once, rather than in each baseline script because ``sota_final.stats``
    pairs two records only when their ``metric`` FIELDS agree and SKIPS them otherwise WITHOUT
    raising. A script that computes this convention even slightly differently therefore does not
    produce a slightly wrong number — it produces a column with no significance test behind it, and
    nothing in the output distinguishes that from a genuine protocol mismatch.
    """
    return metric if metric == "cldice" else ("fg_iou" if fg_scoring else metric)
