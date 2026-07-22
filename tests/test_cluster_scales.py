"""`cluster_scales` turns the support masks' object radii into the classical bank's filter scales, so a
silent change here rebuilds the bank at the wrong sigmas and the only symptom is a slightly different
number in an A/B. Two of these pin regressions that actually happened during development:

  * a `radius >= 1.0` pre-filter dropped every sub-pixel radius, which collapsed a thin-filament support
    to ONE COARSE scale (the fine structure the lever exists for vanished, with no error);
  * without a silhouette floor and a merge rule, a single-thickness morphology was split into several
    near-identical scales, quietly widening the bank and changing the A/B verdict.

The subsample test pins the latency guard: `silhouette_score` is O(n^2) and `_support_scales` pools every
medial-axis pixel of K masks (~10^5 radii on vessels), which is the native-resolution stall that got the
bank-unfreeze lever dropped (CLAUDE.md C13).
"""
from __future__ import annotations

import time

import numpy as np

from active_segmenter.segment.head_fusion_backend import _SCALE_SUBSAMPLE, cluster_scales


def _radii(mean, sd, n, seed=0):
    """n positive radii ~ N(mean, sd), clipped away from zero."""
    return np.clip(np.random.default_rng(seed).normal(mean, sd, n), 0.05, None).tolist()


def test_sub_pixel_radii_are_kept_so_a_thin_support_gets_a_fine_scale():
    """The regression: pre-filtering at `>= 1.0` returned a single ~8 px scale for a thin+thick mixture,
    silently discarding the fine scale the lever is for."""
    out = cluster_scales(_radii(0.6, 0.08, 300) + _radii(8.0, 0.6, 300, seed=1))
    assert len(out) == 2, f"thin+thick mixture must yield two scales, got {out}"
    assert min(out) < 1.0, (f"the FINE scale was dropped ({out}) — sub-pixel radii must survive the "
                            f"filter; the caller clamps centroids up to 1.5 afterwards")
    assert max(out) > 5.0, f"the coarse scale is missing from {out}"


def test_unimodal_morphology_is_not_over_split():
    """One thickness must give ONE scale: over-splitting silently widens the bank and changes the A/B."""
    out = cluster_scales(_radii(5.0, 0.5, 400))
    assert len(out) == 1, f"a single-thickness support must yield one scale, got {out}"
    assert 4.0 < out[0] < 6.0, f"the single scale should sit near the geometric mean, got {out}"


def test_genuinely_bimodal_morphology_splits():
    out = cluster_scales(_radii(2.0, 0.15, 300) + _radii(20.0, 1.0, 300, seed=1))
    assert len(out) == 2, f"well-separated thicknesses must yield two scales, got {out}"
    assert out[0] < 5 < out[1], f"the two centroids should bracket the gap, got {out}"


def test_centroids_closer_than_sqrt2_are_merged():
    """1.4x ~= half an octave: below that two Frangi responses overlap, so they are one scale."""
    merged = cluster_scales(_radii(4.0, 0.05, 300) + _radii(5.0, 0.05, 300, seed=1))
    assert len(merged) == 1, f"a 1.25x pair is one scale split by noise, got {merged}"
    assert 4.2 < merged[0] < 4.8, f"merged centroid should be the geometric mean ~4.47, got {merged}"


def test_clearly_separated_pair_is_not_merged():
    kept = cluster_scales(_radii(4.0, 0.05, 300) + _radii(8.0, 0.05, 300, seed=1))
    assert len(kept) == 2, f"a 2x pair is two distinct scales, got {kept}"


def test_empty_input_returns_a_fine_default_not_a_coarse_one():
    assert cluster_scales([]) == [1.0]
    assert cluster_scales([0.01, 0.02]) == [1.0], "all-below-filter must fall back to the FINE default"


def test_large_input_is_subsampled_deterministically_and_stays_fast():
    """O(n^2) silhouette on ~10^5 pooled radii is the C13 latency trap; the subsample must bound it and
    must not make the derived scales vary between runs."""
    big = _radii(3.0, 0.3, 60_000) + _radii(15.0, 1.2, 60_000, seed=1)
    assert len(big) > _SCALE_SUBSAMPLE
    t0 = time.perf_counter()
    a = cluster_scales(big)
    elapsed = time.perf_counter() - t0
    b = cluster_scales(big)
    assert a == b, f"subsampling must be deterministic across calls: {a} vs {b}"
    assert elapsed < 30, f"clustering {len(big)} radii took {elapsed:.1f}s — the subsample guard is gone"
    assert len(a) == 2, f"the bimodal structure must survive subsampling, got {a}"
