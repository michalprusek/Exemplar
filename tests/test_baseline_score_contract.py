"""The three external-env baselines must emit records that ``sota_final.stats`` PAIRS, not skips.

PerSAM/PerSAM-F, micro-SAM and Cellpose/StarDist run in their own environments and used to report
only a mean to stdout, so six baseline columns in the paper's table had no significance test behind
them. Each now writes a score record. Two ways of writing one are wrong WITHOUT RAISING ANYTHING,
which is what these tests exist to catch:

* ``metric`` holding the DatasetSpec tag (``instance_ap``) instead of the primary key (``ap``) —
  ``stats`` then skips every comparison with METRIC MISMATCH and the column stays untested;
* ``per_image`` ordered image-major instead of seed-major — identical length, reshapes without
  error, averages the wrong axis, and prints a confident WRONG p-value.

A positive test ("the record is paired") cannot catch either on its own, so each is paired with a
mutation showing the assertion actually discriminates against the wrong version.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import numpy as np
import pytest

from active_segmenter.eval.score_record import split_fingerprint, write_score_record

_SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    """Import a benchmark script by path — ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sota_final = _load("sota_final")

# Two seeds x three images, all six values distinct, so seed-major and image-major orderings are
# DISTINGUISHABLE by value (they are always indistinguishable by length or shape).
SEED_ROWS = [[0.10, 0.20, 0.30], [0.40, 0.50, 0.60]]
SEED_ROWS_MEAN = [0.25, 0.35, 0.45]              # mean down the seed axis
SEEDS = [0, 1]
SPLIT_FP = "0123456789abcdef"


def _write_ours(tmp_path, dataset, metric="ap"):
    """Our method's record for the same dataset — the other half of every pairing."""
    write_score_record(str(tmp_path), method="ours", dataset=dataset, metric=metric,
                       per_seed_images=[[0.9, 0.9, 0.9], [0.8, 0.8, 0.8]], seeds=SEEDS,
                       split_fp=SPLIT_FP,
                       protocol=dict(pool=20, test=3, support=8, split_seed=0))


def _stats_out(tmp_path, capsys):
    sota_final.stats(argparse.Namespace(score_dir=str(tmp_path), ours="ours"))
    return capsys.readouterr().out


def _paired_row(out, dataset, method):
    """The printed significance row for exactly one comparison, or None if skipped/absent."""
    for line in out.splitlines():
        parts = line.split()
        if len(parts) > 2 and parts[0] == dataset and parts[1] == method[:16] \
                and "SKIPPED" not in line and parts[2].isdigit():
            return parts
    return None


def _assert_paired(tmp_path, capsys, dataset, method, expect_per_image):
    """The record entered the paired statistics AND was read back in seed-major order."""
    out = _stats_out(tmp_path, capsys)
    assert "SKIPPED" not in out, f"stats refused to pair {method}:\n{out}"
    row = _paired_row(out, dataset, method)
    assert row is not None, f"no paired row for {method} in:\n{out}"
    # n_img is the image count, not seeds x images — proof the seed axis was collapsed, not
    # flattened into extra "independent" pairs.
    assert row[2] == "3", f"expected n_img=3, got: {row}"
    rec = json.loads((tmp_path / f"{method}__{dataset}.json").read_text())
    # The decisive ordering check. An image-major vector has the same length and reshapes cleanly,
    # so only the VALUES distinguish it (see test_image_major_is_a_silent_corruption).
    np.testing.assert_allclose(sota_final._per_image_mean(rec), expect_per_image)
    return rec


# ------------------------------------------------------------------- 1. PerSAM / PerSAM-F
def test_persam_records_are_paired(tmp_path, capsys):
    persam_bench = _load("persam_bench")
    args = argparse.Namespace(score_dir=str(tmp_path), seeds=SEEDS, pool=20, test=3, support=8,
                              sam_type="vit_h", fg_scoring=False)
    persam_bench.write_records(args, "dsb2018", "ap",
                               {"persam": SEED_ROWS, "persam_f": SEED_ROWS},
                               SPLIT_FP, write_score_record)
    _write_ours(tmp_path, "dsb2018")
    rec = _assert_paired(tmp_path, capsys, "dsb2018", "persam", SEED_ROWS_MEAN)
    assert rec["metric"] == "ap"
    assert rec["seeds"] == SEEDS
    assert (tmp_path / "persam_f__dsb2018.json").exists(), "each variant needs its own file"


def test_persam_omits_the_variant_that_did_not_run(tmp_path):
    """``--mode persam`` leaves persam_f empty; an empty record is worse than no record."""
    persam_bench = _load("persam_bench")
    args = argparse.Namespace(score_dir=str(tmp_path), seeds=SEEDS, pool=20, test=3, support=8,
                              sam_type="vit_h", fg_scoring=False)
    persam_bench.write_records(args, "dsb2018", "ap", {"persam": SEED_ROWS, "persam_f": []},
                               SPLIT_FP, write_score_record)
    assert not (tmp_path / "persam_f__dsb2018.json").exists()


# ------------------------------------------------------------------- 2. micro-SAM
def test_microsam_record_is_paired(tmp_path, capsys):
    """One deterministic per-image vector, tiled seed-major; still ordering-sensitive.

    With vals=[.1,.2,.3] over 2 seeds, seed-major is [.1,.2,.3,.1,.2,.3] and recovers [.1,.2,.3];
    image-major would be [.1,.1,.2,.2,.3,.3] and recovers [.15,.2,.25]. Tiling does NOT make the
    ordering moot.
    """
    microsam_bench = _load("microsam_bench")
    args = argparse.Namespace(score_dir=str(tmp_path), model="vit_b_lm", pool=20, test=3,
                              support=8, tile=0, fg_scoring=False)
    microsam_bench.write_record(args, "dsb2018", "ap", [0.1, 0.2, 0.3], SEEDS, SPLIT_FP)
    _write_ours(tmp_path, "dsb2018")
    rec = _assert_paired(tmp_path, capsys, "dsb2018", "microsam_vit_b_lm", [0.1, 0.2, 0.3])
    assert rec["per_image"] == [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]
    assert "Deterministic" in rec["note"], "the seed tiling must be RECORDED, not implied"
    arr = np.asarray(rec["per_image"]).reshape(-1, rec["test_per_seed"])
    assert arr.mean(1).std() == 0.0, "a support-blind model legitimately reports std=0"


def test_microsam_model_variants_do_not_collide(tmp_path):
    """stats() REFUSES a duplicate (dataset, method), so vit_b_lm and vit_l_lm must not share one."""
    microsam_bench = _load("microsam_bench")
    for model in ("vit_b_lm", "vit_l_lm"):
        args = argparse.Namespace(score_dir=str(tmp_path), model=model, pool=20, test=3,
                                  support=8, tile=0, fg_scoring=False)
        microsam_bench.write_record(args, "dsb2018", "ap", [0.1, 0.2, 0.3], SEEDS, SPLIT_FP)
    assert sorted(p.name for p in tmp_path.glob("*.json")) == [
        "microsam_vit_b_lm__dsb2018.json", "microsam_vit_l_lm__dsb2018.json"]


def test_microsam_panel_keys_cover_the_whole_registry():
    """The stale hand-written list benchmarked 6 of 15 datasets under `--datasets all`."""
    from active_segmenter.eval.registry import PANEL

    microsam_bench = _load("microsam_bench")
    assert set(microsam_bench.PANEL_KEYS) == set(PANEL)
    assert len(microsam_bench.PANEL_KEYS) == len(PANEL), "no duplicates"
    assert microsam_bench.PANEL_KEYS[:2] == ["spheroid", "spheroidj"], "report order preserved"


# ------------------------------------------------------------------- 3. Cellpose / StarDist
def test_cellpose_record_is_paired(tmp_path, capsys):
    csb = _load("cellpose_stardist_bench")
    per_seed = [{"per_image": row} for row in SEED_ROWS]      # _score_split's return shape
    csb.write_record(str(tmp_path), "cellpose_cpsam", "dsb2018", "ap", per_seed, SEEDS, SPLIT_FP,
                     dict(pool=20, test=3, support=8, split_seed=0, backend="cellpose_cpsam"))
    _write_ours(tmp_path, "dsb2018")
    rec = _assert_paired(tmp_path, capsys, "dsb2018", "cellpose_cpsam", SEED_ROWS_MEAN)
    # `ap`, not the DatasetSpec tag `instance_ap` the aggregate --out row carries.
    assert rec["metric"] == "ap"


def test_cellpose_score_split_keeps_the_out_schema_and_adds_per_image(tmp_path):
    """--out's aggregate is read by other tooling; the per-image list is purely additive."""
    from active_segmenter.eval.registry import PANEL

    csb = _load("cellpose_stardist_bench")
    gt = np.zeros((8, 8), np.uint8)
    gt[2:5, 2:5] = 1
    pairs = [(gt * 255, gt), (gt * 255, gt)]
    assert PANEL["spheroid"].metric == "iou", "the effective metric passed below is this dataset's"
    out = csb._score_split("iou", "fg_iou", lambda img: np.asarray(gt, int), pairs, "x")
    assert set(out) >= {"primary", "fg_iou", "bf", "ap50", "n", "sec_per_img"}, "--out schema kept"
    assert len(out["per_image"]) == 2
    assert out["primary"] == pytest.approx(float(np.mean(out["per_image"])))


# ---------------------------------------- 4. --fg-scoring, the campaign's metric convention
# An INSTANCE dataset: natively `ap`, so it is the only case where the convention changes anything
# and the only one where a fg_iou-vs-ap mismatch can arise.
FG_DATASET = "dsb2018"


def _fg_pk():
    """What the three scripts derive under --fg-scoring, via the expression they all share."""
    from active_segmenter.eval.registry import PANEL
    from active_segmenter.eval.scoring import effective_metric, primary_key

    assert PANEL[FG_DATASET].metric == "instance_ap", \
        "these tests are vacuous unless the fixture dataset is natively instance-scored"
    return primary_key(effective_metric(PANEL[FG_DATASET].metric, True))


def _write_fg_baseline(tmp_path, which, fg_scoring=True):
    """Write one baseline record through the real script writer; return its method name.

    The writers are driven with the `pk` their own scripts compute, so these tests exercise the
    metric the scripts would actually record rather than a value restated here.
    """
    pk = _fg_pk() if fg_scoring else "ap"
    if which == "persam":
        mod = _load("persam_bench")
        args = argparse.Namespace(score_dir=str(tmp_path), seeds=SEEDS, pool=20, test=3,
                                  support=8, sam_type="vit_h", fg_scoring=fg_scoring)
        mod.write_records(args, FG_DATASET, pk, {"persam": SEED_ROWS, "persam_f": []},
                          SPLIT_FP, write_score_record)
        return "persam"
    if which == "microsam":
        mod = _load("microsam_bench")
        args = argparse.Namespace(score_dir=str(tmp_path), model="vit_b_lm", pool=20, test=3,
                                  support=8, tile=0, fg_scoring=fg_scoring)
        mod.write_record(args, FG_DATASET, pk, [0.1, 0.2, 0.3], SEEDS, SPLIT_FP)
        return "microsam_vit_b_lm"
    mod = _load("cellpose_stardist_bench")
    mod.write_record(str(tmp_path), "cellpose_cpsam", FG_DATASET, pk,
                     [{"per_image": row} for row in SEED_ROWS], SEEDS, SPLIT_FP,
                     dict(pool=20, test=3, support=8, split_seed=0, backend="cellpose_cpsam",
                          fg_scoring=fg_scoring))
    return "cellpose_cpsam"


def test_fg_scoring_convention_is_the_one_specialist_finetune_bench_uses():
    """Pinned against the reference literal in ``scripts/specialist_finetune_bench.py`` main().

    Every dataset in the registry, because a convention that only agrees on the datasets someone
    happened to test is exactly how four clDice columns would quietly diverge from six others.
    """
    from active_segmenter.eval.registry import PANEL
    from active_segmenter.eval.scoring import effective_metric

    for name, spec in PANEL.items():
        ref = spec.metric if spec.metric == "cldice" else "fg_iou"
        assert effective_metric(spec.metric, True) == ref, name
        assert effective_metric(spec.metric, False) == spec.metric, f"{name}: default must not move"


def test_fg_score_of_a_label_map_is_a_selection_not_a_new_computation():
    """micro-SAM and Cellpose/StarDist score instance LABEL MAPS.

    --fg-scoring must read out a number those runs already computed, not measure something else:
    ``score_prediction`` returns ``fg_iou`` under every metric, and ``foreground_iou`` binarises the
    GT itself, so an instance label map needs no conversion. If this ever stops holding, the
    foreground columns silently stop being the foreground of the same prediction.
    """
    from active_segmenter.eval.scoring import score_prediction

    gt = np.zeros((10, 10), np.uint16)
    gt[1:5, 1:5], gt[6:9, 6:9] = 1, 2                 # two GT instances, NOT a binary mask
    pred = np.zeros((10, 10), np.uint16)
    pred[2:5, 1:5], pred[6:9, 6:8] = 7, 9             # deliberately imperfect, so IoU != 0 and != 1
    fg = pred > 0
    inst = [pred == i for i in (7, 9)]
    native = score_prediction("instance_ap", fg, gt, inst)
    fg_only = score_prediction("fg_iou", fg, gt, None)
    assert 0.0 < native["fg_iou"] < 1.0, "a trivial 0/1 fixture would make the equality vacuous"
    assert fg_only["fg_iou"] == native["fg_iou"]


@pytest.mark.parametrize("which,expect", [("persam", SEED_ROWS_MEAN),
                                          ("microsam", [0.1, 0.2, 0.3]),
                                          ("cellpose", SEED_ROWS_MEAN)])
def test_fg_scored_instance_record_pairs_with_a_fg_scored_ours(tmp_path, capsys, which, expect):
    """The whole point: on an instance dataset the record must carry `fg_iou` AND survive stats()."""
    method = _write_fg_baseline(tmp_path, which)
    rec = json.loads((tmp_path / f"{method}__{FG_DATASET}.json").read_text())
    assert rec["metric"] == "fg_iou", \
        f"{which} recorded {rec['metric']!r} under --fg-scoring; stats() pairs on this field"
    _write_ours(tmp_path, FG_DATASET, metric="fg_iou")
    _assert_paired(tmp_path, capsys, FG_DATASET, method, expect)


@pytest.mark.parametrize("which", ["persam", "microsam", "cellpose"])
def test_baseline_without_fg_scoring_is_dropped_by_a_fg_scored_campaign(tmp_path, capsys, which):
    """The defect the flag closes, shown to be real rather than hypothetical.

    Left at its default these scripts write the dataset's own `ap` while the campaign scores our
    method `fg_iou`. Nothing raises: the comparison is skipped, and the paper keeps a baseline
    column with no significance test behind it — indistinguishable from a genuine protocol clash.
    """
    method = _write_fg_baseline(tmp_path, which, fg_scoring=False)
    rec = json.loads((tmp_path / f"{method}__{FG_DATASET}.json").read_text())
    assert rec["metric"] == "ap", "default behaviour must not have moved"
    _write_ours(tmp_path, FG_DATASET, metric="fg_iou")
    out = _stats_out(tmp_path, capsys)
    assert "METRIC MISMATCH" in out
    assert _paired_row(out, FG_DATASET, method) is None


def test_cellpose_score_split_under_fg_scoring_reports_foreground_and_no_ap50():
    """`--out`'s row must describe what it measured: a foreground run has no AP@0.5 to report."""
    from active_segmenter.eval.registry import PANEL

    csb = _load("cellpose_stardist_bench")
    gt = np.zeros((10, 10), np.uint16)
    gt[1:5, 1:5], gt[6:9, 6:9] = 1, 2
    pred = np.zeros((10, 10), np.uint16)
    pred[2:5, 1:5], pred[6:9, 6:8] = 1, 2
    pairs = [(pred * 100, gt)]
    assert PANEL[FG_DATASET].metric == "instance_ap"
    native = csb._score_split("instance_ap", "ap", lambda img: pred, pairs, FG_DATASET)
    fg = csb._score_split("fg_iou", "fg_iou", lambda img: pred, pairs, FG_DATASET)
    assert fg["primary"] == pytest.approx(native["fg_iou"]), "same foreground, read out directly"
    assert fg["ap50"] is None, "a foreground run reports no AP@0.5"
    assert native["ap50"] is not None, "the native instance path still does"


# ------------------------------------------------- the two silent corruptions, and the guards
def test_spec_metric_instead_of_primary_key_is_refused(tmp_path):
    """`instance_ap` is the DatasetSpec tag; a record must carry `ap`."""
    with pytest.raises(ValueError, match="primary_key"):
        write_score_record(str(tmp_path), method="persam", dataset="dsb2018", metric="instance_ap",
                           per_seed_images=SEED_ROWS, seeds=SEEDS, split_fp=SPLIT_FP, protocol={})
    assert not list(tmp_path.glob("*.json")), "a refused record must not reach the score dir"


def test_metric_mismatch_loses_the_comparison_silently(tmp_path, capsys):
    """What the guard above prevents: no exception, no row — just an untested baseline column."""
    path = write_score_record(str(tmp_path), method="persam", dataset="dsb2018", metric="ap",
                              per_seed_images=SEED_ROWS, seeds=SEEDS, split_fp=SPLIT_FP,
                              protocol={})
    rec = json.loads(pathlib.Path(path).read_text())
    rec["metric"] = "instance_ap"                      # what a hand-rolled writer would emit
    pathlib.Path(path).write_text(json.dumps(rec))
    _write_ours(tmp_path, "dsb2018")
    out = _stats_out(tmp_path, capsys)
    assert "METRIC MISMATCH" in out
    assert _paired_row(out, "dsb2018", "persam") is None


def test_image_major_is_a_silent_corruption(tmp_path):
    """Guards the ordering assertion in _assert_paired: it must discriminate, not merely pass."""
    seed_major = [v for row in SEED_ROWS for v in row]
    image_major = [v for col in zip(*SEED_ROWS) for v in col]
    assert len(seed_major) == len(image_major), "why nothing downstream can notice the swap"
    assert not np.allclose(
        sota_final._per_image_mean(dict(per_image=image_major, test_per_seed=3)), SEED_ROWS_MEAN), \
        "the fixture must distinguish the orderings or every ordering assertion is vacuous"
    np.testing.assert_allclose(
        sota_final._per_image_mean(dict(per_image=seed_major, test_per_seed=3)), SEED_ROWS_MEAN)


def test_ragged_seed_rows_are_refused(tmp_path):
    """A seed that died mid-split cannot be reshaped, and padding it would fabricate scores."""
    with pytest.raises(ValueError, match="different number of images"):
        write_score_record(str(tmp_path), method="persam", dataset="dsb2018", metric="ap",
                           per_seed_images=[[0.1, 0.2, 0.3], [0.4, 0.5]], seeds=SEEDS,
                           split_fp=SPLIT_FP, protocol={})


def test_seed_count_mismatch_is_refused(tmp_path):
    with pytest.raises(ValueError, match="seed rows"):
        write_score_record(str(tmp_path), method="persam", dataset="dsb2018", metric="ap",
                           per_seed_images=SEED_ROWS, seeds=[0, 1, 2], split_fp=SPLIT_FP,
                           protocol={})


def test_missing_split_fp_is_refused_at_write_time(tmp_path):
    """stats() would skip such a record; better never to produce one than to debug the skip."""
    with pytest.raises(ValueError, match="split_fp"):
        write_score_record(str(tmp_path), method="persam", dataset="dsb2018", metric="ap",
                           per_seed_images=SEED_ROWS, seeds=SEEDS, split_fp="", protocol={})


def test_unknown_dataset_is_refused(tmp_path):
    with pytest.raises(ValueError, match="PANEL"):
        write_score_record(str(tmp_path), method="persam", dataset="not_a_dataset", metric="ap",
                           per_seed_images=SEED_ROWS, seeds=SEEDS, split_fp=SPLIT_FP, protocol={})


def test_split_fingerprint_matches_sota_final_byte_for_byte():
    """The shared copy must agree with the harness's, or every comparison is refused as DIFFERENT
    TEST IMAGES — a symptom indistinguishable from a genuine protocol error."""
    rng = np.random.default_rng(0)
    pairs = [(rng.integers(0, 255, (7, 5, 3), dtype=np.uint8),
              rng.integers(0, 3, (7, 5), dtype=np.uint16)) for _ in range(3)]
    assert split_fingerprint(pairs) == sota_final.split_fingerprint(pairs)
    assert split_fingerprint(pairs[:2]) != split_fingerprint(pairs), "truncation must change it"
