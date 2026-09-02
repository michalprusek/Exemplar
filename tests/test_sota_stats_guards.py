"""Guards on the statistics stage of ``scripts/sota_final.py``.

Every test here corresponds to a defect that reached, or nearly reached, the paper:

* the module raising ``NameError`` at import time, so no benchmark could run at all;
* records with no ``split_fp`` being warned about and then paired anyway, which reinstates the
  cross-protocol pairing bug the fingerprint was added to prevent;
* a truncated ``per_image`` vector killing the whole stage with a bare traceback that never named
  the offending file;
* two files claiming the same (dataset, method) silently overwriting one another.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "sota_final", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "sota_final.py")
sota_final = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sota_final)          # importing at all is the first regression test


def _record(method, dataset, per_image, test_per_seed, split_fp="abc123", metric="fg_iou"):
    return dict(method=method, dataset=dataset, metric=metric, test_per_seed=test_per_seed,
                seeds=list(range(len(per_image) // test_per_seed)), per_image=per_image,
                split_fp=split_fp)


def _write(tmp_path, rec):
    fp = tmp_path / f"{rec['method']}__{rec['dataset']}.json"
    fp.write_text(json.dumps(rec))
    return fp


def _run_stats(tmp_path, ours="ours"):
    return sota_final.stats(argparse.Namespace(score_dir=str(tmp_path), ours=ours))


def test_module_imports_and_derives_the_panel():
    """PANEL was imported inside one function while used at module scope -> NameError on import."""
    assert sota_final.PANEL_DATASETS, "panel must be derived from the registry, not empty"
    assert "dsb2018" in sota_final.PANEL_DATASETS


def test_missing_split_fp_is_skipped_not_paired(tmp_path, capsys):
    _write(tmp_path, _record("ours", "dsb2018", [0.5, 0.6, 0.7, 0.8], 2))
    _write(tmp_path, _record("rival", "dsb2018", [0.1, 0.2, 0.3, 0.4], 2, split_fp=None))
    _run_stats(tmp_path)
    out = capsys.readouterr().out
    assert "SKIPPED — split identity UNVERIFIABLE" in out
    assert "rival" in out
    # the decisive assertion: no test was performed, so nothing entered the Holm family. An empty
    # family is the CORRECT outcome here (the skip is legitimate and is printed loudly right above),
    # unlike the --ours-matches-nothing case, which now refuses outright.
    assert "Holm-Bonferroni over the 0 comparisons" in out


def test_conflicting_split_fp_is_skipped(tmp_path, capsys):
    _write(tmp_path, _record("ours", "dsb2018", [0.5, 0.6], 2, split_fp="aaa"))
    _write(tmp_path, _record("rival", "dsb2018", [0.1, 0.2], 2, split_fp="bbb"))
    _run_stats(tmp_path)
    assert "DIFFERENT TEST IMAGES" in capsys.readouterr().out


def test_truncated_per_image_names_the_file_instead_of_crashing(tmp_path, capsys):
    """The defect guarded here is a bare traceback from deep in the table loop that named nothing.

    The file must identify itself. Since this record is the only one present, dropping it leaves
    nothing to report, and the stage then exits deliberately rather than printing a header with no
    rows under it -- an empty table exiting 0 reads as "compared and tied". Both halves are
    asserted: the skip names the file, and the exit says why there is nothing left.
    """
    _write(tmp_path, _record("ours", "dsb2018", [0.5, 0.6, 0.7], 2))   # 3 is not a multiple of 2
    with pytest.raises(SystemExit, match="rejected as malformed"):
        _run_stats(tmp_path)
    out = capsys.readouterr().out
    assert "skipping malformed" in out
    assert "ours__dsb2018.json" in out, "the message must name the offending file"


def test_one_bad_record_among_good_ones_does_not_stop_the_report(tmp_path, capsys):
    """The complement of the test above: a rejected record must not take the whole table with it."""
    _write(tmp_path, _record("ours", "dsb2018", [0.8, 0.8], 2))
    _write(tmp_path, _record("rival", "dsb2018", [0.2, 0.2], 2))
    (tmp_path / "broken__x.json").write_text(
        json.dumps(_record("broken", "dsb2018", [0.5, 0.6, 0.7], 2)))
    _run_stats(tmp_path)                                               # must not raise
    out = capsys.readouterr().out
    assert "skipping malformed" in out
    assert "SOTA FINAL" in out


def test_duplicate_dataset_method_refuses_rather_than_picking_a_winner(tmp_path):
    _write(tmp_path, _record("ours", "dsb2018", [0.5, 0.6], 2))
    # a second file, different name, same (dataset, method) in the body — the silent-overwrite route
    (tmp_path / "zz_stale_copy.json").write_text(
        json.dumps(_record("ours", "dsb2018", [0.9, 0.9], 2)))
    with pytest.raises(SystemExit, match="duplicate record"):
        _run_stats(tmp_path)


def test_seeds_are_collapsed_before_the_paired_test(tmp_path, capsys):
    """The unit of analysis is the IMAGE, not the (seed, image) pair.

    Two seeds over two images must report n_img=2. Reporting 4 is the pseudoreplication that
    inflated every p-value in the previous draft by roughly two orders of magnitude.
    """
    _write(tmp_path, _record("ours", "dsb2018", [0.5, 0.6, 0.5, 0.6], 2))
    _write(tmp_path, _record("rival", "dsb2018", [0.1, 0.2, 0.1, 0.2], 2))
    _run_stats(tmp_path)
    row = [ln for ln in capsys.readouterr().out.splitlines()
           if "dsb2018" in ln and "rival" in ln and "SKIPPED" not in ln]
    assert row, "expected one paired row"
    assert row[0].split()[2] == "2", f"n_img must be the image count, got: {row[0]}"


# ---------------------------------------------------------------------------------------------
# Campaign-tree discovery.
#
# These fixtures reproduce the layout `run_campaign.py` ACTUALLY writes. The first version of these
# tests used synthetic trees with one method per directory, which never reproduced the campaign's
# two-labels-one-method case, so the tests passed while `stats()` aborted on the real tree. Build
# what the campaign builds, not what is convenient.
# ---------------------------------------------------------------------------------------------

def _write_in(tmp_path, subdir, rec, support=None):
    d = tmp_path / subdir
    d.mkdir(parents=True, exist_ok=True)
    if support is not None:
        rec = dict(rec, protocol=dict(pool=20, test=10, support=support, split_seed=0))
    fp = d / f"{rec['method']}__{rec['dataset']}.json"
    fp.write_text(json.dumps(rec))
    return fp


def _run_tree(tmp_path, ours="ours", support=None):
    return sota_final.stats(
        argparse.Namespace(score_dir=str(tmp_path), ours=ours, support=support))


def test_two_job_labels_sharing_one_method_do_not_collide(tmp_path, capsys):
    """run_campaign maps insid3_guided AND insid3_dense to `--method insid3`.

    Both records therefore carry an identical (dataset, method), and keying the table on the record
    body made every INSID3 pair collide -- `stats()` aborted on the real campaign tree, i.e. it
    never read the tree it was written to read. Identity must come from the directory LABEL.
    """
    _write_in(tmp_path, "ours_k8", _record("head_fusion_best", "dsb2018", [0.8, 0.8], 2), support=8)
    _write_in(tmp_path, "insid3_guided_k8", _record("insid3", "dsb2018", [0.3, 0.3], 2), support=8)
    _write_in(tmp_path, "insid3_dense_k8", _record("insid3", "dsb2018", [0.2, 0.2], 2), support=8)
    _run_tree(tmp_path, ours="ours", support=8)          # must NOT raise
    out = capsys.readouterr().out
    assert "insid3_guided" in out and "insid3_dense" in out, \
        "both CRF modes must appear as separate columns"


def test_ours_matching_nothing_refuses_instead_of_printing_an_empty_test_table(tmp_path):
    """The worst silent failure in the stage.

    An unmatched --ours used to print the significance banner, the column header, no rows, then
    "Holm-Bonferroni over the 0 tests" -- and exit 0. That reads as "compared and tied". Since the
    default --ours is a stale method name, it was the DEFAULT invocation's output.
    """
    _write_in(tmp_path, "ours_k8", _record("head_fusion_best", "dsb2018", [0.8, 0.8], 2), support=8)
    _write_in(tmp_path, "tyche_k8", _record("tyche", "dsb2018", [0.2, 0.2], 2), support=8)
    with pytest.raises(SystemExit, match="matches no record"):
        _run_tree(tmp_path, ours="head_fusion_uni_adapt", support=8)


def test_ours_resolves_by_harness_method_name_as_well_as_job_label(tmp_path, capsys):
    """Both spellings are natural to type; the directory carries one and the record the other."""
    _write_in(tmp_path, "ours_k8", _record("head_fusion_best_cgate_film_nobank", "dsb2018",
                                           [0.8, 0.8], 2), support=8)
    _write_in(tmp_path, "tyche_k8", _record("tyche", "dsb2018", [0.2, 0.2], 2), support=8)
    _run_tree(tmp_path, ours="head_fusion_best_cgate_film_nobank", support=8)
    out = capsys.readouterr().out
    assert "resolved to job label 'ours'" in out
    assert "+0.600" in out, "the paired row must still be produced after resolution"


def test_a_method_absent_from_this_k_family_is_named_not_silently_dropped(tmp_path, capsys):
    """Matcher runs at K in {1,4} and PerSAM at {1,8} BY DESIGN, so every K table is missing one.

    Dropping them without a word makes a by-design absence and a crashed job look identical, and
    the method gets no column at all -- not even a dash.
    """
    _write_in(tmp_path, "ours_k8", _record("head_fusion_best", "dsb2018", [0.8, 0.8], 2), support=8)
    _write_in(tmp_path, "matcher_k1", _record("matcher", "dsb2018", [0.2, 0.2], 2), support=1)
    _run_tree(tmp_path, ours="ours", support=8)
    out = capsys.readouterr().out
    assert "not in this K family" in out
    assert "matcher" in out and "K=[1]" in out


def test_a_k_dependent_record_in_a_k_less_directory_is_refused(tmp_path):
    """The support-blind claim is VERIFIED from the numbers, not taken from the directory name.

    An in-context method dropped in a `_k`-less directory would otherwise be announced as
    support-blind and replicated into every K table, with paired p-values computed against it.
    The off-the-shelf specialists record protocol.support too (argparse default), so a missing
    support field does not identify them either.
    """
    _write_in(tmp_path, "ours_k8", _record("head_fusion_best", "dsb2018", [0.8, 0.8], 2), support=8)
    # two seeds that DIFFER -> the support draw changed the result -> not support-blind
    _write_in(tmp_path, "universeg", _record("universeg", "dsb2018", [0.2, 0.2, 0.5, 0.5], 2),
              support=8)
    with pytest.raises(SystemExit, match="not support-blind"):
        _run_tree(tmp_path, ours="ours", support=8)


def test_a_genuinely_support_blind_record_joins_every_k_family(tmp_path, capsys):
    """The real off-the-shelf scripts score the split once and replicate it across seeds."""
    _write_in(tmp_path, "ours_k8", _record("head_fusion_best", "dsb2018", [0.8, 0.8], 2), support=8)
    # identical per-seed rows, which is what "ignores the support" means operationally
    _write_in(tmp_path, "cellpose_sam",
              _record("cellpose_sam", "dsb2018", [0.4, 0.4, 0.4, 0.4], 2), support=8)
    _run_tree(tmp_path, ours="ours", support=8)
    out = capsys.readouterr().out
    assert "K-independent" in out and "cellpose_sam" in out
    row = [ln for ln in out.splitlines()
           if "dsb2018" in ln and "cellpose_sam" in ln and "SKIPPED" not in ln]
    assert row, "a K-independent baseline must still be paired inside a K table"


def test_selecting_an_empty_k_family_says_the_path_is_fine(tmp_path):
    """The old message sent the operator to check the path -- the one thing that was not wrong."""
    _write_in(tmp_path, "ours_k8", _record("head_fusion_best", "dsb2018", [0.8, 0.8], 2), support=8)
    with pytest.raises(SystemExit, match="the PATH is fine, the K family is empty"):
        _run_tree(tmp_path, ours="ours", support=32)


def test_flat_layout_still_works_and_is_not_called_k_independent(tmp_path, capsys):
    """Backwards compatibility, plus: on a flat tree every method used to be announced as
    'K-independent (support ignored)', which is false for every in-context method and gets pasted
    into CONFIG-REGISTRY."""
    _write(tmp_path, _record("ours", "dsb2018", [0.8, 0.8], 2))
    _write(tmp_path, _record("rival", "dsb2018", [0.2, 0.2], 2))
    _run_tree(tmp_path, ours="ours")
    out = capsys.readouterr().out
    assert "SOTA FINAL" in out
    assert "K-independent" not in out, "no K was requested, so nothing may be called K-independent"


def test_rejected_records_are_counted_in_the_header(tmp_path, capsys):
    """12-of-13-methods-rejected used to print a one-column table and exit 0."""
    _write_in(tmp_path, "ours_k8", _record("head_fusion_best", "dsb2018", [0.8, 0.8], 2), support=8)
    _write_in(tmp_path, "tyche_k8", _record("tyche", "dsb2018", [0.2, 0.2], 2), support=8)
    (tmp_path / "broken_k8").mkdir()
    (tmp_path / "broken_k8" / "broken__dsb2018.json").write_text(
        json.dumps(_record("broken", "dsb2018", [0.5, 0.6, 0.7], 2)))
    _run_tree(tmp_path, ours="ours", support=8)
    out = capsys.readouterr().out
    assert "were REJECTED as malformed" in out


def test_non_record_json_in_the_tree_is_skipped_not_fatal(tmp_path, capsys):
    _write_in(tmp_path, "ours_k8", _record("head_fusion_best", "dsb2018", [0.8, 0.8], 2), support=8)
    _write_in(tmp_path, "tyche_k8", _record("tyche", "dsb2018", [0.2, 0.2], 2), support=8)
    (tmp_path / "summary.json").write_text(json.dumps({"note": "aggregate, not a record"}))
    _run_tree(tmp_path, ours="ours", support=8)
    out = capsys.readouterr().out
    assert "skipping non-record" in out and "SOTA FINAL" in out


def test_empty_tree_refuses_instead_of_printing_an_empty_table(tmp_path):
    with pytest.raises(SystemExit, match="no record files were found at all"):
        _run_tree(tmp_path, ours="ours", support=8)


def test_directory_k_disagreeing_with_measured_support_is_fatal(tmp_path):
    _write_in(tmp_path, "rival_k16", _record("rival", "dsb2018", [0.2, 0.2], 2), support=8)
    with pytest.raises(SystemExit, match="K MISLABELLED"):
        _run_tree(tmp_path, ours="ours", support=16)


def test_an_uncomputable_test_still_counts_in_the_holm_family(tmp_path, capsys):
    """Dropping it would SHRINK n_tests and make every surviving p_holm smaller — flattering us.

    A comparison whose Wilcoxon cannot be computed was still attempted, so it belongs in the
    family-wise correction. Excluding it is the anti-conservative direction, and it favours our own
    method, which is exactly the kind of error a reviewer is entitled to assume is deliberate.
    """
    import scipy.stats as _sp

    _write(tmp_path, _record("ours", "dsb2018", [0.5, 0.6, 0.7, 0.8], 4))
    _write(tmp_path, _record("rival1", "dsb2018", [0.1, 0.2, 0.3, 0.4], 4))
    _write(tmp_path, _record("rival2", "dsb2018", [0.2, 0.3, 0.4, 0.5], 4))

    # Force the uncomputable branch. Constructing it from data alone is not reliable -- the code
    # short-circuits the all-equal case to p=1.0 before scipy is reached -- and a test that cannot
    # reach the branch it names is exactly the kind this suite has been pruning.
    real = _sp.wilcoxon
    calls = {"n": 0}

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("synthetic: zero_method requires non-zero differences")
        return real(a, b)

    _sp.wilcoxon = flaky
    try:
        _run_stats(tmp_path)
    finally:
        _sp.wilcoxon = real
    out = capsys.readouterr().out
    assert calls["n"] == 2, "both comparisons must be attempted"
    assert "could not be computed" in out, "the uncomputable comparison must name itself"
    assert "Holm-Bonferroni over the 2 comparisons" in out, \
        "both attempted comparisons must be in the family, not just the computable one"
    assert "1 of them uncomputable" in out


def test_orphaned_records_for_unknown_datasets_are_named(tmp_path, capsys):
    """The registry guarantee runs forward only: no registry dataset is forgotten. A record whose
    dataset is NOT a registry key is dropped by every loop with no message, so a rename orphans
    its records invisibly."""
    _write(tmp_path, _record("ours", "dsb2018", [0.8, 0.8], 2))
    _write(tmp_path, _record("rival", "dsb2018", [0.2, 0.2], 2))
    _write(tmp_path, _record("ours", "renamed_dataset", [0.9, 0.9], 2))
    _run_stats(tmp_path)
    out = capsys.readouterr().out
    assert "ORPHANED records" in out and "renamed_dataset" in out


def test_two_methods_sharing_a_truncated_prefix_get_distinct_columns(tmp_path, capsys):
    """`head_fusion_best_cgate_film` and `..._nobank` both render as `head_fusion_best` in a
    16-character header. CLAUDE.md documents both existing, so two different methods could occupy
    adjacent columns under one identical label."""
    _write(tmp_path, _record("head_fusion_best_cgate_film", "dsb2018", [0.7, 0.7], 2))
    _write(tmp_path, _record("head_fusion_best_cgate_film_nobank", "dsb2018", [0.8, 0.8], 2))
    _run_stats(tmp_path, ours="head_fusion_best_cgate_film_nobank")
    out = capsys.readouterr().out
    header = [ln for ln in out.splitlines() if ln.strip().startswith("dataset")][0]
    cols = header.split()[2:]
    assert len(cols) == len(set(cols)), f"ambiguous column headers: {cols}"
    assert "column aliases" in out, "the alias mapping must be printed"
