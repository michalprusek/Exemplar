"""The merge that puts two hosts' campaign trees together.

The campaign runs on machines with separate filesystems, so records that must be compared never
meet until something merges them. A plain copy would be wrong in ways that are invisible afterwards:
stats() SKIPS a mismatched pair rather than reporting it, so a split or protocol disagreement
between hosts shows up as a quietly missing significance test, not an error.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "merge_score_trees.py"


def _rec(method, dataset, per_image, n, split_fp="aaaa", pool=20, seeds=None):
    return dict(method=method, dataset=dataset, metric="fg_iou", test_per_seed=n,
                seeds=seeds if seeds is not None else list(range(len(per_image) // n)),
                per_image=per_image, split_fp=split_fp,
                protocol=dict(pool=pool, test=n, support=8, split_seed=0))


def _write(tree, sub, rec):
    d = pathlib.Path(tree) / sub
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{rec['method']}__{rec['dataset']}.json").write_text(json.dumps(rec))


def _run(out, *trees):
    cmd = [sys.executable, str(SCRIPT), "--out", str(out)]
    for name, path in trees:
        cmd += ["--tree", f"{name}={path}"]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_disjoint_trees_merge_into_one(tmp_path):
    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    _write(a, "ours_k8", _rec("m", "dsb2018", [0.8, 0.8], 2))
    _write(b, "tyche_k8", _rec("tyche", "dsb2018", [0.2, 0.2], 2))
    r = _run(out, ("tulen", a), ("kajman", b))
    assert r.returncode == 0, r.stderr
    assert (out / "ours_k8" / "m__dsb2018.json").exists()
    assert (out / "tyche_k8" / "tyche__dsb2018.json").exists()
    assert "1 from tulen" in r.stdout and "1 from kajman" in r.stdout


def test_identical_duplicates_are_accepted_quietly(tmp_path):
    """Re-running a cell after a restart is normal; the SAME result twice is not a conflict."""
    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    rec = _rec("m", "dsb2018", [0.8, 0.8], 2)
    _write(a, "ours_k8", rec)
    _write(b, "ours_k8", rec)
    r = _run(out, ("tulen", a), ("kajman", b))
    assert r.returncode == 0, r.stderr
    assert "merged 1 record" in r.stdout


def test_the_same_cell_with_different_results_refuses(tmp_path):
    """Keeping one silently would put a number nobody chose into the paper."""
    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    _write(a, "ours_k8", _rec("m", "dsb2018", [0.8, 0.8], 2))
    _write(b, "ours_k8", _rec("m", "dsb2018", [0.4, 0.4], 2))
    r = _run(out, ("tulen", a), ("kajman", b))
    assert r.returncode == 1
    assert "measured twice with DIFFERENT results" in r.stderr
    assert "tulen" in r.stderr and "kajman" in r.stderr
    assert not out.exists(), "a refused merge must not leave a half-written tree"


def test_split_fingerprint_disagreement_across_hosts_is_reported(tmp_path):
    """The hosts hold separate copies of the data. If they load different test images, every
    cross-host comparison is meaningless -- and stats() skips it silently rather than complaining."""
    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    _write(a, "ours_k8", _rec("m", "dsb2018", [0.8, 0.8], 2, split_fp="aaaa"))
    _write(b, "tyche_k8", _rec("tyche", "dsb2018", [0.2, 0.2], 2, split_fp="bbbb"))
    r = _run(out, ("tulen", a), ("kajman", b))
    assert r.returncode == 2
    assert "disagree on the TEST SPLIT" in r.stderr
    assert "aaaa" in r.stderr and "bbbb" in r.stderr


def test_protocol_disagreement_is_reported(tmp_path):
    """Same split, different seed count or pool: still not comparable."""
    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    _write(a, "ours_k8", _rec("m", "dsb2018", [0.8, 0.8], 2))
    _write(b, "tyche_k8", _rec("tyche", "dsb2018", [0.2, 0.2, 0.2, 0.2], 2))   # 2 seeds vs 1
    r = _run(out, ("tulen", a), ("kajman", b))
    assert r.returncode == 2
    assert "disagree on the PROTOCOL" in r.stderr


def test_an_existing_destination_refuses_without_force(tmp_path):
    """Merging into a populated tree mixes campaigns and the result reads as one."""
    a, out = tmp_path / "a", tmp_path / "out"
    _write(a, "ours_k8", _rec("m", "dsb2018", [0.8, 0.8], 2))
    out.mkdir()
    r = _run(out, ("tulen", a))
    assert r.returncode != 0
    assert "already exists" in r.stderr
