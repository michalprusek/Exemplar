"""benchmark/prepare_data.py and the loaders' refusals, on the CPU without any download."""
import glob
import json
import os

import numpy as np
import pytest
from PIL import Image

from benchmark import datasets as D
from benchmark import prepare_data as P
from benchmark import registry as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_fingerprints_equal_the_paper_records():
    """--check compares against TEST_FP, so TEST_FP must be the split_fp of every Exemplar record of the paper."""
    recs = glob.glob(os.path.join(ROOT, "results", "final10", "ridge4_*", "ridge__*.json"))
    assert recs, "the paper's records are missing"
    seen = set()
    for p in recs:
        rec = json.load(open(p))
        assert rec["split_fp"] == P.TEST_FP[rec["dataset"]], p
        seen.add(rec["dataset"])
    assert seen == set(R.ALL_DATASETS) == set(P.TEST_FP) == set(P.POOL_FP) == set(P.PREP)


def test_custom_split_lists():
    assert len(P.SPHEROIDJ["support"]) == 48 and len(P.SPHEROIDJ["test"]) == 24
    assert not set(P.SPHEROIDJ["support"]) & set(P.SPHEROIDJ["test"])
    assert len(P.FISBE["train"]) == R.POOL["fisbe"] == 16 and len(P.FISBE["test"]) == 14
    assert not set(P.FISBE["train"]) & set(P.FISBE["test"])


def test_colour_labels_number_colours_in_sorted_order():
    rgb = np.zeros((2, 3, 3), np.uint8)
    rgb[0, 0] = (200, 0, 0)
    rgb[0, 1] = (10, 50, 0)
    rgb[1, 2] = (10, 40, 9)
    lbl = P.colour_labels(rgb)
    assert lbl.dtype == np.uint16
    assert lbl.tolist() == [[3, 2, 0], [0, 0, 1]]           # (10,40,9) < (10,50,0) < (200,0,0); black is 0
    with pytest.raises(ValueError):
        P.colour_labels(rgb + 1)                           # no black background


def test_round_robin_takes_one_of_each_group_per_pass():
    got = P.round_robin({"b": ["b2", "b1"], "A": ["a1"], "c": ["c1", "c2", "c3"]})
    assert got == [("A", "a1"), ("b", "b1"), ("c", "c1"), ("b", "b2"), ("c", "c2"), ("c", "c3")]


def test_flat_loader_refuses_an_image_without_mask(tmp_path):
    for sub in ("images", "masks"):
        os.makedirs(tmp_path / sub)
    for i in range(3):
        Image.fromarray(np.full((4, 4), i, np.uint8)).save(tmp_path / "images" / f"{i}.png")
    for i in range(2):
        Image.fromarray(np.full((4, 4), 255, np.uint8)).save(tmp_path / "masks" / f"{i}.png")
    with pytest.raises(FileNotFoundError):
        D.load_flat_fewshot(str(tmp_path / "images"), str(tmp_path / "masks"), 1, 10000)


def test_load_split_refuses_a_short_pool(tmp_path, monkeypatch):
    for split in ("support", "test"):
        for sub in ("images", "masks"):
            os.makedirs(tmp_path / split / sub)
        for i in range(3):
            for sub in ("images", "masks"):
                Image.fromarray(np.full((4, 4), 255, np.uint8)).save(tmp_path / split / sub / f"{i}.png")
    monkeypatch.setitem(R.PANEL, "spheroidj", R.DatasetSpec("spheroidj", "fewshot", str(tmp_path)))
    with pytest.raises(RuntimeError, match="pool"):
        R.load_split("spheroidj")                          # 3 support images where the protocol needs 20
