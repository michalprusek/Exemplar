"""Tests for the few-shot support/test directory loader (e.g. the decay/rozpad set)."""
import numpy as np
from PIL import Image

from active_segmenter.eval.datasets import load_fewshot


def _png(path, arr):
    Image.fromarray(arr.astype(np.uint8)).save(str(path))


def test_load_fewshot_reads_pairs_and_binarizes(tmp_path):
    (tmp_path / "support" / "images").mkdir(parents=True)
    (tmp_path / "support" / "masks").mkdir(parents=True)
    img = (np.arange(16, dtype=np.uint8).reshape(4, 4) * 15)
    mask = np.array([[0, 0, 255, 255]] * 4, np.uint8)   # binary 0/255 foreground
    _png(tmp_path / "support" / "images" / "a.png", img)
    _png(tmp_path / "support" / "masks" / "a.png", mask)

    data = load_fewshot(str(tmp_path), "support")
    assert len(data) == 1
    im, lbl = data[0]
    assert im.shape == (4, 4)
    assert set(np.unique(lbl)) <= {0, 1}                  # binarised to {0,1}
    assert (lbl[:, 2:] == 1).all() and (lbl[:, :2] == 0).all()


def test_load_fewshot_train_alias_maps_to_support(tmp_path):
    (tmp_path / "support" / "images").mkdir(parents=True)
    (tmp_path / "support" / "masks").mkdir(parents=True)
    _png(tmp_path / "support" / "images" / "x.png", np.zeros((4, 4), np.uint8))
    _png(tmp_path / "support" / "masks" / "x.png", np.zeros((4, 4), np.uint8))
    assert len(load_fewshot(str(tmp_path), "train")) == 1   # "train" -> "support"


def test_load_fewshot_limit(tmp_path):
    (tmp_path / "support" / "images").mkdir(parents=True)
    (tmp_path / "support" / "masks").mkdir(parents=True)
    for i in range(5):
        _png(tmp_path / "support" / "images" / f"{i}.png", np.zeros((4, 4), np.uint8))
        _png(tmp_path / "support" / "masks" / f"{i}.png", np.zeros((4, 4), np.uint8))
    assert len(load_fewshot(str(tmp_path), "support", limit=2)) == 2
