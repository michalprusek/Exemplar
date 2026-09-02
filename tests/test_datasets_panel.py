import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image

from active_segmenter.eval import datasets as D
from active_segmenter.eval.registry import PANEL, DatasetSpec, load_dataset


def _write(dir_, name, arr):
    dir_.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(str(dir_ / name))


def _make_split(root, split, n):
    for i in range(n):
        img = (np.random.default_rng(i).random((16, 16)) * 255)
        msk = np.zeros((16, 16)); msk[4:10, 4:10] = 255
        _write(root / split / "images", f"im{i}.png", img)
        _write(root / split / "masks", f"im{i}.png", msk)


def test_load_split_dir_subsamples_by_seed(tmp_path):
    _make_split(tmp_path, "train", 20)
    a = D.load_split_dir(str(tmp_path), "train", limit=5, seed=1)
    b = D.load_split_dir(str(tmp_path), "train", limit=5, seed=2)
    assert len(a) == 5 and len(b) == 5
    assert a[0][0].shape == (16, 16) and set(np.unique(a[0][1])) <= {0, 1}


def test_load_flat_fewshot_disjoint_split(tmp_path):
    img_dir = tmp_path / "images"
    msk_dir = tmp_path / "masks"
    for i in range(12):
        _write(img_dir, f"c{i}.png", np.full((8, 8), i * 10))
        _write(msk_dir, f"c{i}.png", (np.arange(64).reshape(8, 8) > 30) * 255)
    sup, test = D.load_flat_fewshot(str(img_dir), str(msk_dir), support=5, test=4, seed=0)
    assert len(sup) == 5 and len(test) == 4


def test_registry_load_dataset_dispatches_fewshot(tmp_path):
    _make_split(tmp_path, "support", 6)
    _make_split(tmp_path, "test", 4)
    spec = DatasetSpec("toy", "fewshot", str(tmp_path))
    sup, test = load_dataset(spec, support=6, test=4, seed=0)
    assert len(sup) == 6 and len(test) == 4


def test_panel_has_diverse_kinds():
    kinds = {s.kind for s in PANEL.values()}
    assert {"fewshot", "traintest", "dsb", "download"} <= kinds
    assert len(PANEL) >= 6
