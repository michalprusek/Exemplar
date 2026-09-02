import numpy as np

from active_segmenter.eval.datasets import make_heterogeneous


def test_generates_multiple_domains():
    data, doms = make_heterogeneous(30, n_domains=3, skew=True, seed=0, return_domains=True)
    assert len(data) == 30
    assert len(set(doms)) >= 2  # heterogeneous pool
    img, lbl = data[0]
    assert img.dtype == np.uint8
    assert lbl.max() >= 1  # has instances


def test_skew_makes_rare_domains():
    _, doms = make_heterogeneous(60, n_domains=4, skew=True, seed=1, return_domains=True)
    counts = np.bincount(doms, minlength=4)
    # skewed: the most common domain has many more than the rarest present one
    assert counts.max() > counts[counts > 0].min()


def test_balanced_when_not_skewed():
    _, doms = make_heterogeneous(60, n_domains=3, skew=False, seed=2, return_domains=True)
    assert len(set(doms)) == 3  # all domains represented when balanced


def test_deterministic():
    a = make_heterogeneous(5, seed=3)
    b = make_heterogeneous(5, seed=3)
    assert np.array_equal(a[0][0], b[0][0])
