import numpy as np

from active_segmenter.eval.oracle import GtOracle


def test_oracle_reveals_only_on_request():
    ds = [(np.zeros((2, 2)), np.array([[1, 0], [0, 2]])) for _ in range(3)]
    o = GtOracle(ds)
    assert o.n_revealed == 0
    lbl = o.reveal(1)
    assert lbl.max() == 2
    assert o.revealed == {1}
    assert o.n_revealed == 1
    o.reveal(1)  # idempotent set
    assert o.n_revealed == 1
