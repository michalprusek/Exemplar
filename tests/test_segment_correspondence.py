import numpy as np

from active_segmenter.config import ClusterConfig, MatchConfig
from active_segmenter.membank.bank import MemoryBank
from active_segmenter.propose import correspondence as corr
from active_segmenter.segment.base import LabeledExample, foreground_from_score
from active_segmenter.segment.correspondence_backend import CorrespondenceBackend


def _toy_support(seed=0):
    rng = np.random.default_rng(seed)
    feat = rng.standard_normal((6, 6, 8)).astype(np.float32)
    label = np.zeros((24, 24), int)
    label[4:12, 4:12] = 1
    return [LabeledExample(image=np.zeros((24, 24)), feat_grid=feat, label_map=label)]


def test_correspondence_backend_reproduces_scoremap():
    """The backend's foreground must equal the raw corr.score_map>0 upsample — the
    exact quantity the current AL testbed measures (regression anchor)."""
    mc = MatchConfig(topk=5, bidirectional=False)
    sup = _toy_support()
    be = CorrespondenceBackend(mc, ClusterConfig(), device="cpu")
    be.fit(sup)
    query = np.random.default_rng(1).standard_normal((6, 6, 8)).astype(np.float32)
    # reference: build the same bank by hand and score
    bank = MemoryBank()
    bank.add_from_annotation(sup[0].feat_grid, sup[0].label_map, {1: 1}, 0)
    ref_score = corr.score_map(query, bank, 1, mc, device="cpu")
    ref_fg = foreground_from_score(ref_score, (24, 24))
    got = be.foreground(np.zeros((24, 24)), query)
    assert np.array_equal(got, ref_fg)


def test_correspondence_backend_predict_is_overlap_safe():
    be = CorrespondenceBackend(
        MatchConfig(topk=5, bidirectional=False), ClusterConfig(min_patches=1), device="cpu"
    )
    be.fit(_toy_support())
    query = np.random.default_rng(2).standard_normal((6, 6, 8)).astype(np.float32)
    insts = be.predict(np.zeros((24, 24)), query)
    assert isinstance(insts, list)
    for m in insts:
        assert m.mask.dtype == bool and m.mask.shape == (24, 24)
