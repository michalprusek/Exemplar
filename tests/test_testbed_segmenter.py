import numpy as np

from active_segmenter.config import EncoderConfig, RunConfig


def test_make_backend_correspondence():
    from scripts.al_testbed import make_backend
    from active_segmenter.segment.correspondence_backend import CorrespondenceBackend

    cfg = RunConfig(device="cpu", encoder=EncoderConfig(resolution=96))
    be = make_backend("correspondence", cfg, "cpu")
    assert isinstance(be, CorrespondenceBackend)


def test_make_backend_unknown_raises():
    from scripts.al_testbed import make_backend

    cfg = RunConfig(device="cpu")
    try:
        make_backend("nope", cfg, "cpu")
        assert False
    except (ValueError, KeyError):
        pass


def test_ctx_backend_routing_smoke():
    """A tiny end-to-end: attach a correspondence backend and route test_iou through it."""
    from scripts.al_testbed import Ctx, make_backend

    rng = np.random.default_rng(0)
    # two toy train + two toy test images, with matching feature grids
    train = [(np.zeros((24, 24)), _lbl()) for _ in range(2)]
    test = [(np.zeros((24, 24)), _lbl()) for _ in range(2)]
    trf = [rng.standard_normal((6, 6, 8)).astype(np.float32) for _ in range(2)]
    tef = [rng.standard_normal((6, 6, 8)).astype(np.float32) for _ in range(2)]
    cls = rng.standard_normal((2, 8)).astype(np.float32)
    ctx = Ctx(trf, train, tef, test, cls, "cpu")
    cfg = RunConfig(device="cpu")
    ctx.set_backend(make_backend("correspondence", cfg, "cpu"))
    iou = ctx.test_iou_backend([0, 1])
    assert 0.0 <= iou <= 1.0


def _lbl():
    lm = np.zeros((24, 24), int)
    lm[4:12, 4:12] = 1
    return lm


def test_geoloop_scorer_runs_on_mock_ctx():
    """s_geoloop (fg-coverage x support-LOO) returns a score per pool item without a model."""
    from scripts.al_testbed import Ctx, s_geoloop

    rng = np.random.default_rng(0)
    n = 5
    train = [(np.zeros((32, 32)), _lbl()) for _ in range(n)]
    trf = [rng.standard_normal((6, 6, 8)).astype(np.float32) for _ in range(n)]
    cls = rng.standard_normal((n, 8)).astype(np.float32)
    ctx = Ctx(trf, train, trf, train, cls, "cpu")
    idxs = [0, 1]
    pool = [2, 3, 4]
    scores = s_geoloop(pool, idxs, ctx, rng)
    assert set(scores) == set(pool)
    assert all(np.isfinite(v) for v in scores.values())


def test_select_batch_returns_k_distinct_picks():
    from scripts.al_testbed import Ctx, _select_batch

    rng = np.random.default_rng(1)
    n = 6
    train = [(np.zeros((32, 32)), _lbl()) for _ in range(n)]
    trf = [rng.standard_normal((6, 6, 8)).astype(np.float32) for _ in range(n)]
    cls = rng.standard_normal((n, 8)).astype(np.float32)
    ctx = Ctx(trf, train, trf, train, cls, "cpu")

    def scorer(pool, idxs, ctx, rng):
        return {i: float(i) for i in pool}   # trivial increasing score

    picks = _select_batch(scorer, [1, 2, 3, 4, 5], [0], ctx, rng, batch=3)
    assert len(picks) == 3 and len(set(picks)) == 3
    assert all(p in [1, 2, 3, 4, 5] for p in picks)


def test_egl_badge_scorers_need_trainable_backend():
    """egl/badge are weight-coupled — they must error without a head backend attached."""
    from scripts.al_testbed import Ctx, s_egl, s_badge

    rng = np.random.default_rng(0)
    n = 4
    train = [(np.zeros((24, 24)), _lbl()) for _ in range(n)]
    trf = [rng.standard_normal((6, 6, 8)).astype(np.float32) for _ in range(n)]
    cls = rng.standard_normal((n, 8)).astype(np.float32)
    ctx = Ctx(trf, train, trf, train, cls, "cpu")   # no backend
    for scorer in (s_egl, s_badge):
        try:
            scorer([2, 3], [0, 1], ctx, rng)
            assert False, "expected RuntimeError without a trainable backend"
        except RuntimeError:
            pass


def test_badge_scorer_runs_with_head_backend():
    from scripts.al_testbed import Ctx, make_backend, s_egl
    from active_segmenter.config import RunConfig

    rng = np.random.default_rng(1)
    n = 5
    train = [(np.zeros((24, 24)), _lbl()) for _ in range(n)]
    trf = [rng.standard_normal((6, 6, 8)).astype(np.float32) for _ in range(n)]
    cls = rng.standard_normal((n, 8)).astype(np.float32)
    ctx = Ctx(trf, train, trf, train, cls, "cpu")
    from active_segmenter.segment.head_backend import TrainableHeadBackend
    ctx.set_backend(TrainableHeadBackend(device="cpu", hidden=16, epochs=3, warm_start=False))
    scores = s_egl([2, 3, 4], [0, 1], ctx, rng)
    assert set(scores) == {2, 3, 4} and all(np.isfinite(v) for v in scores.values())
