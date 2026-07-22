import threading

import numpy as np

from active_segmenter.encoder.cache import EmbeddingCache


def test_cache_computes_once(tmp_path):
    c = EmbeddingCache(str(tmp_path))
    img = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return np.ones((2, 2, 3), np.float32)

    a = c.get_or_compute(img, "res672", fn)
    b = c.get_or_compute(img, "res672", fn)
    assert calls["n"] == 1  # second call hits cache
    assert np.allclose(a, b)
    assert a.dtype == np.float32


def test_cache_key_depends_on_extra(tmp_path):
    c = EmbeddingCache(str(tmp_path))
    img = np.zeros((4, 4, 3), np.uint8)
    c.get_or_compute(img, "res448", lambda: np.zeros((1, 1, 1), np.float32))
    calls = {"n": 0}

    def fn():
        calls["n"] = 1
        return np.zeros((1, 1, 1), np.float32)

    c.get_or_compute(img, "res672", fn)  # different extra -> recompute
    assert calls["n"] == 1


def test_cache_key_depends_on_image(tmp_path):
    c = EmbeddingCache(str(tmp_path))
    img1 = np.zeros((4, 4, 3), np.uint8)
    img2 = np.ones((4, 4, 3), np.uint8)
    c.get_or_compute(img1, "r", lambda: np.zeros((1, 1, 1), np.float32))
    calls = {"n": 0}
    c.get_or_compute(img2, "r", lambda: (calls.__setitem__("n", 1) or np.zeros((1, 1, 1), np.float32)))
    assert calls["n"] == 1


def test_concurrent_writers_of_the_same_key_all_succeed(tmp_path):
    """Two writers missing the SAME key must not fight over one temp file.

    The campaign runs several benchmark jobs at once against one shared feature cache, so this
    collision is the normal case, not a corner case. With a single shared ``<path>.tmp`` the loser's
    ``os.replace`` raised FileNotFoundError, which surfaced as a mid-campaign job crash.
    """
    c = EmbeddingCache(str(tmp_path))
    img = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    expected = np.full((8, 8), 3.5, np.float32)
    start = threading.Barrier(2)
    errors, results = [], []

    def slow_fn():
        start.wait(timeout=5)  # force both writers into get_or_compute's write path together
        return expected

    def worker():
        try:
            results.append(c.get_or_compute(img, "res672", slow_fn))
        except BaseException as exc:  # noqa: BLE001 - the assertion below reports it
            errors.append(exc)

    ts = [threading.Thread(target=worker) for _ in range(2)]
    [t.start() for t in ts]
    [t.join(timeout=10) for t in ts]

    assert not errors, f"concurrent writers raised {errors!r}"
    assert len(results) == 2
    for r in results:
        assert np.allclose(r, expected)
    assert np.allclose(np.load(c._path(c._key(img, "res672"))), expected)
    assert not list(tmp_path.glob("*.tmp")), "temp files leaked into the cache dir"
