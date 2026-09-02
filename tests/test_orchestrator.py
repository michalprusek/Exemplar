import numpy as np

from active_segmenter.config import RunConfig, EncoderConfig, ClusterConfig
from active_segmenter.loop.orchestrator import ALLoop, run_al_vs_random
from active_segmenter.refine.identity import IdentityRefiner
from active_segmenter.config import RefineConfig


class FakeEncoder:
    """Deterministic separable encoder: bright pixels -> +e0 features, dark -> -e0.
    No model needed — lets the whole AL loop run on CPU."""

    def __init__(self, grid=8, dim=6):
        self.grid = grid
        self.dim = dim

    def extract(self, image):
        from skimage.transform import resize

        img = np.asarray(image, np.float32)
        if img.ndim == 3:
            img = img.mean(2)
        g = resize(img, (self.grid, self.grid), order=1, mode="edge", anti_aliasing=False)
        feat = np.zeros((self.grid, self.grid, self.dim), np.float32)
        fg = g > 127
        feat[..., 0] = np.where(fg, 1.0, -1.0)
        feat[..., 1] = 0.01  # tiny constant so norms are well-defined
        feat /= np.linalg.norm(feat, axis=2, keepdims=True)
        return feat

    def extract_cls(self, image):
        img = np.asarray(image, np.float32)
        if img.ndim == 3:
            img = img.mean(2)
        v = np.zeros(self.dim, np.float32)
        v[0] = 1.0 if img.mean() > 60 else -1.0
        v[1] = float(img.mean()) / 255.0
        return v / (np.linalg.norm(v) + 1e-8)


def _make_dataset(n=8, size=32, seed=0):
    """Bright square 'cells' on dark background; label map has per-instance ids."""
    rng = np.random.RandomState(seed)
    data = []
    for _ in range(n):
        img = np.zeros((size, size), np.uint8)
        lbl = np.zeros((size, size), np.int32)
        k = rng.randint(1, 3)
        for inst in range(1, k + 1):
            y, x = rng.randint(2, size - 10), rng.randint(2, size - 10)
            img[y:y + 8, x:x + 8] = 255
            lbl[y:y + 8, x:x + 8] = inst
        data.append((img, lbl))
    return data


def _cfg():
    return RunConfig(device="cpu", cache_dir="/tmp/asg_test",
                     encoder=EncoderConfig(resolution=128),
                     cluster=ClusterConfig(score_thresh=0.0, min_patches=1, distance_threshold=1.5))


def test_loop_runs_and_bank_grows():
    from active_segmenter.eval.oracle import GtOracle

    pool = _make_dataset(8, seed=1)
    test = _make_dataset(4, seed=2)
    cfg = _cfg()
    loop = ALLoop(cfg, FakeEncoder(), IdentityRefiner(RefineConfig()),
                  GtOracle(pool), seed=0)
    results = loop.run(list(range(len(pool))), test, rounds=3, cold_k=2, acquisition_name="uncertainty")
    assert len(results) == 3
    ns = [r.n_annotated for r in results]
    assert ns == sorted(ns) and ns[0] == 2 and ns[-1] > ns[0]  # grows each round
    assert all(0.0 <= r.fg_iou <= 1.0 for r in results)
    assert results[-1].fg_iou > 0.5  # separable fake -> decent IoU


def test_al_vs_random_equal_length_curves():
    pool = _make_dataset(8, seed=3)
    test = _make_dataset(4, seed=4)
    out = run_al_vs_random(_cfg(), pool, test, rounds=3, cold_k=2,
                           encoder=FakeEncoder(), refiner=IdentityRefiner(RefineConfig()), seed=0)
    assert set(out.keys()) == {"al", "random"}
    assert len(out["al"]) == len(out["random"]) == 3
    # shared cold start -> identical round-0 IoU
    assert abs(out["al"][0].fg_iou - out["random"][0].fg_iou) < 1e-9


def test_oracle_reveals_grow_each_round():
    from active_segmenter.eval.oracle import GtOracle

    pool = _make_dataset(8, seed=5)
    test = _make_dataset(4, seed=6)
    oracle = GtOracle(pool)
    loop = ALLoop(_cfg(), FakeEncoder(), IdentityRefiner(RefineConfig()), oracle, seed=0)
    loop.run(list(range(len(pool))), test, rounds=3, cold_k=2, acquisition_name="random")
    assert oracle.n_revealed == 4  # cold_k=2 + one per the 2 subsequent rounds
