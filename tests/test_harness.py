import csv
import os

from active_segmenter.config import RunConfig, EncoderConfig, ClusterConfig, RefineConfig
from active_segmenter.eval import harness
from active_segmenter.refine.identity import IdentityRefiner
from tests.test_orchestrator import FakeEncoder, _make_dataset


def _cfg():
    return RunConfig(device="cpu", cache_dir="/tmp/asg_test",
                     encoder=EncoderConfig(resolution=128),
                     cluster=ClusterConfig(score_thresh=0.0, min_patches=1, distance_threshold=1.5))


def test_harness_writes_artifacts(tmp_path):
    pool = _make_dataset(8, seed=1)
    test = _make_dataset(4, seed=2)
    report = harness.run(_cfg(), pool, test, name="fake", rounds=3, cold_k=2,
                         out_dir=str(tmp_path), encoder=FakeEncoder(),
                         refiner=IdentityRefiner(RefineConfig()), seed=0)
    d = os.path.join(str(tmp_path), "fake")
    assert os.path.exists(os.path.join(d, "curve.csv"))
    assert os.path.exists(os.path.join(d, "curve.png"))
    assert os.path.exists(os.path.join(d, "config.json"))
    with open(os.path.join(d, "curve.csv")) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert "al_fg_iou" in rows[0] and "random_fg_iou" in rows[0]
    assert report["al"][0].n_annotated == 2
