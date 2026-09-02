import numpy as np

from active_segmenter.config import RefineConfig
from active_segmenter.refine import build_refiner
from active_segmenter.refine.identity import IdentityRefiner
from active_segmenter.types import InstanceMask


def _grid_instances():
    m1 = np.zeros((4, 4), bool); m1[0:2, 0:2] = True
    m2 = np.zeros((4, 4), bool); m2[2:4, 2:4] = True
    return [
        InstanceMask(mask=m1, points=None, class_id=1, instance_id=0),
        InstanceMask(mask=m2, points=None, class_id=1, instance_id=1),
    ]


def test_identity_upsamples_to_image():
    r = IdentityRefiner(RefineConfig(kind="identity"))
    img = np.zeros((16, 16), np.uint8)
    out = r.refine(img, _grid_instances(), feat_grid=None)
    assert len(out) == 2
    assert all(m.mask.shape == (16, 16) for m in out)


def test_identity_preserves_separation():
    r = IdentityRefiner(RefineConfig())
    img = np.zeros((16, 16), np.uint8)
    out = r.refine(img, _grid_instances(), feat_grid=None)
    # the two instances stay disjoint (no merge into a shared raster)
    assert not np.logical_and(out[0].mask, out[1].mask).any()
    assert out[0].mask.any() and out[1].mask.any()


def test_build_refiner_identity():
    r = build_refiner(RefineConfig(kind="identity"), device="cpu")
    assert isinstance(r, IdentityRefiner)


def test_identity_passthrough_native_masks():
    r = IdentityRefiner(RefineConfig())
    m = np.zeros((16, 16), bool); m[2:8, 2:8] = True
    inst = [InstanceMask(mask=m, points=None, class_id=1, instance_id=0)]
    out = r.refine(np.zeros((16, 16), np.uint8), inst, feat_grid=None)
    assert out[0].mask.shape == (16, 16)
    assert out[0].mask.sum() == m.sum()  # already native -> unchanged
