"""CPU smoke test for the HyperBank backend (classical path — no DINOv3 needed).

Trains the vendored classical bank on a tiny synthetic blob image and checks it segments it.
Fusion path is GPU/feature-dependent and exercised in the benchmark, not here.
"""
import numpy as np

from active_segmenter.segment.base import LabeledExample
from active_segmenter.segment.hyperbank_backend import HyperBankBackend


def _blob_image(H=256):
    yy, xx = np.mgrid[0:H, 0:H]
    m = (yy - 128) ** 2 + (xx - 128) ** 2 < 48 ** 2
    img = (m * 200 + np.random.default_rng(0).integers(0, 20, (H, H))).astype(np.float32)
    return img, m.astype(int)


def test_hyperbank_classical_fits_and_segments():
    img, mask = _blob_image()
    feat = np.zeros((4, 4, 8), np.float32)  # unused in fusion=False
    ex = LabeledExample(image=img, feat_grid=feat, label_map=mask)
    be = HyperBankBackend(device="cpu", fusion=False, epochs_base=8)
    be.fit([ex, ex])
    fg = be.foreground(img, feat)
    assert fg.shape == (256, 256) and fg.dtype == bool
    iou = (fg & (mask > 0)).sum() / ((fg | (mask > 0)).sum() + 1e-6)
    assert iou > 0.3  # a fit bank should recover most of a clean blob
    insts = be.predict(img, feat)
    assert isinstance(insts, list)
