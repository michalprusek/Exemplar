"""Two numerical pieces the held-out choice and the sums depend on: the device skeleton equals skimage's, and the
8-block upper-triangle product of `gmm` equals X'X of the bfloat16-rounded rows."""
import numpy as np
import torch
from skimage.morphology import skeletonize

from exemplar.readout import gmm, zs_skel


def _masks():
    rs = np.random.RandomState(0)
    out = []
    for i in range(120):                                  # random blobs of varying density
        m = rs.rand(40, 50) < (0.3 + 0.4 * (i % 3) / 2)
        if i % 2:
            from scipy import ndimage
            m = ndimage.binary_closing(m, iterations=1 + i % 3)
        out.append(m)
    yy, xx = np.mgrid[:64, :64]
    for i in range(80):                                   # vessel-like: thick sinusoidal curves and branches
        y0 = 32 + 12 * np.sin(xx / (4 + i % 7) + i)
        m = np.abs(yy - y0) < 1 + i % 4
        m |= (np.abs(xx - (10 + i % 40)) < 1 + i % 3) & (yy > 20)
        out.append(m)
    out.append(np.zeros((30, 30), bool))
    out.append(np.ones((30, 30), bool))
    return out


def test_zs_skel_equals_skimage():
    for m in _masks():
        ref = skeletonize(m)
        got = zs_skel(torch.from_numpy(m)[None])[0].numpy()
        assert np.array_equal(got, ref)
    batch = np.stack([np.pad(m, ((0, 64 - m.shape[0]), (0, 64 - m.shape[1]))) for m in _masks()[:16]])
    got = zs_skel(torch.from_numpy(batch)).numpy()
    for b in range(len(batch)):
        assert np.array_equal(got[b], skeletonize(batch[b]))


def test_gmm_block_path():
    g = torch.Generator().manual_seed(0)
    for d in (37, 1030):                                  # below and above the 8-block threshold (1024)
        X = torch.randn(50, d, generator=g)
        Xr = X.to(torch.bfloat16).float()
        ref = Xr.double().T @ Xr.double()
        out = gmm(X)
        assert out.dtype == torch.float32 and out.shape == (d, d)
        assert float((out.double() - ref).abs().max() / ref.abs().max()) < 1e-5
        assert torch.equal(out[:500, 600:], out[600:, :500].T)   # lower blocks are the transposes of the upper ones
