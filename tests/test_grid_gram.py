"""(1) The Gram matrix computed on the patch grids (Kronecker factorisation of the bilinear interpolation) equals the
explicit Gram of the interpolated per-pixel rows, for the whole image and for a quadrant."""
import torch

from exemplar.readout import grid_gram, rows


def _inputs(H=37, W=45, seed=0):
    g = torch.Generator().manual_seed(seed)
    xc = torch.randn(1, 5, 5, 6, generator=g)
    fine = torch.randn(1, 4, 9, 11, generator=g)
    bank = torch.randn(1, 3, H, W, generator=g)
    pc = torch.randn(5 + 4 + 3, generator=g) * 0.1
    ps = torch.rand(5 + 4 + 3, generator=g) + 0.5
    return xc, fine, bank, (H, W), pc, ps


def _grid_gram(xc, fine, bank, hw, pc, ps, r0, r1, c0, c1):
    C0, C1, cb = xc.shape[1], fine.shape[1], bank.shape[1]
    grids = [((xc[0].permute(1, 2, 0) - pc[:C0]) / ps[:C0]).contiguous(),
             ((fine[0].permute(1, 2, 0) - pc[C0:C0 + C1]) / ps[C0:C0 + C1]).contiguous()]
    bankn = (bank[0] - pc[-cb:, None, None]) / ps[-cb:, None, None]
    return grid_gram(grids, bankn, hw, r0, r1, c0, c1)


def test_grid_gram_equals_pixel_gram():
    xc, fine, bank, hw, pc, ps = _inputs()
    H, W = hw
    for (r0, r1, c0, c1) in ((0, H, 0, W), (0, H // 2, W // 2, W), (H // 2, H, 0, W // 2)):
        G = _grid_gram(xc, fine, bank, hw, pc, ps, r0, r1, c0, c1)
        yy, xx = torch.meshgrid(torch.arange(r0, r1), torch.arange(c0, c1), indexing="ij")
        idx = (yy * W + xx).reshape(-1)
        R = rows(xc, fine, bank, hw, idx, pc, ps).double()
        Gp = R.T @ R
        rel = float((G.double() - Gp).abs().max() / Gp.abs().max())
        assert G.shape == Gp.shape == (13, 13)
        assert rel < 2e-6, rel
        assert float(G[-1, -1]) == float(idx.numel())     # the constant column counts the pixels
