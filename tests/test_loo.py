"""(2) The leave-one-image-out errors computed from the stored per-image sums equal brute-force refits without that
image, with each fold's standardisation and class weights taken from the other images only; and the per-image sums
equal the explicit sums of the pixel rows."""
import numpy as np
import pytest
import torch

from exemplar.readout import (column_shift, fold_pieces, heldout_cv, image_sums, label_tensor, rows, shift_sample,
                              totals)
from synth import features, support


def _rows_and_labels(K=4):
    images, masks = support(K)
    feats = [features(im) for im in images]
    i0 = shift_sample(feats[0][3], np.random.default_rng(0))
    pc, ps = column_shift(*feats[0], i0)
    out = []
    for (xc, fine, bank, hw), m in zip(feats, masks):
        idx = torch.arange(hw[0] * hw[1])
        R = rows(xc, fine, bank, hw, idx, pc, ps)
        y = torch.from_numpy(m.reshape(-1) > 0)
        out.append((R, y))
    return feats, masks, pc, ps, out


def _exact_groups(data):
    """Per-image sums formed directly from the rows (no bfloat16 rounding), to test the fold algebra alone."""
    groups = []
    for R, y in data:
        Rd = R.double()
        Rf = Rd[y]
        groups.append(dict(G=(Rd.T @ Rd).float(), Gf=(Rf.T @ Rf).float(), bb=Rf.sum(0).float(),
                           nn=float(R.shape[0]), nf=float(y.sum())))
    return groups


def _brute_force_cv(data, vname, lam):
    """Mean over folds of the held-out weighted squared error of a ridge refitted without that image, in float64."""
    errs, sols = [], []
    for k in range(len(data)):
        Xo = torch.cat([data[j][0] for j in range(len(data)) if j != k]).double()
        yo = torch.cat([data[j][1] for j in range(len(data)) if j != k]).double()
        mu = Xo[:, :-1].mean(0)
        sd = torch.sqrt(((Xo[:, :-1] ** 2).mean(0) - mu ** 2).clamp_min(0))
        sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
        rho = min(max(float(yo.mean()), 1e-6), 1 - 1e-6)
        q_fg, q_bg = (1.0, 1.0) if vname == "plain" else (0.5 / rho, 0.5 / (1 - rho))

        def z(X):
            return torch.cat([(X[:, :-1] - mu) / sd, X[:, -1:]], 1)
        Zo = z(Xo)
        qo = torch.where(yo > 0, q_fg, q_bg).double()
        n = float(qo.sum())
        A = Zo.T @ (Zo * qo[:, None]) / n
        b = Zo.T @ (qo * yo) / n
        P = torch.ones(A.shape[0], dtype=torch.float64)
        P[-1] = 0.0
        w = torch.linalg.solve(A + lam * torch.diag(P), b)
        Xk, yk = data[k][0].double(), data[k][1].double()
        qk = torch.where(yk > 0, q_fg, q_bg).double()
        errs.append(float((qk * (yk - z(Xk) @ w) ** 2).sum() / qk.sum()))
        sols.append(w)
    return float(np.mean(errs)), sols


@pytest.mark.parametrize("vname", ["plain", "bal"])
def test_loo_from_sums_equals_brute_force(vname):
    _, _, _, _, data = _rows_and_labels()
    tot = totals(_exact_groups(data))
    folds, _ = fold_pieces(tot, vname)
    B32 = [fo["B"].float() for fo in folds]
    Pd = torch.ones(tot["Gt"].shape[0])
    Pd[-1] = 0.0
    for lam in (1e-3, 0.05, 2.0):
        cv, _, _, sols = heldout_cv(folds, B32, Pd, lam)
        cv_bf, sols_bf = _brute_force_cv(data, vname, lam)
        assert abs(cv - cv_bf) / cv_bf < 1e-3, (vname, lam, cv, cv_bf)
        for w, wb in zip(sols, sols_bf):           # the fold solutions themselves, in standardised coordinates
            assert float((w.double() - wb).abs().max() / wb.abs().max()) < 1e-3


def test_image_sums_match_pixel_sums():
    feats, masks, pc, ps, data = _rows_and_labels(K=2)
    for (xc, fine, bank, hw), m, (R, y) in zip(feats, masks, data):
        g = image_sums(xc, fine, bank, hw, label_tensor(m, hw, "cpu"), pc, ps, quadrants=False)[0]
        Rd = R.double()
        Gp, Gfp, bbp = Rd.T @ Rd, Rd[y].T @ Rd[y], Rd[y].sum(0)
        assert float((g["G"].double() - Gp).abs().max() / Gp.abs().max()) < 2e-6     # exact on the grids
        # the foreground Gram is formed from rows rounded to bfloat16 (8 bits of mantissa)
        assert float((g["Gf"].double() - Gfp).abs().max() / Gfp.abs().max()) < 2e-2
        assert torch.allclose(g["bb"].double(), bbp, rtol=1e-5, atol=1e-3)          # the sum uses unrounded rows
        assert g["nn"] == float(R.shape[0]) and g["nf"] == float(y.sum())
