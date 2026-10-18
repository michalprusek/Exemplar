"""The closed-form readout: weighted ridge regression over every pixel of the support images.

Every pixel p of a support image gives one row f_p = [U_c F_c ; U_m F_m ; B_p ; 1] (the coarse and the fine grid
interpolated bilinearly to the pixel, the 35 priors, a constant; d = 4132). With y_p in {0, 1} and pixel weights q_p,

    w = argmin (1/n) sum_p q_p (y_p - w'f_p)^2 + lambda w'Iw = (A/n + lambda I)^-1 (b/n),   I = diag(1, ..., 1, 0)

in coordinates where every column but the constant is standardised over the support pixels. Two weightings come from
the same sums: plain (q = 1) and balanced (foreground and background carry equal total weight).

Per-image sums. Each support image keeps its raw sums: its Gram over all pixels (computed exactly on the patch grids,
`grid_gram`: bilinear interpolation factorises, U = U_y (x) U_x), its foreground Gram and sum (formed from the foreground
rows, rounded to bfloat16, TF32 products, `gmm`) and its counts. Every column is first shifted and scaled by fixed
constants from 4096 pixels of the first support image (the DINOv3 channels have a mean far above their spread).

Choices on held-out support images. Each fold leaves one support image out (at K = 1 one quadrant of the single image)
and is a fit from scratch on the others: its standardisation and class weights come from the other images' sums. lambda
of each weighting minimises the exact mean leave-one-image-out squared error (Newton's method on log lambda,
`newton_lam_folds`). The two readouts are blended, w_a = a w_plain + (1 - a) w_bal, cut at 1/2; a is chosen by the
held-out metric (IoU or centreline Dice) of the blended held-out maps, which are exactly a L_plain + (1 - a) L_bal of the
two weightings' held-out maps: a on 0, 0.2, ..., 1, refined by +-0.1 and +-0.05, ties to the larger a, and a = 1 (the
plain readout at 1/2) unless the gain exceeds one standard error (`choose_blend`).

Exact refit. Adding a mask adds one image's sums; standardisation, fold weights, lambda (Newton started where a cold
fit starts) and a are recomputed from the updated sums, so the result is bit-identical to a cold fit of the same masks.

Ablation (Table 2's input rows, `blocks`): the readout on a subset of the column blocks coarse, fine, priors. The sums
are always formed over every column the inputs carry; the removed blocks' rows and columns are dropped from every
group's sums before the standardisation, lambda and the blend are chosen, i.e. the ridge is solved on the rest, and the
removed blocks get weight 0 (`block_selection`). A subset without the fine view is fed inputs without it (fine = None),
as the original never computes it.

Precision: sums and solves in float32 (Cholesky, no refinement); a fold whose float32 factor is not definite makes that
lambda unchoosable; the full solve keeps one float64 fallback. The products in `grid_gram` run with TF32 off, those in
`gmm` with TF32 on; both switches are restored afterwards.

Source: scripts/closed_form_bench.py (AutoSeg), `ridge_fit` (1437-2239) and the functions it calls, for the reported
configuration --readout ridge_all --gram_mode grid --gram_dtype bfloat16 --lam_search newton --loo_std fold
--precision single --ex_metric dataset --blend --primary blend1_ex. Line references (in parentheses) are to that file.
Line references point to the authors' research code, which is not public.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

ACC = torch.float32            # dtype of the sums and the solves (--precision single; 309)
LAM_MIN, LAM_MAX = 1e-4, 100.0  # Newton's bounds: the range of --ridge_lams 1e-4,...,100 (195, 724)
LAM_START = 1.0                # Newton's start, also at a refit (--lam_start, --refit_lam_start cold; 203, 1762)
CHUNK_PX = 65536               # foreground rows per product (--chunk_px; 251)
SHIFT_PX = 4096                # pixels of the first support image that fix the column shift (1537)
HOST_BANK_ELEMENTS = 50_000_000  # held-out priors larger than this wait on the host (1665)
BLOCKS = ("coarse", "fine", "bank")   # the column blocks, in the order of the weights (--blocks default)


def check_blocks(blocks) -> tuple:
    """A non-empty subset of BLOCKS, as a tuple in the weights' order (a comma string is accepted). (624-626)"""
    if isinstance(blocks, str):
        blocks = [b.strip() for b in blocks.split(",") if b.strip()]
    want = set(blocks)
    if not want or not want <= set(BLOCKS):
        raise ValueError(f"blocks must be a non-empty subset of {BLOCKS}, got {blocks!r}")
    return tuple(b for b in BLOCKS if b in want)


def block_selection(blk_dims, blocks, d, device):
    """The kept columns of an ablation: the indices of the blocks in `blocks` and of the constant, and the full width
    d_full the weights are scattered back to (removed blocks get weight 0). blk_dims: (coarse, fine, priors) widths,
    0 for a block the inputs do not carry. (1692-1701)"""
    want, parts_, o_ = set(blocks), [], 0
    for nm_, c_ in zip(BLOCKS, blk_dims):
        if nm_ in want:
            parts_.append(torch.arange(o_, o_ + c_, device=device))
        o_ += c_
    if o_ + 1 != d:
        raise RuntimeError(f"blocks: column blocks {blk_dims} do not add up to the Gram size {d}")
    return torch.cat(parts_ + [torch.tensor([o_], device=device)]), o_ + 1


# ---------------------------------------------------------------------------------------------------------------------
# rows, interpolation and the Gram matrices

def gather(xc, fine, bank, hw, idx):
    """Rows [coarse | fine | priors] at the flat pixel indices idx: the grids sampled bilinearly
    (grid_sample, align_corners=False, padding_mode="border", which equals the F.interpolate of the prediction).
    (509-518)"""
    h, w = hw
    g = torch.stack([2 * ((idx % w).float() + 0.5) / w - 1,
                     2 * ((idx // w).float() + 0.5) / h - 1], -1)[None, None]
    out = [F.grid_sample(t, g, mode="bilinear", align_corners=False, padding_mode="border")[0, :, 0].T
           for t in (xc, fine) if t is not None]
    out.append(bank.reshape(bank.shape[1], -1)[:, idx].T)
    return torch.cat(out, 1)


def interp_matrix(n_px, n_grid, device):
    """[n_px, n_grid] weights of grid_sample(bilinear, align_corners=False, padding_mode='border') along one axis:
    pixel i reads grid coordinate (i + 0.5) n_grid / n_px - 0.5, clamped to [0, n_grid - 1]. (1015-1027)"""
    i = torch.arange(n_px, device=device, dtype=torch.float64)
    gco = ((i + 0.5) * n_grid / n_px - 0.5).clamp(0, n_grid - 1)
    i0 = gco.floor().long().clamp(max=n_grid - 1)
    fr = gco - i0
    i1 = (i0 + 1).clamp(max=n_grid - 1)
    U = torch.zeros(n_px, n_grid, dtype=torch.float64, device=device)
    r = torch.arange(n_px, device=device)
    U.index_put_((r, i0), 1 - fr, accumulate=True)
    U.index_put_((r, i1), fr, accumulate=True)
    return U.float()


def grid_gram(grids, bankn, hw, r0, r1, c0, c1):
    """Exact Gram matrix of the rows [U_1 D_1, ..., U_m D_m, B, 1] over the pixel rectangle [r0:r1, c0:c1], from the
    grids. grids: [h, w, C] each, already shifted and scaled; bankn: [Cb, H, W], shifted and scaled. float32 products
    with TF32 off. (1029-1071)"""
    dev = bankn.device
    allow = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    H, W = hw
    Uy = [interp_matrix(H, g.shape[0], dev)[r0:r1] for g in grids]
    Ux = [interp_matrix(W, g.shape[1], dev)[c0:c1] for g in grids]
    Bn = bankn[:, r0:r1, c0:c1]
    cb, m = Bn.shape[0], len(grids)
    dims = [g.shape[2] for g in grids] + [cb, 1]
    o = np.cumsum([0] + dims)
    Gm = torch.zeros(o[-1], o[-1], dtype=ACC, device=dev)
    flat = [g.reshape(-1, g.shape[2]) for g in grids]
    for i in range(m):
        for j in range(i, m):
            Z = torch.einsum("ab,bwc->awc", Uy[i].T @ Uy[j], grids[j])
            Z = torch.einsum("xw,awc->axc", Ux[i].T @ Ux[j], Z)           # [h_i, w_i, C_j]
            blk = (flat[i].T @ Z.reshape(-1, Z.shape[2])).to(ACC)
            Gm[o[i]:o[i + 1], o[j]:o[j + 1]] = blk
            Gm[o[j]:o[j + 1], o[i]:o[i + 1]] = blk.T
            del Z
        R = torch.einsum("ha,chw->caw", Uy[i], Bn)
        R = torch.einsum("wb,caw->cab", Ux[i], R)                         # U_i' B: [Cb, h_i, w_i]
        blk = (flat[i].T @ R.reshape(cb, -1).T).to(ACC)
        Gm[o[i]:o[i + 1], o[m]:o[m + 1]] = blk
        Gm[o[m]:o[m + 1], o[i]:o[i + 1]] = blk.T
        v = torch.einsum("a,b,abc->c", Uy[i].sum(0), Ux[i].sum(0), grids[i]).to(ACC)
        Gm[o[i]:o[i + 1], -1] = v
        Gm[-1, o[i]:o[i + 1]] = v
    Bf = Bn.reshape(cb, -1)
    bbm = torch.zeros(cb, cb, dtype=ACC, device=dev)
    for c0_ in range(0, Bf.shape[1], 1 << 20):
        ch = Bf[:, c0_:c0_ + (1 << 20)]
        bbm += (ch @ ch.T).to(ACC)
    Gm[o[m]:o[m + 1], o[m]:o[m + 1]] = bbm
    sb = Bf.to(ACC).sum(1)
    Gm[o[m]:o[m + 1], -1] = sb
    Gm[-1, o[m]:o[m + 1]] = sb
    Gm[-1, -1] = float(Bf.shape[1])
    torch.backends.cuda.matmul.allow_tf32 = allow
    return Gm


def gmm(X):
    """X'X with float32 output: rows rounded to bfloat16 on the way in, TF32 products, the upper block triangle in 8
    column blocks (edges round(i d / 8)) and its transposes. (1476-1498, --gram_dtype bfloat16)"""
    X = X.to(torch.bfloat16).float()
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    d_ = X.shape[1]
    if d_ < 1024:
        out = X.T @ X
    else:
        e_ = [round(i_ * d_ / 8) for i_ in range(9)]
        out = torch.empty(d_, d_, device=X.device, dtype=X.dtype)
        for i_ in range(8):
            blk = X[:, e_[i_]:e_[i_ + 1]].T @ X[:, e_[i_]:]
            out[e_[i_]:e_[i_ + 1], e_[i_]:] = blk
            out[e_[i_ + 1]:, e_[i_]:e_[i_ + 1]] = blk[:, e_[i_ + 1] - e_[i_]:].T
    torch.backends.cuda.matmul.allow_tf32 = prev
    return out


def rows(xc, fine, bank, hw, idx, pc, ps):
    """Shifted and scaled rows with the constant column. (1544-1546, `rowsof`)"""
    return torch.cat([((gather(xc, fine, bank, hw, idx).float() - pc) / ps),
                      torch.ones(len(idx), 1, device=bank.device)], 1)


def shift_sample(hw, rng) -> np.ndarray:
    """The 4096 pixel indices of the first support image that fix the column shift; the readout's only draw from
    the support RNG. (1536-1537)"""
    n0 = hw[0] * hw[1]
    return rng.choice(n0, min(n0, SHIFT_PX), replace=False)


def column_shift(xc, fine, bank, hw, i0):
    """Fixed per-column shift and scale (mean and UNBIASED standard deviation over the sampled pixels; below 1e-6 the
    scale is 1). Any fixed affine map is exact here; the solution is mapped back. (1538-1540)"""
    i0 = torch.from_numpy(np.asarray(i0)).to(bank.device)
    x0 = gather(xc, fine, bank, hw, i0).float()
    pc, ps_ = x0.mean(0), x0.std(0)
    ps_ = torch.where(ps_ < 1e-6, torch.ones_like(ps_), ps_)
    return pc, ps_


def label_tensor(label_map, hw, device):
    """The binary mask as a float32 [H, W] tensor (nearest resize only if the shapes differ). (1547-1549)"""
    lab2 = torch.from_numpy((np.asarray(label_map) > 0).astype(np.float32)).to(device)
    if lab2.shape != torch.Size(hw):
        lab2 = F.interpolate(lab2[None, None], size=hw, mode="nearest")[0, 0]
    return lab2


def image_sums(xc, fine, bank, hw, lab2, pc, ps, quadrants: bool):
    """One support image's raw sums per held-out group (the whole image, or its four quadrants at K = 1): Gram G over
    all pixels (exact on the grids), foreground Gram Gf, foreground sum bb (unrounded float32 rows), pixel and
    foreground counts. (1575-1623, --gram_mode grid)"""
    dev = bank.device
    lab = lab2.reshape(-1) > 0.5
    h, w = hw
    npx = h * w
    gid = None
    if quadrants:
        ar = torch.arange(npx, device=dev)
        gid = ((ar // w) >= h // 2).long() * 2 + ((ar % w) >= w // 2).long()
    ng = 4 if quadrants else 1
    G, Gf, bb = [None] * ng, [None] * ng, [None] * ng
    nn_, nf = [0.0] * ng, [0.0] * ng
    C0 = xc.shape[1]
    grids = [((xc[0].permute(1, 2, 0).float() - pc[:C0]) / ps[:C0]).contiguous()]
    if fine is not None:
        C1 = fine.shape[1]
        grids.append(((fine[0].permute(1, 2, 0).float() - pc[C0:C0 + C1]) / ps[C0:C0 + C1]).contiguous())
    cbk = bank.shape[1]
    bankn = (bank[0].float() - pc[-cbk:, None, None]) / ps[-cbk:, None, None]
    rects = [(0, 0, h, 0, w)] if gid is None else \
        [(q, (q // 2) * (h // 2), h if q // 2 else h // 2, (q % 2) * (w // 2), w if q % 2 else w // 2)
         for q in range(4)]
    for g, ra, rb, ca, cb_ in rects:
        gg = grid_gram(grids, bankn, hw, ra, rb, ca, cb_)
        G[g] = gg if G[g] is None else G[g] + gg
        nn_[g] += float((rb - ra) * (cb_ - ca))
    del grids, bankn
    fgi = torch.nonzero(lab)[:, 0]
    for c0 in range(0, len(fgi), CHUNK_PX):
        idx = fgi[c0:c0 + CHUNK_PX]
        A = rows(xc, fine, bank, hw, idx, pc, ps).to(torch.float32)
        for g, m in ([(0, None)] if gid is None else [(q, gid[idx] == q) for q in range(4)]):
            Af = A if m is None else A[m]
            if Af.shape[0] == 0:
                continue
            gf_ = gmm(Af).to(ACC)
            b_ = Af.to(ACC).sum(0)
            Gf[g] = gf_ if Gf[g] is None else Gf[g] + gf_
            bb[g] = b_ if bb[g] is None else bb[g] + b_
            nf[g] += float(Af.shape[0])
        del A
    return [dict(G=G[g], Gf=Gf[g], bb=bb[g], nn=nn_[g], nf=nf[g]) for g in range(ng)]


# ---------------------------------------------------------------------------------------------------------------------
# standardisation and solves

def tstd(Gw, bw, mu_, sd_, oo_=None):
    """A Gram matrix and a target sum in standardised coordinates, x -> (x - mu) / sd on every column but the
    constant, in O(d^2) and O(d). (1716-1727)"""
    if oo_ is None:
        oo_ = torch.outer(sd_, sd_)
    gx, n_ = Gw[:-1, -1], Gw[-1, -1]
    out = torch.empty_like(Gw)
    out[:-1, :-1] = (Gw[:-1, :-1] - torch.outer(gx, mu_) - torch.outer(mu_, gx) + n_ * torch.outer(mu_, mu_)) / oo_
    out[:-1, -1] = (gx - n_ * mu_) / sd_
    out[-1, :-1] = out[:-1, -1]
    out[-1, -1] = n_
    return out, torch.cat([(bw[:-1] - mu_ * bw[-1]) / sd_, bw[-1:]])


def tback(wst, mu_, sd_):
    """Weights on standardised columns -> weights on the shifted, scaled columns. (1729-1733)"""
    wx = wst[:-1] / sd_
    return torch.cat([wx, (wst[-1] - (mu_ * wx).sum())[None]])


def unshift(wv, pc, ps):
    """Weights on the shifted, scaled columns -> weights on the raw columns that `logit_map` reads. (1735-1742)"""
    wx = wv[:-1] / ps.to(wv.dtype)
    return torch.cat([wx, (wv[-1] - (pc.to(wv.dtype) * wx).sum())[None]])


def chol_solve(M, r, counters=None):
    """M w = r by a float32 Cholesky factor; one float64 fallback when float32 is not definite (counted); None if
    neither is. (697-706, --precision single)"""
    L, info = torch.linalg.cholesky_ex(M.float())
    if not int(info):
        return torch.cholesky_solve(r.float()[:, None], L)[:, 0]
    if counters is not None:
        counters["single_fallbacks"] = counters.get("single_fallbacks", 0) + 1
    L, info = torch.linalg.cholesky_ex(M.double())
    return None if int(info) else torch.cholesky_solve(r.double()[:, None], L)[:, 0].float()


def totals(groups):
    """The per-group sums as lists, empty foreground filled with zeros, and the support totals. (1683-1711)"""
    G = [x["G"] for x in groups]
    Gf = [x["Gf"] for x in groups]
    bb = [x["bb"] for x in groups]
    nn_ = [x["nn"] for x in groups]
    nf = [x["nf"] for x in groups]
    for g in range(len(groups)):                         # a group without foreground pixels (empty mask)
        if Gf[g] is None:
            Gf[g] = torch.zeros_like(G[g])
            bb[g] = torch.zeros(G[g].shape[0], dtype=ACC, device=G[g].device)
    Gt, nt, nft = sum(G), sum(nn_), sum(nf)
    Gft, bbt = sum(Gf), sum(bb)
    return dict(G=G, Gf=Gf, bb=bb, nn=nn_, nf=nf, Gt=Gt, nt=nt, nft=nft, Gft=Gft, bbt=bbt)


def fold_pieces(tot, vname):
    """Per held-out group: the training system of the fold standardised and weighted from the OTHER groups only, and
    the held-out group's own sums in that fold's coordinates and weights. B and r are the training Gram and target
    over their weight; Sg, bg, ng the held-out Gram, target and weight; c0 = q_fg n_fg (the sum of q y^2).
    Returns (folds, fold_stats) with fold_stats[g] = (mu_g, sd_g). (1765-1781, --loo_std fold)"""
    G, Gf, bb, nn_, nf = tot["G"], tot["Gf"], tot["bb"], tot["nn"], tot["nf"]
    Gt, nt, nft, Gft, bbt = tot["Gt"], tot["nt"], tot["nft"], tot["Gft"], tot["bbt"]
    folds, fold_tb = [], []
    for g in range(len(G)):
        Gtr, ntr_, nftr = Gt - G[g], nt - nn_[g], nft - nf[g]
        mu_g = Gtr[-1, :-1] / ntr_
        sd_g = torch.sqrt((torch.diagonal(Gtr)[:-1] / ntr_ - mu_g ** 2).clamp_min(0))
        sd_g = torch.where(sd_g < 1e-6, torch.ones_like(sd_g), sd_g)
        f_g = min(max(nftr / ntr_, 1e-6), 1 - 1e-6)
        wfg_g, wbg_g = (1.0, 1.0) if vname == "plain" else (0.5 / f_g, 0.5 / (1 - f_g))
        Gftr, bbtr = Gft - Gf[g], bbt - bb[g]
        Btr, rtr = tstd(wbg_g * (Gtr - Gftr) + wfg_g * Gftr, wfg_g * bbtr, mu_g, sd_g)
        nwtr = wbg_g * (ntr_ - nftr) + wfg_g * nftr
        Sh, bh = tstd(wbg_g * (G[g] - Gf[g]) + wfg_g * Gf[g], wfg_g * bb[g], mu_g, sd_g)
        folds.append(dict(B=Btr / nwtr, r=rtr / nwtr, Sg=Sh, bg=bh,
                          ng=wbg_g * (nn_[g] - nf[g]) + wfg_g * nf[g], c0=wfg_g * nf[g]))
        fold_tb.append((mu_g, sd_g))
    return folds, fold_tb


def heldout_cv(folds, B32, Pd, lam, counters=None):
    """The exact mean leave-one-group-out weighted squared error at one lambda and its first two derivatives in
    lambda: per fold one float32 Cholesky factor of M = B + lambda I gives w = M^-1 r, dw/dlambda = -M^-1 I w and
    d2w/dlambda2 = -2 M^-1 I dw/dlambda, and the held-out error E = (c0 - 2 w'b + w'S w) / n from the held-out sums.
    Returns (CV, dCV/dlambda, d2CV/dlambda2, fold solutions), or None when a fold's factor is not definite (that
    lambda is then unchoosable; counted). (763-816, the body of `ev`)"""
    f = d1 = d2 = 0.0
    sols = []
    for g, fo in enumerate(folds):
        M = B32[g].clone()
        M.diagonal()[:-1] += lam
        L, info = torch.linalg.cholesky_ex(M)
        if int(info):
            if counters is not None:
                counters["single_rejected_evals"] = counters.get("single_rejected_evals", 0) + 1
            return None

        def sv(v, L=L):
            return torch.cholesky_solve(v.float()[:, None], L)[:, 0]
        w = sv(fo["r"])
        w1 = -sv(Pd * w)
        w2 = -2 * sv(Pd * w1)
        Sg, bg, n_g = fo["Sg"], fo["bg"], fo["ng"]
        Sw, Sw1 = Sg @ w, Sg @ w1
        f += (fo["c0"] - 2 * float(w @ bg) + float(w @ Sw)) / n_g
        d1 += (-2 * float(w1 @ bg) + 2 * float(w1 @ Sw)) / n_g
        d2 += (-2 * float(w2 @ bg) + 2 * float(w1 @ Sw1) + 2 * float(w2 @ Sw)) / n_g
        sols.append(w)
    nf_ = len(folds)
    return f / nf_, d1 / nf_, d2 / nf_, sols


def newton_outer(ev, start):
    """The safeguarded Newton iteration on t = log(lambda) for ev(t) -> (lambda, CV, CV', CV'') in t, or None: start
    at `start` (a larger lambda while not definite), step -CV'/CV'' (one decade downhill where CV'' <= 0), at most one
    decade, halved up to six times while not definite or CV rises, inside [LAM_MIN, LAM_MAX]; stop when the step is
    under 0.01 in t or after 30 steps. Returns ([(lambda, {"mse": (-CV, 0.5)}) per evaluation], evaluations). (722-751)"""
    tlo, thi = float(np.log(LAM_MIN)), float(np.log(LAM_MAX))
    t = float(np.clip(np.log(start), tlo, thi))
    cur, n = ev(t), 1
    while cur is None and t < thi - 1e-9:                # not definite at the start: a larger lambda
        t = min(thi, t + float(np.log(10.0)))
        cur, n = ev(t), n + 1
    if cur is None:
        raise RuntimeError("newton: no definite system up to the largest lambda")
    evals, step_max = [cur], float(np.log(10.0))
    for _ in range(30):
        _, f, g1, g2 = cur
        step = -g1 / g2 if g2 > 0 else -float(np.sign(g1)) * step_max
        tn = float(np.clip(t + float(np.clip(step, -step_max, step_max)), tlo, thi))
        nxt = None
        for _h in range(6):                              # halve while not definite or CV rises
            if abs(tn - t) < 0.01:
                break
            nxt, n = ev(tn), n + 1
            if nxt is not None:
                evals.append(nxt)
                if nxt[1] <= f:
                    break
            nxt = None
            tn = t + (tn - t) / 2
        if nxt is None:
            break
        t, cur = tn, nxt
    # (lambda, {"mse": (score, cut)}) as in the research code; the cut, always 0.5, becomes the readout's tau (`solve`)
    return [(e[0], {"mse": (-e[1], 0.5)}) for e in evals], n


def newton_lam_folds(folds, Pd, fsol, start, counters=None):
    """lambda by Newton's method on log(lambda) for the exact leave-one-group-out error; every evaluated fold solution
    goes to fsol[(lambda, group)]. (753-818, --precision single)"""
    B32 = [fo["B"].float() for fo in folds]

    def ev(t):
        lam = float(np.exp(t))
        out = heldout_cv(folds, B32, Pd, lam, counters)
        if out is None:
            return None
        f, d1, d2, sols = out
        for g, w in enumerate(sols):
            fsol[(lam, g)] = w
        return lam, f, lam * d1, lam * lam * d2 + lam * d1      # CV, dCV/dt, d2CV/dt2 at t = log(lambda)
    return newton_outer(ev, start)


# ---------------------------------------------------------------------------------------------------------------------
# score maps and the held-out metric

def logit_map(xc, fine, bank, hw, w, as_tensor=False):
    """s = U(F_c w_c) + U(F_m w_m) + B w_B + w_0: each grid reduced to one channel by its weights, that channel
    interpolated bilinearly to the image; the 4132-map stack is never formed. (534-549)"""
    dev = bank.device
    o, i = 0, 0
    w = torch.as_tensor(w, dtype=torch.float32, device=dev)
    for t in (xc, fine):
        if t is None:
            continue
        c = t.shape[1]
        m = (t * w[i:i + c].view(1, c, 1, 1)).sum(1, keepdim=True)
        o = o + F.interpolate(m, size=hw, mode="bilinear", align_corners=False)
        i += c
    cb = bank.shape[1]
    o = o + (bank * w[i:i + cb].view(1, cb, 1, 1)).sum(1, keepdim=True)
    if as_tensor:
        return o + float(w[-1])
    return (o[0, 0] + float(w[-1])).cpu().numpy()


# skimage's 2-D skeletonize (Zhang-Suen, `_fast_skeletonize`, skimage 0.26; BSD-3-Clause, see LICENSE) on the device: the same 256-entry table, the
# same neighbour code (1 nw, 2 n, 4 ne, 8 e, 16 se, 32 s, 64 sw, 128 w), two passes per iteration, each reading the state
# at the start of the pass, a zero border. (1079-1092)
_ZS_LUT_VALUES = [
    0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 3, 0, 3, 3,
    0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0,
    0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 3, 1, 3, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0]
_ZS_TABLES = {}


def _zs_tables(device):
    key = str(device)
    if key not in _ZS_TABLES:
        _ZS_TABLES[key] = (torch.tensor(_ZS_LUT_VALUES, dtype=torch.long, device=device),
                           torch.tensor([[1., 2., 4.], [128., 0., 8.], [64., 32., 16.]], device=device).view(1, 1, 3, 3))
    return _ZS_TABLES[key]


def zs_skel(m):
    """[B, H, W] bool -> skeletons, exactly skimage.morphology.skeletonize per slice. Only border pixels are visited:
    each pass reads the candidates' codes from the state at its start, removes, and adds the removed pixels' still-set
    neighbours to the candidates. (1094-1124)"""
    dev = m.device
    _ZS_LUT, _ZS_K = _zs_tables(dev)
    B, H, W = m.shape
    Wp = W + 2
    x = F.pad(m.to(torch.uint8), (1, 1, 1, 1)).reshape(-1).clone()
    offs = torch.tensor([-Wp - 1, -Wp, -Wp + 1, 1, Wp + 1, Wp, Wp - 1, -1], device=dev)
    wts = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=dev)
    code0 = F.conv2d(F.pad(m.to(torch.float32), (1, 1, 1, 1))[:, None], _ZS_K, padding=1)[:, 0].reshape(-1)
    C = torch.nonzero((x == 1) & (code0 != 255))[:, 0]
    del code0
    while True:
        removed_any = False
        for first in (True, False):
            if C.numel() == 0:
                break
            code = (x[C[:, None] + offs[None, :]].long() * wts[None, :]).sum(1)
            v = _ZS_LUT[code]
            rem = (v == 3) | ((v == 1) if first else (v == 2))
            if bool(rem.any()):
                removed_any = True
                R = C[rem]
                x[R] = 0
                N = (R[:, None] + offs[None, :]).reshape(-1)
                N = N[x[N] == 1]
                C = torch.unique(torch.cat([C[~rem], N]))
        if not removed_any:
            break
    return x.reshape(B, H + 2, Wp)[:, 1:-1, 1:-1] > 0


def metric_masks(P, gt, sgt, metric):
    """The held-out metric of a batch of masks P [B, H, W] against gt (and its skeleton sgt): centreline Dice
    ("cldice") or foreground IoU ("fg_iou"), with the edge cases of metrics.cldice / foreground_iou. (1260-1282)"""
    dev = P.device
    out = []
    for i in range(0, P.shape[0], 16):
        Pc = P[i:i + 16]
        if metric == "cldice":
            SP = zs_skel(Pc)
            ps, gs_ = Pc.flatten(1).any(1), bool(gt.any())
            tprec = (SP & gt).flatten(1).sum(1).double() / SP.flatten(1).sum(1).clamp_min(1).double()
            tsens = ((sgt[None] & Pc).flatten(1).sum(1).double() / max(float(sgt.sum()), 1.0)) \
                if bool(sgt.any()) else torch.zeros(Pc.shape[0], device=dev, dtype=torch.float64)
            tprec = torch.where(SP.flatten(1).any(1), tprec, torch.zeros_like(tprec))
            c = torch.where(tprec + tsens > 0, 2 * tprec * tsens / (tprec + tsens).clamp_min(1e-300),
                            torch.zeros_like(tprec))
            c = torch.where(~ps & (not gs_), torch.ones_like(c), c)
            c = torch.where(ps ^ gs_, torch.zeros_like(c), c)
            out.append(c)
        else:
            inter = (Pc & gt).flatten(1).sum(1).double()
            union = (Pc | gt).flatten(1).sum(1).double()
            out.append(torch.where(union > 0, inter / union.clamp_min(1), torch.ones_like(union)))
    return torch.cat(out).cpu().numpy()


def choose_blend(Mp_, Mb_, metric, pm, bm):
    """The blend weight a of the plain and the balanced readout, cut at 1/2, by the exact held-out metric.

    Mp_, Mb_: per held-out image (score map, mask, mask skeleton or None) of the plain and the balanced readout;
    pm, bm: the two readouts (dicts with lam and w). Returns (blend_ex, blend1_ex): the best a, and the reported choice
    under the one-standard-error rule (a = 1, the plain readout at 1/2, unless the mean paired gain over the held-out
    images exceeds one standard error of it). (2118-2175, --blend_rule grid part)"""
    if len(Mp_) != len(Mb_) or any(not torch.equal(p_[1], b_[1]) for p_, b_ in zip(Mp_, Mb_)):
        raise RuntimeError("blend: the two weightings' held-out maps are not the same images")
    if not all(bool(torch.isfinite(p_[0]).all()) and bool(torch.isfinite(b_[0]).all()) for p_, b_ in zip(Mp_, Mb_)):
        raise RuntimeError("blend: a held-out score map holds a non-finite value")
    bc_ = {}

    def blend_scores(avs):                               # per held-out image, the metric of each blend cut at 1/2;
        avs = [round(float(x), 3) for x in avs]          # the new a of one call are batched per image
        new_ = [x for x in dict.fromkeys(avs) if x not in bc_]
        if new_:
            per_ = []
            for p_, b_ in zip(Mp_, Mb_):
                av_t = torch.tensor(new_, device=p_[0].device, dtype=p_[0].dtype)[:, None, None]
                Pm = (av_t * p_[0][None] + (1 - av_t) * b_[0][None]) > 0.5
                per_.append(metric_masks(Pm, p_[1], p_[2], metric))
            per_ = np.stack(per_)                        # [images, new a]
            for j_, x in enumerate(new_):
                bc_[x] = per_[:, j_]
        return [bc_[x] for x in avs]

    w_b = lambda av: (av * np.asarray(pm["w"], np.float64) + (1 - av) * np.asarray(bm["w"], np.float64)).astype(np.float32)
    # a on a 0.2 grid, then +-0.1 and +-0.05 around the best; ties go to the larger a
    blend_scores(np.round(np.arange(0.0, 1.0001, 0.2), 3))
    best_ = max(sorted(bc_, reverse=True), key=lambda x: bc_[x].mean())
    blend_scores([best_ + dv for dv in (-0.1, -0.05, 0.05, 0.1) if 0.0 <= round(best_ + dv, 3) <= 1.0])
    best_ = max(sorted(bc_, reverse=True), key=lambda x: bc_[x].mean())
    blend_ex = dict(variant=f"ba{int(round(100 * best_)):03d}", a=best_, lam=pm["lam"], lam_bal=bm["lam"],
                    tau=0.5, heldout=float(bc_[best_].mean()), w=w_b(best_), per_img=bc_[best_],
                    per_img_half=bc_[1.0])
    d_ = bc_[best_] - bc_[1.0]
    se_ = float(d_.std(ddof=1) / np.sqrt(len(d_))) if len(d_) > 1 else 0.0
    if float(d_.mean()) > se_:
        blend1_ex = dict(blend_ex, gain=float(d_.mean()), se=se_)
    else:
        blend1_ex = dict(variant="plain", a=1.0, lam=pm["lam"], tau=0.5, heldout=float(bc_[1.0].mean()),
                         w=pm["w"], per_img=bc_[1.0], per_img_half=bc_[1.0], gain=float(d_.mean()), se=se_)
    return blend_ex, blend1_ex


# ---------------------------------------------------------------------------------------------------------------------
# the fit

def ridge_fit(features, label_maps, metric, rng=None, shift_idx=None, state=None, counters=None, blocks=BLOCKS):
    """Fit the readout on K support images.

    features(i) -> (xc [1, Cc, hc, wc], fine [1, Cm, hm, wm] or None, bank [1, Cb, H, W], (H, W)): the frozen inputs of
    support image i, computed when needed (only for images not already in `state`). label_maps: the K masks (> 0 is
    foreground). metric: "fg_iou" or "cldice", the held-out metric of the blend. rng: the support RNG, drawn once for
    the column shift (`shift_sample`), or shift_idx: those indices given (a cold refit replays the original draw).
    state: the state returned by an earlier fit whose images are the first images of this support (an exact refit:
    their sums and held-out inputs are reused, the new images' added). counters: a dict that counts float64 fallbacks
    and rejected Newton evaluations. blocks: the column blocks the readout may use (default all three; a subset is the
    ablation of Table 2's input rows, solved on the kept blocks only, see `block_selection`).

    Returns (res, state). res["blend1_ex"] is the reported readout (keys lam, lam_bal, a, tau = 0.5, w, heldout,
    variant, gain, se); res["blend_ex"] the best blend without the one-SE rule; res["plain_mse"] and res["bal_mse"]
    the two weightings at 1/2. w is a float32 vector on the raw columns [coarse | fine | priors | constant].
    (1437-2239)"""
    K = len(label_maps)
    blocks = check_blocks(blocks)
    old = [] if state is None else state["imgs"]
    if len(old) > K or (old and K < 2):
        raise RuntimeError("refit state: the kept images must be the first images of a support of K >= 2")
    groups, held = [], []
    pc = ps_ = None
    i0 = None
    blk_dims = None
    if old:
        pc, ps_, i0, blk_dims = state["pc"], state["ps"], state["i0"], state["blk_dims"]
    for gi in range(K):
        if gi < len(old):                                # refit: this image's sums and held-out inputs are kept
            groups.extend(old[gi]["groups"])
            kp_ = old[gi]["held"]
            if metric == "cldice" and kp_["sgt"] is None:   # kept under another metric: its skeleton is now needed
                kp_["sgt"] = zs_skel(kp_["gt"][None])[0]
            held.append(kp_)
            continue
        xc, fine, bank, hw = features(gi)
        dev = bank.device
        blk_dims = (xc.shape[1], 0 if fine is None else fine.shape[1], bank.shape[1])   # (1529)
        if pc is None:
            i0 = np.asarray(shift_idx) if shift_idx is not None else shift_sample(hw, rng)
            pc, ps_ = column_shift(xc, fine, bank, hw, i0)
        lab2 = label_tensor(label_maps[gi], hw, dev)
        groups.extend(image_sums(xc, fine, bank, hw, lab2, pc, ps_, quadrants=K < 2))
        gtt = lab2 > 0.5
        sgt = zs_skel(gtt[None])[0] if metric == "cldice" else None
        held.append(dict(xc=xc, fine=None if fine is None else fine.to(torch.bfloat16),
                         bank=bank.to("cpu", torch.float32) if bank.numel() > HOST_BANK_ELEMENTS else bank,
                         hw=hw, gt=gtt, sgt=sgt))
        del xc, fine, bank
    sel = None
    if blocks != BLOCKS:                                 # ablation: the ridge on the kept column blocks only (1693)
        sel = block_selection(blk_dims, blocks, groups[0]["G"].shape[0], groups[0]["G"].device)
    res = solve(groups, held, metric, K, pc, ps_, counters, sel=sel)
    new_state = dict(pc=pc, ps=ps_, i0=i0, blk_dims=blk_dims, incremental=K >= 2,
                     imgs=[dict(groups=[groups[g]], held=held[g]) for g in range(K)] if K >= 2 else [])
    return res, new_state


def solve(groups, held, metric, K, pc, ps, counters=None, sel=None):
    """Standardisation, both weightings' folds, lambda by Newton, the full solves, the held-out maps and the blend,
    from the per-group sums. sel: (kept column indices, full width) of a `blocks` ablation, or None. (1683-2175)"""
    if sel is not None:                                  # every group's sums restricted to the kept columns (1702-1705)
        sel_idx, d_full = sel
        groups = [dict(x, G=x["G"][sel_idx][:, sel_idx],
                       Gf=None if x["Gf"] is None else x["Gf"][sel_idx][:, sel_idx],
                       bb=None if x["bb"] is None else x["bb"][sel_idx]) for x in groups]
        pc_u, ps_u = pc[sel_idx[:-1]], ps[sel_idx[:-1]]
    else:
        pc_u, ps_u = pc, ps

    def unsh(wv):                                        # back to the raw columns that `logit_map` reads (1735-1742)
        out_ = unshift(wv, pc_u, ps_u)
        if sel is None:
            return out_
        full_ = torch.zeros(d_full, dtype=out_.dtype, device=out_.device)   # removed blocks: weight 0
        full_[sel_idx] = out_
        return full_
    tot = totals(groups)
    Gt, nt, nft = tot["Gt"], tot["nt"], tot["nft"]
    G, Gf, bb, nn_, nf = tot["G"], tot["Gf"], tot["bb"], tot["nn"], tot["nf"]
    ng = len(G)
    dev = Gt.device
    d1 = Gt.shape[0]
    mu = Gt[-1, :-1] / nt
    sd = torch.sqrt((torch.diagonal(Gt)[:-1] / nt - mu ** 2).clamp_min(0))
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
    oo = torch.outer(sd, sd)
    Pd = torch.ones(d1, dtype=ACC, device=dev)
    Pd[-1] = 0.0
    f1 = min(max(nft / nt, 1e-6), 1 - 1e-6)
    weights = {"plain": (1.0, 1.0), "bal": (0.5 / f1, 0.5 / (1 - f1))}
    res, kept_maps = {}, {}
    for vname in ("plain", "bal"):
        wfg, wbg = weights[vname]
        S = []
        for g in range(ng):                              # the whole support, globally standardised: the full solve
            Gw = wbg * (G[g] - Gf[g]) + wfg * Gf[g]
            bw = wfg * bb[g]                             # b = the sum of the foreground rows
            S.append(tstd(Gw, bw, mu, sd, oo) + (wbg * (nn_[g] - nf[g]) + wfg * nf[g],))
        Gs = sum(x[0] for x in S); bs = sum(x[1] for x in S); ns = sum(x[2] for x in S)
        fsol = {}                                        # (lambda, group) -> fold solution
        folds, fold_tb = fold_pieces(tot, vname)
        table, n_lam = newton_lam_folds(folds, Pd, fsol, LAM_START, counters)
        lam_r, crit_r = max(table, key=lambda t: (t[1]["mse"][0], t[0]))   # ties to the larger lambda
        sol_f = chol_solve(Gs / ns + torch.diag(lam_r * Pd), bs / ns, counters)
        if sol_f is None:
            raise RuntimeError(f"ridge: the full system is not definite at lambda {lam_r} ({vname})")
        # tau = crit_r["mse"][1] is the cut 0.5 carried through the Newton table (`newton_outer`)
        res[f"{vname}_mse"] = dict(lam=lam_r, tau=crit_r["mse"][1], heldout=crit_r["mse"][0],
                                   w=unsh(tback(sol_f, mu, sd)).float().cpu().numpy(),
                                   newton=[(l_, -c_["mse"][0]) for l_, c_ in table], newton_evals=n_lam)
        # the held-out map of every held-out group at the chosen lambda, from its fold's solution and statistics
        maps_ = []
        for g in range(ng):
            sol = fsol.get((lam_r, g))                   # solved while lambda was chosen
            if sol is None:
                sol = chol_solve(folds[g]["B"] + torch.diag(lam_r * Pd), folds[g]["r"], counters)
            wg = unsh(tback(sol, *fold_tb[g])).float()
            ig = g if K >= 2 else 0
            kp = held[ig]
            L = logit_map(kp["xc"], None if kp["fine"] is None else kp["fine"].float(),
                          kp["bank"].to(dev), kp["hw"], wg, as_tensor=True)[0, 0]
            h_, w_ = kp["hw"]
            if K >= 2:
                rr = (slice(0, h_), slice(0, w_))
            else:
                rr = (slice((g // 2) * (h_ // 2), h_ if g // 2 else h_ // 2),
                      slice((g % 2) * (w_ // 2), w_ if g % 2 else w_ // 2))
            gt_ = kp["gt"][rr]
            if K < 2 and not gt_.any():
                continue                                 # an empty quadrant says nothing about the cut
            maps_.append((L[rr], gt_, None if kp["sgt"] is None else kp["sgt"][rr]))
        if maps_:
            kept_maps[vname] = maps_
    if not {"plain", "bal"} <= set(kept_maps):
        raise RuntimeError(f"blend: both weightings' held-out maps are needed, got {sorted(kept_maps)}")
    res["blend_ex"], res["blend1_ex"] = choose_blend(kept_maps["plain"], kept_maps["bal"], metric,
                                                     res["plain_mse"], res["bal_mse"])
    return res
