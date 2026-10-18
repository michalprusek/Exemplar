"""Cross-check against the ORIGINAL AutoSeg code, on the CPU (optional: set EXEMPLAR_AUTOSEG_TREE to an AutoSeg tree).

The readout of the reference implementation lives in closures inside `main()` of scripts/closed_form_bench.py. Here the
argument parser of that `main()` is run with the paper's scored command and its nested functions are executed in a
namespace (they then read `a`, `dev`, `ACC`, ... as globals), so the ORIGINAL `ridge_fit`, `fast_fine` and
`zoom_unified_parallel` run unchanged on synthetic inputs, beside the extracted functions. The priors are compared
through the backend AutoSeg builds from the feature string, the coarse view through AutoSeg's Dinov3Encoder with a
stand-in model, the loaders on a small dataset tree. Equality is exact (bit for bit) throughout.

The ablations of Table 2 (`test_ablation_matches_bench`) run end to end through the ORIGINAL pipeline: AutoSeg's
backend built from the arm's feature string, the bench's own `blocks()` (coarse view through AutoSeg's encoder, priors
through the backend, `fast_fine`), the bench's magnification decision (the statement of `main()` that sets it,
`--zoom_fixed` included, executed as it stands) and `ridge_fit` under the arm's `--blocks`, against this package's
FrozenInputs + Exemplar with the same option, on a stand-in ViT. The arms and options are those of
scripts/c117/ridge4_paper.sh.

    EXEMPLAR_AUTOSEG_TREE=/path/to/AutoSeg pytest tests/test_crosscheck_autoseg.py
"""
import ast
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

TREE = os.environ.get("EXEMPLAR_AUTOSEG_TREE", "")
pytestmark = pytest.mark.skipif(not TREE, reason="set EXEMPLAR_AUTOSEG_TREE to an AutoSeg tree")

FEAT = "head_fusion_best_nobank_noloss_nocolor_flat_es_ep500_wd4_do5_mix_noaux_fzu_fam_lf12_natres"
SCORED = ["--readout", "ridge_all", "--gram_mode", "grid", "--gram_dtype", "bfloat16", "--enc_amp", "bf16",
          "--solve_dtype", "float32", "--fast_fine", "--fast_zoom", "--ridge_variants", "plain,bal", "--ex_rule",
          "--primary", "blend1_ex", "--only_primary", "--score_rules", "mse,ex", "--lam_search", "newton",
          "--ex_metric", "dataset", "--loo_std", "fold", "--precision", "single", "--method_name", "ridge",
          "--blend", "--blend_rule", "both", "--seeds", "10", "--k", "8", "--datasets", "drive",
          "--feat_method", FEAT, "--gpu", "0", "--cache", "unused", "--score_dir", "unused"]


@pytest.fixture(scope="module")
def tree():
    for p in (TREE, os.path.join(TREE, "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return TREE


@pytest.fixture(scope="module")
def bench(tree):
    """The nested functions of closed_form_bench.main(), with the scored command's arguments, on the CPU."""
    src = open(os.path.join(tree, "scripts", "closed_form_bench.py")).read()
    main = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    i_parse = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "a")
    names = {"_ZS_LUT", "_ZS_K", "_imean", "_istd", "_fchk", "_exchk", "_dump_tag", "RIDGE_LAMS", "prof", "_rf",
             "featc", "feat_pre"}

    def wanted(n):
        if isinstance(n, ast.FunctionDef):
            return True
        if isinstance(n, ast.Assign):
            tg = n.targets[0]
            ids = [tg.id] if isinstance(tg, ast.Name) else [e.id for e in getattr(tg, "elts", []) if isinstance(e, ast.Name)]
            return bool(set(ids) & names)
        return False
    from active_segmenter.eval.metrics import cldice, foreground_iou
    import closed_form_bench as cfb
    ns = dict(vars(cfb))                                 # the module's globals (imports, POOL, CLDICE, ...)
    ns.update(torch=torch, F=F, dev="cpu", ACC=torch.float32, n_fb=[0], n_rej=[0], pool_ex=ThreadPoolExecutor(4),
              gate_ex=ThreadPoolExecutor(1), cldice_fn=cldice, iou_fn=foreground_iou)
    argv, sys.argv = sys.argv, ["closed_form_bench.py"] + SCORED
    try:
        exec(compile(ast.Module(body=main.body[:i_parse + 1], type_ignores=[]), "cfb_parser", "exec"), ns)
    finally:
        sys.argv = argv
    exec(compile(ast.Module(body=[n for n in main.body[i_parse + 1:] if wanted(n)], type_ignores=[]),
                 "cfb_nested", "exec"), ns)
    # the statement of the seed loop that decides the fine view's magnification (and applies --zoom_fixed)
    zoom_if = [n for n in ast.walk(main) if isinstance(n, ast.If)
               and ast.unparse(n.test) == "getattr(be, 'fine_zoom_auto', False)"]
    assert len(zoom_if) == 1, "the bench's magnification statement moved"
    ns["_zoom_stmt"] = compile(ast.Module(body=zoom_if, type_ignores=[]), "cfb_zoom", "exec")
    return ns


@pytest.fixture
def no_cuda_sync(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)   # the bench synchronises unconditionally


def _support(n, thin=False, noise=0.25):
    from synth import blob_image
    return [blob_image(i, thin=thin, noise=noise) for i in range(n)]


def _same(r1, r2, keys=("plain_mse", "bal_mse", "blend_ex", "blend1_ex")):
    for k in keys:
        assert np.array_equal(r1[k]["w"], r2[k]["w"]), k
        for f in ("lam", "lam_bal", "a", "tau", "variant", "heldout", "gain", "se"):
            assert r1[k].get(f) == r2[k].get(f), (k, f, r1[k].get(f), r2[k].get(f))


# noise 0.25: lambda at its lower bound, a blend found but the one-SE rule keeps a = 1 (fg_iou, K = 4);
# noise 1.2: interior lambdas and a blend a < 1 chosen
@pytest.mark.parametrize("noise", [0.25, 1.2])
@pytest.mark.parametrize("metric,K", [("fg_iou", 4), ("cldice", 4), ("fg_iou", 1), ("cldice", 1)])
def test_ridge_fit_matches_bench(bench, no_cuda_sync, monkeypatch, metric, K, noise):
    from exemplar.readout import ridge_fit
    from synth import features
    pairs = _support(K, thin=metric == "cldice", noise=noise)
    feats = {id(im): features(im) for im, _ in pairs}
    monkeypatch.setitem(bench, "blocks", lambda be, image: feats[id(image)])
    sup = [SimpleNamespace(image=im, label_map=m) for im, m in pairs]
    ref, _ = bench["ridge_fit"](None, sup, np.random.default_rng(5), metric)
    mine, _ = ridge_fit(lambda i: feats[id(pairs[i][0])], [m for _, m in pairs], metric, rng=np.random.default_rng(5))
    _same(ref, mine)


@pytest.mark.parametrize("noise", [0.25, 1.2])
def test_refit_matches_bench(bench, no_cuda_sync, monkeypatch, noise):
    from exemplar.readout import ridge_fit
    from synth import features
    pairs = _support(5, noise=noise)
    feats = {id(im): features(im) for im, _ in pairs}
    monkeypatch.setitem(bench, "blocks", lambda be, image: feats[id(image)])
    sup = [SimpleNamespace(image=im, label_map=m) for im, m in pairs]
    st = {}
    bench["ridge_fit"](None, sup[:4], np.random.default_rng(9), "cldice", state=st)
    ref, _ = bench["ridge_fit"](None, sup, np.random.default_rng(9), "cldice", state=st)
    fn = lambda i: feats[id(pairs[i][0])]
    masks = [m for _, m in pairs]
    _, state = ridge_fit(fn, masks[:4], "cldice", rng=np.random.default_rng(9))
    mine, _ = ridge_fit(fn, masks, "cldice", state=state)
    _same(ref, mine)


@pytest.mark.parametrize("blocks", ["coarse,fine", "bank", "coarse,bank", "fine,bank"])
@pytest.mark.parametrize("metric,K", [("fg_iou", 4), ("cldice", 4), ("fg_iou", 1)])
def test_ridge_fit_blocks_matches_bench(bench, no_cuda_sync, monkeypatch, blocks, metric, K):
    """--blocks: the column selection of the original ridge_fit (sel_idx, the zero-filled weights) against
    readout.ridge_fit(blocks=...), on synthetic inputs; without the fine view both are fed fine = None."""
    from exemplar.readout import ridge_fit
    from synth import features
    pairs = _support(K, thin=metric == "cldice", noise=1.2)
    no_fine = "fine" not in blocks.split(",")
    feats = {}
    for im, _ in pairs:
        xc, fine, bank, hw = features(im)
        feats[id(im)] = (xc, None if no_fine else fine, bank, hw)
    monkeypatch.setitem(bench, "blocks", lambda be, image: feats[id(image)])
    monkeypatch.setattr(bench["a"], "blocks", blocks)
    sup = [SimpleNamespace(image=im, label_map=m) for im, m in pairs]
    ref, _ = bench["ridge_fit"](None, sup, np.random.default_rng(5), metric)
    mine, _ = ridge_fit(lambda i: feats[id(pairs[i][0])], [m for _, m in pairs], metric, rng=np.random.default_rng(5),
                        blocks=blocks)
    _same(ref, mine)
    xc, fine, bank, _ = feats[id(pairs[0][0])]
    widths = {"coarse": xc.shape[1], "fine": 0 if fine is None else fine.shape[1], "bank": bank.shape[1]}
    o = 0
    for name, c in widths.items():                       # removed blocks carry weight 0, kept ones do not
        for k in ("plain_mse", "bal_mse", "blend1_ex"):
            w = mine[k]["w"][o:o + c]
            assert (not np.any(w)) == (name not in blocks.split(",")) or c == 0, (name, k)
        o += c
    assert len(mine["blend1_ex"]["w"]) == o + 1


class _FakeViT(nn.Module):
    """Stands in for DINOv3: patch tokens from a 16-px convolution, 5 prefix tokens, 25 hidden states."""

    def __init__(self, d=8):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.w1 = torch.randn(d, 3, 16, 16, generator=g) * 0.05
        self.w2 = torch.randn(d, 3, 16, 16, generator=g) * 0.05
        self.d = d
        self.calls = []                                  # output_hidden_states of every call

    def forward(self, pixel_values, output_hidden_states=False):
        self.calls.append(bool(output_hidden_states))
        t1 = F.conv2d(pixel_values, self.w1, stride=16).flatten(2).transpose(1, 2)
        t2 = torch.tanh(F.conv2d(pixel_values, self.w2, stride=16)).flatten(2).transpose(1, 2)
        pre = torch.ones(t1.shape[0], 5, self.d)
        last = torch.cat([pre, t1], 1)
        mid = torch.cat([pre * 2, t2], 1)
        return SimpleNamespace(last_hidden_state=last, hidden_states=tuple([mid] * 25))


# layers: the reported reading (-1, 12) and the ablation "last layer only" (AutoSeg: EncoderConfig(layers=()))
LAYER_SETS = [((-1, 12), (-1, 12)), ((-1,), ())]


@pytest.mark.parametrize("layers,cfg_layers", LAYER_SETS)
@pytest.mark.parametrize("hw,z,cap", [((100, 130), 1, 160), ((700, 900), 1, 160), ((700, 900), 1, 30),
                                      ((200, 260), 2, 320), ((400, 500), 3, 30), ((60, 50), 4, 640)])
def test_fine_view_matches_bench(bench, hw, z, cap, layers, cfg_layers):
    from active_segmenter.config import EncoderConfig
    from exemplar.fineview import fine_view
    model = _FakeViT()
    rs = np.random.RandomState(hw[0] + z)
    for image in (rs.randint(0, 255, hw).astype(np.uint8), rs.rand(*hw, 3).astype(np.float32)):
        be = SimpleNamespace(encoder=SimpleNamespace(cfg=EncoderConfig(layers=cfg_layers), model=model),
                             fine_encoder=None, fine_extra=(), fine_zoom_extra=0, bank_only=False, scale_fusion=True,
                             fine_zoom=z, _fine_registered=lambda: z > 1, tile_batch=6, fine_max_grid=cap)
        model.calls.clear()
        ref = bench["fast_fine"](be, image)
        calls_ref, model.calls[:] = list(model.calls), []
        mine = fine_view(model, image, z, cap, "cpu", layers=layers)
        assert ref.shape == mine.shape and torch.equal(ref, mine), (hw, z, cap)
        assert ref.stride() == mine.stride()
        assert ref.shape[1] == 8 * len(layers) and model.calls == calls_ref    # same output_hidden_states


def test_zoom_matches_bench_and_backend(bench, tree):
    import active_segmenter.segment.head_fusion_backend as hfb
    from exemplar.fineview import decide_zoom
    from synth import blob_image, disc_image
    cases = [[blob_image(i)[1] for i in range(4)], [blob_image(i, thin=True)[1] for i in range(3)],
             [disc_image(20)[1], blob_image(1, thin=True)[1]], [disc_image(20)[1], disc_image(30)[1]],
             [np.zeros((40, 40), np.int32)]]
    be = SimpleNamespace(fine_zoom_max=4, fine_zoom_side=8192, fine_zoom_tau=0.8)
    for maps in cases:
        sides = [max(m.shape) for m in maps]
        z_ref = bench["zoom_unified_parallel"](maps, sides, 16, be)
        z_hfb = hfb._support_zoom_unified(maps, sides, 16, 4, 8192, 0.8)[0]
        measures = [{} for _ in maps]
        assert decide_zoom(maps, sides) == decide_zoom(maps, sides, measures) == z_ref == z_hfb
        assert decide_zoom(maps, sides, measures) == z_ref           # served from the kept measurements


def test_priors_match_backend(tree):
    import al_testbed
    from active_segmenter.config import EncoderConfig, RunConfig
    from active_segmenter.segment.base import reset_backend_for_new_support
    from exemplar.priors import make_prior_bank, priors
    enc = SimpleNamespace(enc=SimpleNamespace(cfg=EncoderConfig(layers=(-1, 12))))
    be = al_testbed.make_backend(FEAT, RunConfig(device="cpu", cache_dir="unused"), "cpu", enc=enc, support_k=8)
    reset_backend_for_new_support(be)
    assert (be.fine_zoom_rule, be.fine_zoom_max, be.fine_zoom_side, be.fine_zoom_tau, be.fine_align_zoomed,
            be.tile_batch, be.max_side, be.contrast_norm, be.color_adaptive) == ("unified", 4, 8192, 0.8, True, 6,
                                                                                 10 ** 9, False, False)
    bank = make_prior_bank("cpu")
    rs = np.random.RandomState(0)
    for image in (rs.randint(0, 65535, (90, 70)).astype(np.uint16), (rs.rand(60, 80, 3) * 255).astype(np.uint8),
                  rs.rand(33, 41).astype(np.float32)):
        ref = be._classical(image, inference=True).float()
        mine = priors(image, bank, "cpu").float()
        assert torch.equal(ref, mine)


def _encoders(layers, cfg_layers, model):
    """AutoSeg's Dinov3Encoder and this package's, both around the same stand-in model."""
    from active_segmenter.config import EncoderConfig
    from active_segmenter.encoder.dinov3 import Dinov3Encoder as RefEncoder
    from exemplar.encoder import Dinov3Encoder
    ref = RefEncoder.__new__(RefEncoder)
    ref.cfg, ref.device, ref._torch, ref.model = EncoderConfig(layers=cfg_layers), "cpu", torch, model
    ref._mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    ref._std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    mine = Dinov3Encoder.__new__(Dinov3Encoder)
    mine.device, mine._torch, mine.model, mine._mean, mine._std = "cpu", torch, model, ref._mean, ref._std
    mine.layers = layers
    return ref, mine


@pytest.mark.parametrize("layers,cfg_layers", LAYER_SETS)
def test_coarse_view_matches_encoder(tree, layers, cfg_layers):
    from active_segmenter.config import EncoderConfig
    from active_segmenter.encoder.factory import cache_tag
    from exemplar.encoder import CACHE_TAG, cache_tag as my_cache_tag
    model = _FakeViT()
    ref, mine = _encoders(layers, cfg_layers, model)
    rs = np.random.RandomState(1)
    for image in (rs.randint(0, 255, (300, 200)).astype(np.uint8), rs.rand(50, 90, 4).astype(np.float32)):
        model.calls.clear()
        r = ref._forward_grid(image, 672)
        calls_ref, model.calls[:] = list(model.calls), []
        m = mine.coarse(image)
        assert np.array_equal(r, m) and r.shape[-1] == 8 * len(layers) and model.calls == calls_ref
        assert np.array_equal(ref.extract(image), m)                 # extract: one forward, no tiling, as the bench
    assert cache_tag(EncoderConfig(layers=cfg_layers)) == my_cache_tag(layers)
    assert cache_tag(EncoderConfig(layers=(-1, 12))) == CACHE_TAG


# Table 2's arms (scripts/c117/ridge4_paper.sh): (--blocks, --zoom_fixed, feature string without _lf12?)
ARMS = {"reported": ("coarse,fine,bank", 0, False), "nocls": ("coarse,fine", 0, False), "bankonly": ("bank", 0, False),
        "coarseonly": ("coarse,bank", 0, False), "fineonly": ("fine,bank", 0, False),
        "nolf12": ("coarse,fine,bank", 0, True), "nozoom": ("coarse,fine,bank", 1, False)}


@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("metric,K", [("fg_iou", 3), ("cldice", 1)])
def test_ablation_matches_bench(bench, tree, no_cuda_sync, monkeypatch, arm, metric, K):
    import al_testbed
    from active_segmenter.config import EncoderConfig, RunConfig
    from active_segmenter.segment.base import reset_backend_for_new_support
    from exemplar import Exemplar, FrozenInputs
    from exemplar.readout import logit_map
    from synth import blob_image
    blocks, zoom_fixed, last_only = ARMS[arm]
    feat = FEAT.replace("_lf12", "") if last_only else FEAT
    layers, cfg_layers = LAYER_SETS[1] if last_only else LAYER_SETS[0]
    assert al_testbed.encoder_layers_for_method(feat) == cfg_layers
    model = _FakeViT()
    ref_enc, my_enc = _encoders(layers, cfg_layers, model)
    # the original: backend from the feature string, the bench's options, its blocks(), its magnification statement
    be = al_testbed.make_backend(feat, RunConfig(device="cpu", cache_dir="unused"), "cpu",
                                 enc=SimpleNamespace(enc=SimpleNamespace(cfg=EncoderConfig(layers=cfg_layers),
                                                                         model=model)), support_k=K)
    reset_backend_for_new_support(be)
    for k_, v_ in (("blocks", blocks), ("zoom_fixed", zoom_fixed), ("feat_method", feat)):
        monkeypatch.setattr(bench["a"], k_, v_)
    monkeypatch.setitem(bench, "enc", SimpleNamespace(extract=ref_enc.extract))
    pairs = [blob_image(i, H=48, W=60, thin=metric == "cldice", noise=0.6) for i in range(K + 2)]
    sup = [SimpleNamespace(image=im, label_map=m) for im, m in pairs[:K]]
    loc = dict(bench, be=be, sup=sup, ds="synthetic", s=0)
    exec(bench["_zoom_stmt"], loc)
    model.calls.clear()
    ref, _ = bench["ridge_fit"](be, sup, np.random.default_rng(11), metric)
    calls_ref, model.calls[:] = list(model.calls), []
    # this package, the same option
    inputs = FrozenInputs("cpu", encoder=my_enc, layers=layers, fine="fine" in blocks.split(","))
    mine = Exemplar("cpu", metric="cldice" if metric == "cldice" else "iou", inputs=inputs, blocks=blocks,
                    zoom_fixed=zoom_fixed or None)
    mine.fit([im for im, _ in pairs[:K]], [m for _, m in pairs[:K]], rng=np.random.default_rng(11))
    assert int(be.fine_zoom) == mine.zoom and (zoom_fixed == 0 or mine.zoom == zoom_fixed)
    assert model.calls == calls_ref
    _same(ref, mine.result)
    # the test pass: the bench's blocks() + logit_map against the package's score map, on images not in the support
    for im, _ in pairs[K:]:
        xc, fine, bank, hw = bench["blocks"](be, im)
        assert (fine is None) == ("fine" not in blocks.split(","))
        for k_ in ("plain_mse", "bal_mse", "blend1_ex"):
            lr = bench["logit_map"](xc, fine, bank, hw, ref[k_]["w"])
            x2, f2, b2, hw2 = mine._inputs(im, mine.zoom)
            lm = logit_map(x2, f2, b2, hw2, mine.result[k_]["w"])
            assert np.array_equal(lr, lm), (arm, k_)
        assert np.array_equal(lr > ref["blend1_ex"]["tau"], mine.predict(im))


def test_loaders_and_fingerprint_match(tree, tmp_path):
    import tifffile
    from PIL import Image
    import active_segmenter.eval.datasets as RD
    import active_segmenter.eval.metrics as RM
    import active_segmenter.eval.registry as RR
    import closed_form_bench as cfb
    from benchmark import datasets as D, metrics as M, registry as R
    from benchmark.run import split_fingerprint
    rs = np.random.RandomState(0)
    for split in ("support", "test"):
        for sub in ("images", "masks"):
            os.makedirs(tmp_path / "few" / split / sub)
        for i in range(5):
            Image.fromarray(rs.randint(0, 255, (20, 24), np.uint8)).save(tmp_path / "few" / split / "images" / f"{i}.png")
            Image.fromarray((rs.rand(20, 24) > 0.5).astype(np.uint8) * 255).save(tmp_path / "few" / split / "masks" / f"{i}.png")
    os.makedirs(tmp_path / "flat" / "images")
    os.makedirs(tmp_path / "flat" / "masks")
    for i in range(12):
        Image.fromarray(rs.randint(0, 255, (18, 22, 3), np.uint8)).save(tmp_path / "flat" / "images" / f"im{i:02d}.png")
        Image.fromarray((rs.rand(18, 22) > 0.5).astype(np.uint8) * 255).save(tmp_path / "flat" / "masks" / f"im{i:02d}.bmp")
    for split in ("train", "test"):
        for sub in ("images", "masks"):
            os.makedirs(tmp_path / "inst" / "prep" / split / sub)
        for i in range(4):
            tifffile.imwrite(tmp_path / "inst" / "prep" / split / "images" / f"{i}.tif", rs.randint(0, 999, (16, 16)).astype(np.uint16))
            tifffile.imwrite(tmp_path / "inst" / "prep" / split / "masks" / f"{i}.tif", rs.randint(0, 5, (16, 16)).astype(np.uint16))

    def same(x, y):
        assert len(x) == len(y)
        for (a1, b1), (a2, b2) in zip(x, y):
            assert np.array_equal(a1, a2) and a1.dtype == a2.dtype and np.array_equal(b1, b2) and b1.dtype == b2.dtype
    same(RD.load_fewshot(str(tmp_path / "few"), "support", 3), D.load_fewshot(str(tmp_path / "few"), "support", 3))
    flat = (str(tmp_path / "flat" / "images"), str(tmp_path / "flat" / "masks"), 4, 10000)
    for r_, m_ in zip(RD.load_flat_fewshot(*flat), D.load_flat_fewshot(*flat)):
        assert r_                          # masks with another extension than the image are found (not a vacuous match)
        same(r_, m_)
    for r_, m_ in zip(RR._load_instance_split(str(tmp_path / "inst"), "train", 3),
                      R._load_instance_split(str(tmp_path / "inst"), "train", 3)):
        assert np.array_equal(r_, m_)
    pairs = D.load_fewshot(str(tmp_path / "few"), "test")
    assert cfb.split_fingerprint(pairs) == split_fingerprint(pairs)
    drop = {"ctc_u373"}                    # in the research lists, evaluated in earlier versions, not in the paper's ten
    assert {k: v for k, v in cfb.POOL.items() if k not in drop} == R.POOL
    assert cfb.CLDICE == R.CLDICE
    assert [d for d in cfb.ALL_DATASETS if d not in drop] == R.ALL_DATASETS
    for name in R.ALL_DATASETS:            # the roots differ on purpose: the research code defaults to cluster paths
        a, b = RR.PANEL[name], R.PANEL[name]
        assert (a.kind, a.url, a.images_sub, a.masks_sub) == (b.kind, b.url, b.images_sub, b.masks_sub)
        assert os.path.basename(a.root) == os.path.basename(b.root)
    for _ in range(50):
        p, g = rs.rand(30, 30) > rs.rand(), rs.rand(30, 30) > rs.rand()
        assert RM.cldice(p, g) == M.cldice(p, g) and RM.foreground_iou(p, g) == M.foreground_iou(p, g)
