"""(5) The ablation options of Table 2 on the CPU, without AutoSeg (the cross-check against the original code is in
test_crosscheck_autoseg.py): a readout restricted to some column blocks refits exactly and gives the removed blocks
weight 0, the API never feeds an excluded fine view to the readout, a fixed magnification holds through `add`, and
the runner writes the options into the record and scores Table 2's readout rows (alt plain_mse / bal_mse) as each
weighting's map cut at 1/2."""
import json
import os

import numpy as np
import pytest

from exemplar import Exemplar
from exemplar.readout import ridge_fit
from synth import SyntheticInputs, features, support

SUBSETS = ["coarse,fine", "bank", "coarse,bank", "fine,bank"]


def _feats(images, blocks):
    out = []
    for im in images:
        xc, fine, bank, hw = features(im)
        out.append((xc, fine if "fine" in blocks.split(",") else None, bank, hw))
    return out


def _identical(r1, r2):
    for key in ("plain_mse", "bal_mse", "blend_ex", "blend1_ex"):
        assert np.array_equal(r1[key]["w"], r2[key]["w"]), key
        for f in ("lam", "lam_bal", "a", "tau", "variant", "heldout"):
            assert r1[key].get(f) == r2[key].get(f), (key, f)


@pytest.mark.parametrize("blocks", SUBSETS)
def test_blocks_refit_equals_cold_fit_and_zeroes_removed_blocks(blocks):
    images, masks = support(5, noise=1.2)
    feats = _feats(images, blocks)
    fn = lambda i: feats[i]
    cold, _ = ridge_fit(fn, masks, "fg_iou", rng=np.random.default_rng(4), blocks=blocks)
    _, state = ridge_fit(fn, masks[:4], "fg_iou", rng=np.random.default_rng(4), blocks=blocks)
    refit, _ = ridge_fit(fn, masks, "fg_iou", state=state, blocks=blocks)
    _identical(refit, cold)
    xc, fine, bank, _ = feats[0]
    widths = [("coarse", xc.shape[1]), ("fine", 0 if fine is None else fine.shape[1]), ("bank", bank.shape[1])]
    w, o = cold["blend1_ex"]["w"], 0
    for name, c in widths:
        assert (not np.any(w[o:o + c])) == (name not in blocks.split(",")) or c == 0, name
        o += c
    assert len(w) == o + 1
    full, _ = ridge_fit(lambda i: features(images[i]), masks, "fg_iou", rng=np.random.default_rng(4))
    assert not np.array_equal(full["blend1_ex"]["w"][:len(w)], w)      # the option changes the fit


def test_api_drops_an_excluded_fine_view_of_custom_inputs():
    images, masks = support(4, noise=1.2)
    m = Exemplar(device="cpu", inputs=SyntheticInputs(), blocks="coarse,bank").fit(images, masks,
                                                                                   rng=np.random.default_rng(2))
    feats = _feats(images, "coarse,bank")
    ref, _ = ridge_fit(lambda i: feats[i], masks, "fg_iou", rng=np.random.default_rng(2), blocks="coarse,bank")
    _identical(m.result, ref)
    xc, _, bank, _ = feats[0]
    assert len(m.weights) == xc.shape[1] + bank.shape[1] + 1
    assert m.predict(images[0]).shape == images[0].shape


def test_api_fixed_magnification_holds_through_add():
    images, masks = support(5, noise=1.2)
    m1 = Exemplar(device="cpu", inputs=SyntheticInputs(), zoom_fixed=3).fit(images[:4], masks[:4],
                                                                            rng=np.random.default_rng(6))
    assert m1.zoom == 3 and m1._state["incremental"]
    m1.add(images[4], masks[4])
    m2 = Exemplar(device="cpu", inputs=SyntheticInputs(), zoom_fixed=3).fit(images, masks, rng=np.random.default_rng(6))
    assert m1.zoom == m2.zoom == 3
    _identical(m1.result, m2.result)
    with pytest.raises(ValueError):
        Exemplar(device="cpu", inputs=SyntheticInputs(), zoom_fixed=-1)
    with pytest.raises(ValueError):
        Exemplar(device="cpu", inputs=SyntheticInputs(), blocks="coarse,priors")


@pytest.mark.parametrize("argv,proto", [
    ([], dict(blocks="coarse,fine,bank", zoom_fixed=0, lf12=True)),
    (["--blocks", "bank,fine", "--zoom_fixed", "1"], dict(blocks="fine,bank", zoom_fixed=1, lf12=True)),
    (["--blocks", "coarse,bank"], dict(blocks="coarse,bank", zoom_fixed=0, lf12=True)),
    (["--last_layer_only"], dict(blocks="coarse,fine,bank", zoom_fixed=0, lf12=False)),
])
def test_runner_writes_the_options_and_the_readout_rows(tmp_path, monkeypatch, argv, proto):
    from benchmark import run
    from benchmark.metrics import foreground_iou
    from exemplar.readout import logit_map
    images, masks = support(8, noise=1.2)
    pool = list(zip(images[:5], masks[:5]))
    test = list(zip(images[5:], masks[5:]))
    monkeypatch.setattr(run, "load_split", lambda ds: (pool, test))
    a = run.parse_args(["--datasets", "dsb2018", "--k", "3", "--seeds", "2", "--score_dir", str(tmp_path)] + argv)
    inputs = SyntheticInputs()
    run.run_dataset("dsb2018", a, inputs, "cpu", "cpu")
    rec = json.load(open(os.path.join(tmp_path, "ridge__dsb2018.json")))
    p = rec["protocol"]
    assert p["blocks"] == proto["blocks"] and p["zoom_fixed"] == proto["zoom_fixed"]
    assert ("_lf12_" in p["feat_method"]) == proto["lf12"] and p["feat_method"].endswith("_fzu_fam" + (
        "_lf12" if proto["lf12"] else "") + "_natres")
    if proto["zoom_fixed"]:
        assert p["zooms_decided"] == [proto["zoom_fixed"]] * 2
    n = rec["test_per_seed"]
    assert rec["per_image"] == rec["alt"]["blend1_ex"] and len(rec["per_image"]) == 2 * n
    for s in range(2):                                   # each readout's map cut at 1/2, as the bench scores alt
        rng = np.random.default_rng(s)
        sel = rng.choice(len(pool), 3, replace=False)
        m = Exemplar(device="cpu", inputs=inputs, blocks=a.blocks, zoom_fixed=a.zoom_fixed or None)
        m.fit([pool[i][0] for i in sel], [pool[i][1] for i in sel], rng=rng)
        assert m.zoom == p["zooms_decided"][s]
        for key in ("plain_mse", "bal_mse", "blend1_ex"):
            assert m.result[key]["tau"] == 0.5
            for ti, (im, gt) in enumerate(test):
                xc, fine, bank, hw = m._inputs(im, m.zoom)
                pred = logit_map(xc, fine, bank, hw, m.result[key]["w"]) > 0.5
                assert rec["alt"][key][s * n + ti] == foreground_iou(pred, gt), (key, s, ti)


# the paper's arms of Table 2 (scripts/c117/ridge4_paper.sh) -> this runner's options
ARM_FLAGS = {"ridge4_k8": [], "ridge4_nocls_k8": ["--blocks", "coarse,fine"], "ridge4_bankonly_k8": ["--blocks", "bank"],
             "ridge4_coarseonly_k8": ["--blocks", "coarse,bank"], "ridge4_fineonly_k8": ["--blocks", "fine,bank"],
             "ridge4_nolf12_k8": ["--last_layer_only"], "ridge4_nozoom_k8": ["--zoom_fixed", "1"]}


def test_runner_records_pass_the_ablation_tables_protocol_check(tmp_path, monkeypatch):
    """A re-run of every Table 2 arm with this runner carries the protocol fields paper/make_ablation_table.py checks
    before it fills a row (RIDGE_PROTO and the row's own fields), so the shipped generator accepts it."""
    import importlib.util
    from benchmark import run
    monkeypatch.setenv("ASG_SEM_TREE", str(tmp_path))
    monkeypatch.setenv("ASG_ABL_SET", "ridge4")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "make_ablation_table.py")
    spec = importlib.util.spec_from_file_location("make_ablation_table_under_test", path)
    mat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mat)
    rows = {d: expect for block in mat.RIDGE_BLOCKS for d, key, _, _, expect in block if key is None}
    rows[mat.RIDGE_FULL] = None
    assert set(rows) == set(ARM_FLAGS), "the table's arms changed"
    images, masks = support(6, noise=1.2)
    monkeypatch.setattr(run, "load_split", lambda ds: (list(zip(images[:4], masks[:4])), list(zip(images[4:], masks[4:]))))
    for arm, flags in ARM_FLAGS.items():
        out = tmp_path / arm
        a = run.parse_args(["--datasets", "drive", "--k", "8", "--seeds", "1", "--score_dir", str(out)] + flags)
        a.k = 3                                          # the synthetic pool holds 4 images; support is checked apart
        run.run_dataset("drive", a, SyntheticInputs(), "cpu", "cpu")
        rec = json.load(open(out / "ridge__drive.json"))
        want = {**mat.RIDGE_PROTO, **(rows[arm] or dict(blocks="coarse,fine,bank", zoom_fixed=0))}
        assert rec["method"] == "ridge" and rec["metric"] == "cldice"
        for k, v in want.items():
            if k == "support":
                assert rec["protocol"][k] == 3
                continue
            assert rec["protocol"].get(k) == v, (arm, k, rec["protocol"].get(k), v)
        assert {"plain_mse", "bal_mse"} <= set(rec["alt"])               # the readout rows read these
