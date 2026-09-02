"""The ablation arms must be DISTINCT configurations, each differing from the full method by exactly
one component.

A composable method string like `head_fusion_best_nocls_nobank` is validated against a token
allow-list, so an unknown token raises -- but a token that is parsed and then never passed through to
the backend does NOT raise. It silently yields the same configuration as its neighbour, and the
ablation table then reports two identical rows as if they were an ablation. That is unfalsifiable
from the numbers alone: the reader sees two similar values and concludes the component does not
matter, which is precisely the claim the table exists to test.
"""
from __future__ import annotations

import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

# The configuration fingerprint. It MUST cover every backend field any arm token can change: a field
# left out makes two genuinely different arms compare equal, so the distinctness check below passes
# vacuously -- which is how the self-configuration arms (`noloss`/`nocolor`/`rgbfeat`, added to
# run_ablation.ARMS later) went unchecked. Extend this whenever a new ablation token is wired.
FIELDS = ("aug", "dino_only", "competitive_gate", "film", "dino_scale", "bank_unfreeze_adaptive",
          "adaptive_loss",        # `noloss`  -> the morphology-driven loss constructor
          "color_adaptive",       # `nocolor` -> support-driven colour/stain channel selection
          "rgb_feat",             # `rgbfeat` -> raw R,G,B appended as native features
          # Everything below is settable from a method token and was invisible to this fingerprint,
          # which meant `..._lean`, `..._flat` and `..._h64` all compared EQUAL to the unqualified
          # method and to each other. The distinctness test then passed while proving nothing --
          # exactly the failure this file exists to prevent, repeated one layer up.
          "stem", "hidden", "depth", "hidden_film", "dropout", "eca", "se", "mix", "bilinear",
          "banknorm", "early_stop", "weight_decay", "epochs",
          "proj_dim", "corr_prior",
          # `noup` -> plain bilinear in place of the prior-guided upsampler. Added 2026-08-25 with
          # the token; without it an arm differing ONLY by the upsampler compares equal to the full
          # method, which is the vacuous pass this fingerprint exists to prevent.
          "upsampler")
FULL = "head_fusion_best_cgate_film_nobank"          # the campaign's `ours` row


def _config(method):
    from al_testbed import make_backend
    from active_segmenter.config import RunConfig
    be = make_backend(method, RunConfig(device="cpu"), "cpu")
    return {f: getattr(be, f, "?") for f in FIELDS}


@pytest.fixture(scope="module")
def arms():
    from run_ablation import ARMS
    return ARMS


def test_every_arm_resolves_and_is_a_distinct_configuration(arms):
    cfgs = {label: _config(m) for label, m in arms.items()}
    cfgs["FULL"] = _config(FULL)
    seen = {}
    for label, c in cfgs.items():
        key = tuple(sorted(c.items()))
        assert key not in seen, (
            f"ablation arms {seen[key]!r} and {label!r} are the SAME configuration {c}; one of "
            f"their tokens is parsed but never reaches the backend, so the table would report two "
            f"identical rows as an ablation")
        seen[key] = label


def test_each_arm_differs_from_the_full_method_by_the_component_it_names(arms):
    """An arm that differs in two components measures the wrong thing and the row label lies."""
    full = _config(FULL)
    expected = {                       # arm label -> the fields it is SUPPOSED to change
        # architecture ablation: what the head is built from
        "abl_nocls":      {"dino_only", "competitive_gate", "film"},   # backbone-only: gate/FiLM off too
        "abl_bank":       {"competitive_gate", "film"},                # bank only, no gate, no FiLM
        "abl_cgate":      {"film"},                                    # bank + gate
        "abl_film":       {"competitive_gate"},                        # bank + FiLM
        "abl_coarseonly": {"dino_scale"},                              # full method, one DINO scale
        # self-configuration ablation: the paper's novelty axis, held at the FULL architecture
        "abl_sc_noloss":  {"adaptive_loss"},                           # fixed loss weights
        "abl_sc_nocolor": {"color_adaptive"},                          # grayscale input, no selection
        # all self-config off: FiLM is itself a support-conditioned component, so this arm drops it too
        "abl_sc_none":    {"film", "adaptive_loss", "color_adaptive"},
        # simplification A/B: instead of SELECTING a channel, hand the head raw R,G,B and let it weight
        "abl_rgbfeat":    {"color_adaptive", "rgb_feat"},
    }
    for label, method in arms.items():
        diff = {f for f, v in _config(method).items() if v != full[f]}
        assert diff == expected[label], (
            f"{label} ({method}) differs from the full method in {sorted(diff)}, expected "
            f"{sorted(expected[label])}")


def test_the_full_method_is_not_among_the_arms(arms):
    """It is the campaign's `ours` row at K=8. Re-running it under another label would write a
    second record of the same configuration, which is the duplicate `stats()` refuses."""
    assert FULL not in arms.values()


def test_the_ablation_protocol_matches_the_campaign_on_every_axis():
    """An ablation measured under a different protocol is not comparable to the row it ablates,
    and the difference is invisible in the table.

    This test exists because the first version omitted `--metric_override`. The campaign scores
    instance datasets as foreground IoU; without the flag the ablation scored them as instance AP,
    so stats() would have skipped every comparison on dsb2018, MoNuSeg and CTC-U373 as METRIC
    MISMATCH -- half the table, with no error. It was found by running real campaign records through
    stats and reading the MIXED METRICS warning, not by reading the code.
    """
    import run_ablation as A
    import run_campaign as C

    for ds in A.DATASETS:
        cmd = A.cmd_for(dict(label="abl_x", method="m", ds=ds))
        want = "cldice" if C.PANEL[ds].metric == "cldice" else "fg_iou"
        assert f"--metric_override {want}" in cmd, f"{ds}: wrong or missing metric override in {cmd}"
        assert f"--seeds {C.SEEDS}" in cmd, f"{ds}: seed count differs from the campaign"
        assert "--test 10000" in cmd, f"{ds}: test slice differs from the campaign"
        assert f"--pool {C.pool_for(ds)}" in cmd, f"{ds}: pool differs from the campaign"
        assert "--res 672" in cmd, f"{ds}: resolution differs from the campaign"
        assert C.FEAT_CACHE in cmd, f"{ds}: feature cache differs from the campaign"


def test_smoke_mode_never_writes_into_the_reported_tree():
    """A reduced record sitting in the campaign tree looks exactly like a measurement."""
    import run_ablation as A
    import run_campaign as C

    j = dict(label="abl_x", method="m", ds="spheroidj")
    smoke, real = A.cmd_for(j, smoke=True), A.cmd_for(j, smoke=False)
    assert C.TREE not in smoke, "smoke would write into the reported campaign tree"
    assert A.SMOKE_DIR in smoke
    assert "--seeds 1" in smoke and "--test 4" in smoke, "smoke must actually be cheap"
    assert C.TREE in real and "--test 10000" in real


def test_the_memory_floor_is_higher_for_the_methods_that_need_more():
    """One floor did not fit every method.

    SegGPT ensembles the whole support set inside its attention, so its peak memory grows with K.
    `seggpt_k16_dsb2018` died with CUDA OOM after 8 seconds while holding 19.1 GiB, on a device
    where the 9000 MiB check had just passed: the guard was right about free memory and wrong about
    how much the job would go on to want. A job that OOMs is recorded as FAILED, not requeued, so
    an under-estimating floor turns into missing table cells rather than a delay.
    """
    import run_campaign as C

    assert C.min_free_mib("seggpt", 16) > C.min_free_mib("seggpt", 8) > C.MIN_FREE_MIB, \
        "SegGPT's floor must grow with K"
    assert C.min_free_mib("seggpt", 16) >= 19100, \
        "the floor must exceed the 19.1 GiB the failing job was holding"
    # everything else keeps the DINOv3-sized default, including SegGPT at the small K that ran fine
    for m in ("ours", "universeg", "tyche", "matcher", "cellpose_ft"):
        assert C.min_free_mib(m, 8, "dsb2018") == C.MIN_FREE_MIB
    assert C.min_free_mib("seggpt", 1, "dsb2018") == C.MIN_FREE_MIB


def test_the_memory_floor_also_accounts_for_the_dataset():
    """`ours_k8_hrf` OOMed at 15.4 GiB while two siblings held 11.7 and 17.2 GiB on one 44 GiB card.

    HRF's images are 3504 px, so the native-resolution pathway holds far more activation than on a
    584 px set. Demand is (method, K, dataset); a floor that knows only the method under-books the
    largest IMAGES exactly as it under-booked the largest K, and an OOM is recorded as FAILED rather
    than requeued -- so it becomes a missing cell for our own method.
    """
    import run_campaign as C

    assert C.min_free_mib("ours", 8, "hrf") >= 15400, \
        "the floor must exceed what the failing job was holding"
    assert C.min_free_mib("ours", 8, "hrf") > C.min_free_mib("ours", 8, "dsb2018")
    # the two axes compose: a heavy method on a heavy dataset takes the larger of the two
    assert C.min_free_mib("seggpt", 16, "hrf") >= C.min_free_mib("seggpt", 16, "dsb2018")


# The configuration every reported number was produced with, pinned by literal value.
REPORTED = {"stem": "wide", "hidden": 256, "depth": 2, "hidden_film": 64, "proj_dim": 32,
            "epochs": 60, "dropout": 0.0, "weight_decay": 0.0, "se": 0, "bilinear": 0,
            "eca": False, "mix": False, "banknorm": False, "early_stop": False,
            "aug": "flip"}


def test_the_reported_method_still_resolves_to_the_reported_configuration():
    """Every number in the paper rests on this, and nothing else asserts it.

    The other tests in this file compare each arm RELATIVE to the full method, so moving a default
    moves the baseline and every arm together and stays invisible. That is a live risk rather than a
    theoretical one: this branch exists to sweep exactly these defaults, and leaving one swept would
    silently re-baseline every table while the whole suite still passed.
    """
    from al_testbed import make_backend
    from active_segmenter.config import RunConfig
    be = make_backend(FULL, RunConfig(device="cpu"), "cpu")
    got = {k: getattr(be, k) for k in REPORTED}
    assert got == REPORTED, f"the reported method drifted: {got} != {REPORTED}"


def test_every_lever_token_reaches_the_backend():
    """A token parsed but never passed through yields an arm identical to another one.

    This is the meta-test: it walks the token families rather than a hand-listed set, so the NEXT
    lever added without extending FIELDS fails here instead of silently producing a duplicate arm
    that gets reported as an ablation.
    """
    from al_testbed import make_backend
    from active_segmenter.config import RunConfig
    base = {f: getattr(make_backend(FULL, RunConfig(device="cpu"), "cpu"), f, "?") for f in FIELDS}
    for token, field, expect in [("lean", "stem", "lean"), ("flat", "stem", "flat"),
                                 ("h64", "hidden", 64), ("d1", "depth", 1),
                                 ("f32", "hidden_film", 32), ("do5", "dropout", 0.5),
                                 ("wd4", "weight_decay", 1e-4), ("se16", "se", 16),
                                 ("bil16", "bilinear", 16), ("ep200", "epochs", 200),
                                 ("es_ep500", "early_stop", True), ("eca", "eca", True),
                                 ("mix", "mix", True), ("dih", "aug", "d4"), ("banknorm", "banknorm", True),
                                 ("pd16", "proj_dim", 16), ("noup", "upsampler", None)]:
        be = make_backend(FULL + "_" + token, RunConfig(device="cpu"), "cpu")
        assert getattr(be, field) == expect, f"token {token!r} did not set {field}"
        now = {f: getattr(be, f, "?") for f in FIELDS}
        assert now != base, f"token {token!r} left the configuration identical to the full method"


def test_make_backend_works_for_every_method_name_not_only_the_lever_grammar():
    """Only `head_fusion_best_*` names enter the token parser, but the constructor reads the lever
    variables unconditionally. Assigning their defaults inside that branch left every other arm --
    the ablation baseline, the fold-in, the AL loop -- raising UnboundLocalError from the single
    function all benchmarks go through."""
    from al_testbed import make_backend
    from active_segmenter.config import RunConfig
    for name in ("head_fusion", "head_fusion_adaptive", "head_fusion_best", "head_fusion_al",
                 "head_fusion_uni_adapt", "head_fusion_sf", "head_fusion_thin"):
        be = make_backend(name, RunConfig(device="cpu"), "cpu")
        assert be.stem == "wide" and be.epochs in (30, 60) and be.weight_decay == 0.0, name


def test_a_named_token_is_not_also_read_as_a_numeric_one():
    """`d4` was matched by BOTH the augmentation rule and the depth rule `d<N>`, so it silently set
    depth=4 and every arm using it died at fit time. It failed loudly, but the next such collision
    might not: a named token shadowing `h<N>` or `f<N>` would quietly change the width."""
    from al_testbed import make_backend
    from active_segmenter.config import RunConfig
    for tok in ("dih", "flat", "lean", "film", "eca", "mix", "es", "banknorm"):
        be = make_backend(FULL + "_" + tok, RunConfig(device="cpu"), "cpu") if tok != "es" \
            else make_backend(FULL + "_es_ep500", RunConfig(device="cpu"), "cpu")
        assert be.depth == 2 and be.hidden == 256 and be.hidden_film == 64, \
            f"named token {tok!r} was also read by a numeric-prefix rule"
