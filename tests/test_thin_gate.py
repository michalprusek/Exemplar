"""The native-classical size gate must be defined by the model, not fitted to the panel.

The gate used to compare the support's mean image side against a hardcoded 1500, a number placed
between HRF (3504 px) and the rest of the panel because HRF regressed. That is per-dataset tuning
however it is routed, and the paper claims the opposite. These tests pin the replacement: the bound
is the head's own feature-resolution cap, so the rule is defined by the architecture and says what
happens on a dataset nobody has measured.
"""
from __future__ import annotations

import inspect

from active_segmenter.segment.head_fusion_backend import HeadFusionBackend, thin_gate


def test_gate_is_off_above_the_training_cap_and_on_below_it():
    cap = 1536
    # thin support, image larger than the cap -> the native bank would face a downscaled-trained
    # head, the exact distribution shift that cost hrf 0.607 -> 0.469
    assert thin_gate(0.8, 3504, 0.4, cap) is False
    # thin support that the cap does not downscale at all -> no shift by construction
    assert thin_gate(0.8, 630, 0.4, cap) is True


def test_the_boundary_is_inclusive_because_at_the_cap_no_downscaling_happens():
    """An image exactly at the cap is passed through untouched, so the shift is zero, so the gate
    must be ON. A strict `<` would switch it off for the one size where the argument is exact."""
    assert thin_gate(0.8, 1536, 0.4, 1536) is True


def test_blobby_support_never_enables_the_native_bank_regardless_of_size():
    """The size test is a safety bound on top of the morphology test, not a replacement for it."""
    assert thin_gate(0.1, 256, 0.4, 1536) is False


def test_the_old_fitted_constant_is_gone_from_the_default_configuration():
    """`thin_max_side` must default to None (derive from the cap), not to the fitted 1500.

    The whole point is that no dataset-fitted number survives in the shipped configuration. An
    explicit value is still accepted, for ablation only.
    """
    default = inspect.signature(HeadFusionBackend.__init__).parameters["thin_max_side"].default
    assert default is None, f"thin_max_side still defaults to a fitted constant: {default!r}"


def test_the_default_cap_matches_the_heads_own_training_cap():
    """If these two ever diverge, the gate silently stops being 'the cap' and becomes a constant."""
    sig = inspect.signature(HeadFusionBackend.__init__).parameters
    assert sig["max_side"].default == 1536
    # the fitted 1500 sat within 2.4% of this, which is why the two rules agree on the panel
    assert abs(1500 - sig["max_side"].default) / sig["max_side"].default < 0.03


def test_the_benchmark_entry_point_does_not_reinstate_the_fitted_constant():
    """`make_backend` is the single path every benchmark takes.

    Removing the default while still naming 1500 at the call site would leave the shipped
    configuration unchanged and the claim false, with nothing to show for it but a different
    default nobody exercises.
    """
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "scripts" / "al_testbed.py").read_text()
    line = [ln for ln in src.splitlines() if "thin_max_side=" in ln and "#" not in ln.split("thin_max_side")[0]]
    assert line, "expected make_backend to pass thin_max_side"
    assert "1500" not in " ".join(line), f"fitted constant reinstated at the call site: {line}"


def test_the_gate_is_actually_wired_to_the_training_cap_in_fit():
    """`thin_gate` and the `__init__` default can both be correct while `fit` still hardcodes 1500.

    Reverting the call site to `cap = 1500` left this whole file green, so the tests certified a
    change the segmenter did not make. Exercising the real path needs a GPU fit, so this asserts on
    the source: the bound must come from the head's own cap, and the fitted constant must not appear.
    """
    import pathlib
    import re
    src = (pathlib.Path(__file__).resolve().parents[1] / "active_segmenter" / "segment"
           / "head_fusion_backend.py").read_text()
    body = src[src.index("if self.thin_adaptive"):]
    cap_line = next(ln for ln in body.splitlines() if ln.strip().startswith("cap ="))
    assert "self.max_side" in cap_line, f"gate no longer derives from the training cap: {cap_line}"
    assert not re.search(r"\b1500\b", cap_line), f"fitted constant back at the call site: {cap_line}"
    gate_line = next(ln for ln in body.splitlines() if "_thin_active =" in ln)
    assert "thin_gate(" in gate_line, f"fit no longer calls the shared rule: {gate_line}"


def test_the_training_cap_is_not_the_same_on_every_head_config():
    """A coupling the replacement introduces, recorded so it is not discovered by a silent regression.

    The bound is now `self.max_side`, which `al_testbed` sets to 768 for the fast AL config and 1024
    for the unfrozen-bank config, not 1536. Neither enables `thin_adaptive` today, so nothing is
    wrong now; but if either ever does, the gate silently becomes `ms <= 768` and switches OFF for a
    1000-px dataset such as MoNuSeg -- a far larger behaviour change than the constant it replaced.
    """
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "scripts" / "al_testbed.py").read_text()
    assert "max_side=" in src
    thin_line = next(ln for ln in src.splitlines() if "thin_adaptive=(" in ln)
    assert "head_fusion_al" not in thin_line and "head_fusion_tc" not in thin_line, (
        "a config with a non-default max_side now enables thin_adaptive; re-check the gate bound "
        "before trusting it")
