"""The self-configuring gates must NOT leak across support draws.

Every gate in `HeadFusionBackend` latches on `is None` and is therefore decided at the first `fit`.
The multi-draw benchmark protocol builds one backend and loops over seeds, so without an explicit
reset every seed after the first inherits seed 0's colour channel, CLAHE strength, thin gate,
affinity calibration and FiLM prototypes — which silently understates the across-seed variance of
the self-configuration step. These tests pin the reset contract.
"""
import numpy as np

from active_segmenter.segment.base import LabeledExample
from active_segmenter.segment.head_fusion_backend import HeadFusionBackend

# every attribute `reset_support_state` promises to restore to its freshly-constructed value
_RESET_ATTRS = ("head", "_n_cls", "_n_classical", "_clahe_strength", "_thin_active",
                "_contrast_source", "_color_dataset", "_fg_proto", "_bg_proto", "_fg_protos",
                "_bg_protos", "_inst_r", "_inst_merge_cos", "_bank",
                "tile_classical", "fine_scales", "trainable_classical",
                # support-derived lever state: `_bank_keep` (bankselect's Fisher channel mask) and
                # `_bank_scales` (scaleconf's cluster-derived bank scales). Both are fitted from the
                # K masks of ONE draw, so leaving either would apply seed 0's selection to every later
                # seed -- and because `_bank_scales` changes the bank WIDTH, a leaked one also
                # mismatches the head that was rebuilt for a different channel count.
                "_bank_keep", "_bank_scales",
                # latched OFF (permanently) on a degenerate draw; `fit` guards prototype construction on
                # `(corr_prior or film)`, so leaving these unrestored silently disables FiLM for every
                # later seed — the leak this whole contract exists to prevent.
                "corr_prior", "film")


def _thin_example(H=128):
    """A 3-px-wide cross → high per-component tubularity, so the thin gate fires."""
    m = np.zeros((H, H), bool)
    m[H // 2 - 1:H // 2 + 2, :] = True
    m[:, H // 2 - 1:H // 2 + 2] = True
    img = (m * 200.0).astype(np.float32)
    feat = np.random.default_rng(1).standard_normal((8, 8, 16)).astype(np.float32)
    return LabeledExample(image=img, feat_grid=feat, label_map=m.astype(int))


def _blob_example(H=128):
    """A filled disk → low tubularity, so the thin gate must stay off."""
    yy, xx = np.mgrid[0:H, 0:H]
    m = (yy - H // 2) ** 2 + (xx - H // 2) ** 2 < (H // 4) ** 2
    img = (m * 200.0).astype(np.float32)
    feat = np.random.default_rng(2).standard_normal((8, 8, 16)).astype(np.float32)
    return LabeledExample(image=img, feat_grid=feat, label_map=m.astype(int))


def test_reset_restores_freshly_constructed_state():
    """After a reset the backend must be indistinguishable from a new one on every latched field."""
    kw = dict(device="cpu", epochs=2, proj_dim=8, amp=False, thin_adaptive=True,
              color_adaptive=True, clahe_adaptive=True, contrast_norm=True)
    be, fresh = HeadFusionBackend(**kw), HeadFusionBackend(**kw)

    be.fit([_thin_example(), _thin_example()])
    # the fit must actually have latched something, else the test would pass vacuously
    assert be._contrast_source is not None and be._thin_active is not None

    be.reset_support_state()
    for attr in _RESET_ATTRS:
        assert getattr(be, attr) is getattr(fresh, attr) or \
               getattr(be, attr) == getattr(fresh, attr), f"{attr} survived reset_support_state()"


def test_thin_gate_is_re_derived_for_a_new_support_draw():
    """The bug this guards: a blobby draw following a thin draw kept the thin draw's gate."""
    be = HeadFusionBackend(device="cpu", epochs=2, proj_dim=8, amp=False, thin_adaptive=True)

    be.fit([_thin_example(), _thin_example()])
    assert be._thin_active is True, "thin support should engage the native-classical gate"
    assert be.tile_classical and be.fine_scales, "thin gate must flip its two side-effect flags"

    be.reset_support_state()
    be.fit([_blob_example(), _blob_example()])
    assert be._thin_active is False, "blobby support must re-derive the gate, not inherit it"
    assert not be.tile_classical and not be.fine_scales, "side-effect flags must be restored too"


def test_reset_drops_classical_cache_but_keeps_fine_feature_cache():
    """`_ccache` can go stale (CLAHE strength is not in its key); `_fcache` cannot, and is expensive."""
    be = HeadFusionBackend(device="cpu", epochs=2, proj_dim=8, amp=False, color_adaptive=True)
    be.fit([_blob_example(), _blob_example()])
    be._fcache[12345] = "fine-grid-sentinel"           # depends only on encoder+image, must survive
    assert be._ccache, "fit should have populated the classical-feature cache"

    be.reset_support_state()
    assert not be._ccache, "stale classical features must be dropped"
    assert be._fcache == {12345: "fine-grid-sentinel"}, "fine DINO grids must be kept"


def test_degenerate_draw_does_not_disable_film_for_later_seeds():
    """A support draw whose foreground vanishes at grid resolution latches corr_prior/film off.
    That must NOT survive the reset, or every subsequent seed silently runs a different method."""
    be = HeadFusionBackend(device="cpu", epochs=2, proj_dim=8, amp=False, film=True, corr_prior=True)
    assert be.film and be.corr_prior, "precondition: the levers start enabled"
    empty = LabeledExample(image=np.zeros((64, 64), np.float32),
                           feat_grid=np.zeros((4, 4, 8), np.float32),
                           label_map=np.zeros((64, 64), int))
    try:
        be.fit([empty])
    except Exception:
        pass                                  # the fit may legitimately bail; the latch is what matters
    be.reset_support_state()
    assert be.film, "film stayed off after reset -> later seeds run without a headline lever"
    assert be.corr_prior, "corr_prior stayed off after reset"
