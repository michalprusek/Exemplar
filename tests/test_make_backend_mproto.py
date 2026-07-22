"""The `mproto` composable lever token wires multi-prototype correspondence into best_v2, and is
default-OFF (absent the token → byte-identical single-prototype best_v2)."""
from scripts.al_testbed import make_backend


def test_mproto_token_sets_nproto_and_corr():
    be = make_backend("head_fusion_best_cgate_film_nobank_mproto", cfg=None, dev="cpu")
    assert be.n_proto == 4 and be.corr_prior is True


def test_default_off_parity():
    be = make_backend("head_fusion_best_cgate_film_nobank", cfg=None, dev="cpu")
    assert be.n_proto == 1 and be.corr_prior is False


def test_mproto_composes_with_corr_token():
    """mproto and the plain corr token both imply corr_prior; mproto sets n_proto>1."""
    be = make_backend("head_fusion_best_cgate_film_nobank_corr_mproto", cfg=None, dev="cpu")
    assert be.corr_prior is True and be.n_proto == 4


def test_bdou_token_sets_boundary_dou():
    be = make_backend("head_fusion_best_cgate_film_nobank_bdou", cfg=None, dev="cpu")
    assert be.boundary_dou is True


def test_default_off_boundary_dou():
    be = make_backend("head_fusion_best_cgate_film_nobank", cfg=None, dev="cpu")
    assert be.boundary_dou is False


def test_prec_token_sets_prec_loss():
    be = make_backend("head_fusion_best_cgate_film_nobank_prec", cfg=None, dev="cpu")
    assert be.prec_loss is True


def test_default_off_prec_loss():
    be = make_backend("head_fusion_best_cgate_film_nobank", cfg=None, dev="cpu")
    assert be.prec_loss is False
