"""(3) The exact refit: adding one image's sums to a fit of K-1 images gives the same readout, bit for bit, as a cold
fit of all K images (the identity check of the bench's refit: weights, lambdas, blend weight, cut and readout), also
through the public API, including a refit where the K masks decide another magnification."""
import numpy as np
import pytest

from exemplar import Exemplar
from exemplar.fineview import decide_zoom
from exemplar.readout import ridge_fit
from synth import SyntheticInputs, blob_image, disc_image, features, support


def _identical(r1, r2):
    for key in ("plain_mse", "bal_mse", "blend_ex", "blend1_ex"):
        a, b = r1[key], r2[key]
        assert np.array_equal(a["w"], b["w"]), key
        for f in ("lam", "lam_bal", "a", "tau", "variant", "heldout"):
            assert a.get(f) == b.get(f), (key, f, a.get(f), b.get(f))


@pytest.mark.parametrize("noise", [0.25, 1.2])
@pytest.mark.parametrize("metric", ["fg_iou", "cldice"])
def test_refit_equals_cold_fit(metric, noise):
    images, masks = support(5, noise=noise)
    feats = [features(im) for im in images]
    fn = lambda i: feats[i]
    cold, _ = ridge_fit(fn, masks, metric, rng=np.random.default_rng(7))
    km1, state = ridge_fit(fn, masks[:4], metric, rng=np.random.default_rng(7))
    refit, _ = ridge_fit(fn, masks, metric, state=state)
    _identical(refit, cold)
    assert not np.array_equal(km1["blend1_ex"]["w"], cold["blend1_ex"]["w"])     # the new mask changed the fit


def test_api_add_equals_cold_fit_same_magnification():
    images, masks = support(5, noise=1.2)
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    m1 = Exemplar(device="cpu", inputs=SyntheticInputs()).fit(images[:4], masks[:4], rng=rng_a)
    z0 = m1.zoom
    m1.add(images[4], masks[4])
    m2 = Exemplar(device="cpu", inputs=SyntheticInputs()).fit(images, masks, rng=rng_b)
    assert m1.zoom == m2.zoom == z0
    _identical(m1.result, m2.result)
    assert np.array_equal(m1.predict(images[0]), m2.predict(images[0]))


def test_api_add_equals_cold_fit_when_magnification_changes():
    # [disc, lines] needs the largest magnification (median weighted patch Dice < 0.8 up to z = 4); a second disc
    # moves the median to the discs' value, which reaches 0.8 at z = 1
    d20, lines, d30 = disc_image(20, seed=1), blob_image(5, thin=True), disc_image(30, seed=2)
    images = [d20[0], lines[0], d30[0]]
    masks = [d20[1], lines[1], d30[1]]
    sides = [max(im.shape) for im in images]
    z2, z3 = decide_zoom(masks[:2], sides[:2]), decide_zoom(masks, sides)
    assert z2 != z3, "the construction must move the magnification"
    m1 = Exemplar(device="cpu", inputs=SyntheticInputs()).fit(images[:2], masks[:2], rng=np.random.default_rng(1))
    m1.add(images[2], masks[2])
    m2 = Exemplar(device="cpu", inputs=SyntheticInputs()).fit(images, masks, rng=np.random.default_rng(1))
    assert m1.zoom == m2.zoom == z3
    _identical(m1.result, m2.result)


def test_api_add_to_single_mask_fit():
    images, masks = support(2)
    m1 = Exemplar(device="cpu", inputs=SyntheticInputs()).fit(images[:1], masks[:1], rng=np.random.default_rng(2))
    m1.add(images[1], masks[1])                     # K = 1 holds quadrant folds: the update refits cold
    m2 = Exemplar(device="cpu", inputs=SyntheticInputs()).fit(images, masks, rng=np.random.default_rng(2))
    _identical(m1.result, m2.result)
