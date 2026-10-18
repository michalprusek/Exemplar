"""(5) The prior bank returns the 35 channels in the documented order and shape: each family's slice is that family's
response with its own GroupNorm, and the complement pairs (Sauvola r / 1 - r, intensity / 1 - intensity) cancel after
the joint normalisation."""
import numpy as np
import torch
import torch.nn.functional as F

from exemplar.priors import FAMILY_SLICES, PRIOR_CHANNELS, _gray01, make_prior_bank, normalize_image, priors


def _image(H=70, W=90):
    yy, xx = np.mgrid[:H, :W]
    img = 40 + 120 * ((yy - 30) ** 2 + (xx - 40) ** 2 < 150) + 60 * (np.abs(yy - xx / 2) < 1.5)
    rgb = np.stack([img, img * 0.5, img * 0.25], -1) + np.random.RandomState(0).rand(H, W, 3) * 20
    return rgb.astype(np.uint8)


def test_bank_shape_and_order():
    image = _image()
    bank = make_prior_bank("cpu")
    out = priors(image, bank, "cpu")
    assert out.shape == (1, 35, 70, 90) and out.dtype == torch.float32
    assert len(PRIOR_CHANNELS) == 35 and PRIOR_CHANNELS[0] == "frangi_s1_dark" and PRIOR_CHANNELS[-1] == "log_s0.016_abs"
    t = normalize_image(torch.from_numpy(_gray01(image))[None, None])
    fam = {"frangi": bank.frangi, "sauvola": bank.sauvola, "struct": bank.struct, "intensity": bank.intensity,
           "gradmag": bank.gradmag, "log": bank.log_bank}
    with torch.no_grad():
        for name, sl in FAMILY_SLICES.items():
            raw = fam[name](t)
            assert raw.shape[1] == sl.stop - sl.start, name
            ref = F.group_norm(raw, 1, eps=1e-5)
            assert torch.allclose(out[:, sl], ref, atol=1e-5), name
    for c in (10, 12, 14, 20):                                # r and 1 - r, normalised together, cancel
        assert float((out[0, c] + out[0, c + 1]).abs().max()) < 1e-4, PRIOR_CHANNELS[c]
    # Frangi: per sigma dark-then-bright, and the two polarities never fire on one pixel
    assert float((out[0, 0:10:2] > out[0, 0:10:2].min()).logical_and(out[0, 1:10:2] > out[0, 1:10:2].min())
                 .float().sum()) == 0.0
    # LoG: the absolute channel is the magnitude of the signed one before normalisation
    with torch.no_grad():
        raw = bank.log_bank(t)
    assert torch.equal(raw[:, 1::2], raw[:, 0::2].abs())
