import numpy as np
import pytest

from active_segmenter.config import RefineConfig
from active_segmenter.types import InstanceMask


@pytest.mark.gpu
def test_sam_refine_sharpens_coarse_mask():
    """On tulen: a coarse blocky mask over a disc is sharpened by SAM to a crisp
    disc (boundary IoU improves over the coarse input)."""
    from active_segmenter.refine.sam import SamRefiner

    H = 128
    yy, xx = np.mgrid[0:H, 0:H]
    disc = ((yy - 64) ** 2 + (xx - 64) ** 2) < 25 ** 2
    img = np.stack([disc * 200] * 3, -1).astype(np.uint8)

    coarse = np.zeros((H, H), bool)
    coarse[44:84, 44:84] = True  # a square bounding the disc (poor IoU)
    coarse_iou = (coarse & disc).sum() / (coarse | disc).sum()

    r = SamRefiner(RefineConfig(kind="sam"), device="cuda")
    out = r.refine(img, [InstanceMask(mask=coarse, points=None, class_id=1, instance_id=0)])
    ref = out[0].mask
    ref_iou = (ref & disc).sum() / (ref | disc).sum()
    assert ref_iou > coarse_iou
    assert ref_iou > 0.85


@pytest.mark.gpu
def test_mask_prompt_runs_and_sharpens():
    """The mask-prompt path (coarse mask fed as SAM's low-res shape prompt) returns a crisp,
    non-empty instance mask. Verifies the transformers 4.57 SAM2 ``input_masks`` API."""
    from active_segmenter.refine.sam import SamRefiner

    H = 128
    yy, xx = np.mgrid[0:H, 0:H]
    disc = ((yy - 64) ** 2 + (xx - 64) ** 2) < 25 ** 2
    img = np.stack([disc * 200] * 3, -1).astype(np.uint8)
    coarse = np.zeros((H, H), bool)
    coarse[44:84, 44:84] = True
    coarse_iou = (coarse & disc).sum() / (coarse | disc).sum()

    r = SamRefiner(RefineConfig(kind="sam", prompt_mode="mask"), device="cuda")
    out = r.refine(img, [InstanceMask(mask=coarse, points=None, class_id=1, instance_id=0)])
    assert len(out) == 1 and out[0].mask.shape == (H, H) and out[0].mask.any()
    assert (out[0].mask & disc).sum() / (out[0].mask | disc).sum() > coarse_iou


@pytest.mark.gpu
def test_amodal_mode_preserves_overlap():
    """Two heavily overlapping discs, amodal mode: each instance keeps its full extent so the
    two refined masks SHARE pixels (overlap preserved), unlike the negatives-separated mode."""
    from active_segmenter.refine.sam import SamRefiner

    H = 128
    yy, xx = np.mgrid[0:H, 0:H]
    d1 = ((yy - 64) ** 2 + (xx - 56) ** 2) < 26 ** 2
    d2 = ((yy - 64) ** 2 + (xx - 72) ** 2) < 26 ** 2
    img = np.stack([np.clip(d1 * 150 + d2 * 150, 0, 255)] * 3, -1).astype(np.uint8)
    inst = [
        InstanceMask(mask=d1, points=None, class_id=1, instance_id=0),
        InstanceMask(mask=d2, points=None, class_id=1, instance_id=1),
    ]
    r = SamRefiner(RefineConfig(kind="sam", prompt_mode="mask", amodal=True), device="cuda")
    out = r.refine(img, inst)
    assert len(out) == 2
    assert np.logical_and(out[0].mask, out[1].mask).any()  # overlap preserved


@pytest.mark.gpu
def test_sam_refine_two_instances_stay_separate():
    from active_segmenter.refine.sam import SamRefiner

    H = 128
    yy, xx = np.mgrid[0:H, 0:H]
    d1 = ((yy - 40) ** 2 + (xx - 40) ** 2) < 18 ** 2
    d2 = ((yy - 88) ** 2 + (xx - 88) ** 2) < 18 ** 2
    img = np.stack([(d1 | d2) * 200] * 3, -1).astype(np.uint8)
    inst = [
        InstanceMask(mask=d1, points=None, class_id=1, instance_id=0),
        InstanceMask(mask=d2, points=None, class_id=1, instance_id=1),
    ]
    r = SamRefiner(RefineConfig(kind="sam"), device="cuda")
    out = r.refine(img, inst)
    assert len(out) == 2
    # each refined mask covers its own disc, not the other
    assert (out[0].mask & d1).sum() > (out[0].mask & d2).sum()
    assert (out[1].mask & d2).sum() > (out[1].mask & d1).sum()
