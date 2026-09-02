import numpy as np

from active_segmenter.eval.scoring import PRIMARY, primary_key, score_prediction


def test_primary_key_maps_each_metric():
    assert primary_key("iou") == "fg_iou"
    assert primary_key("cldice") == "cldice"
    assert primary_key("instance_ap") == "ap"
    assert primary_key("unknown") == "fg_iou"          # safe default
    assert set(PRIMARY) == {"iou", "cldice", "instance_ap"}


def test_score_iou_reports_fgiou_and_bf():
    fg = np.zeros((16, 16), bool); fg[4:12, 4:12] = True
    lbl = np.zeros((16, 16), int); lbl[4:12, 4:12] = 1
    out = score_prediction("iou", fg, lbl)
    assert out["fg_iou"] == 1.0 and "bf" in out and "cldice" not in out


def test_score_cldice_adds_cldice_for_tubular():
    m = np.zeros((20, 20), bool); m[9:12, 2:18] = True
    out = score_prediction("cldice", m, m.astype(int))
    assert out["cldice"] == 1.0


def test_score_instance_ap_uses_instances():
    lbl = np.zeros((10, 10), int); lbl[1:4, 1:4] = 1; lbl[6:9, 6:9] = 2  # two instances
    inst = [lbl == 1, lbl == 2]
    out = score_prediction("instance_ap", lbl > 0, lbl, instances=inst)
    assert out["ap"] == 1.0
    # no instances provided -> ap 0 (not a crash)
    assert score_prediction("instance_ap", lbl > 0, lbl, instances=[])["ap"] == 0.0
