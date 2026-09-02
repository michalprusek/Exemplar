import numpy as np

from active_segmenter.eval import metrics


def test_foreground_iou_perfect():
    gt = np.array([[0, 1], [2, 0]])
    pred = gt > 0
    assert metrics.foreground_iou(pred, gt) == 1.0


def test_foreground_iou_half():
    gt = np.array([[1, 1], [0, 0]])
    pred = np.array([[1, 0], [0, 0]], bool)
    assert abs(metrics.foreground_iou(pred, gt) - 0.5) < 1e-9


def test_foreground_iou_empty_both_is_one():
    gt = np.zeros((3, 3), int)
    pred = np.zeros((3, 3), bool)
    assert metrics.foreground_iou(pred, gt) == 1.0


def test_instance_ap_perfect():
    gt = np.array([[1, 1, 0], [0, 2, 2]])
    preds = [gt == 1, gt == 2]
    out = metrics.instance_ap(preds, gt)
    assert out["ap50"] == 1.0


def test_instance_ap_missed_one():
    gt = np.array([[1, 1, 0], [0, 2, 2]])
    preds = [gt == 1]  # missed instance 2 -> 1 TP, 1 FN at IoU .5
    out = metrics.instance_ap(preds, gt)
    assert abs(out["ap50"] - 0.5) < 1e-9


def test_instance_ap_false_positive():
    gt = np.array([[1, 1, 0], [0, 0, 0]])
    preds = [gt == 1, np.array([[0, 0, 1], [0, 0, 0]], bool)]  # 1 TP + 1 FP
    out = metrics.instance_ap(preds, gt)
    assert abs(out["ap50"] - 0.5) < 1e-9  # precision 1/2 at the single detection level


def test_panoptic_perfect():
    gt = np.array([[1, 1, 0], [0, 2, 2]])
    preds = [gt == 1, gt == 2]
    out = metrics.panoptic_sq_rq(preds, gt)
    assert abs(out["rq"] - 1.0) < 1e-9
    assert out["sq"] > 0.99


def test_overlap_gt_as_mask_list():
    # overlap-capable GT: two instances sharing a pixel, given as a mask list
    m1 = np.array([[1, 1, 0], [0, 0, 0]], bool)
    m2 = np.array([[0, 1, 1], [0, 0, 0]], bool)  # shares (0,1) with m1
    out = metrics.instance_ap([m1, m2], gt_masks=[m1, m2])
    assert out["ap50"] == 1.0


def test_boundary_f1_perfect_is_one():
    m = np.zeros((32, 32), bool)
    m[8:24, 8:24] = True
    assert metrics.boundary_f1(m, m, tol=2) == 1.0


def test_boundary_f1_shifted_within_tol_high():
    a = np.zeros((40, 40), bool)
    a[10:30, 10:30] = True
    b = np.zeros((40, 40), bool)
    b[11:31, 11:31] = True  # 1px shift, tol=2 -> near 1
    assert metrics.boundary_f1(a, b, tol=2) > 0.8


def test_boundary_f1_disjoint_is_zero():
    a = np.zeros((40, 40), bool)
    a[2:8, 2:8] = True
    b = np.zeros((40, 40), bool)
    b[30:36, 30:36] = True
    assert metrics.boundary_f1(a, b, tol=2) == 0.0


def test_cldice_perfect_is_one():
    m = np.zeros((32, 32), bool); m[10:22, 4:28] = True  # a bar
    assert metrics.cldice(m, m) == 1.0


def test_cldice_disjoint_is_zero():
    a = np.zeros((32, 32), bool); a[2:6, 2:6] = True
    b = np.zeros((32, 32), bool); b[24:28, 24:28] = True
    assert metrics.cldice(a, b) == 0.0


def test_cldice_drops_when_a_tube_is_broken():
    # a thick horizontal tube; cutting a gap breaks its connectivity -> clDice must drop
    # below the intact (perfect) score, which is the topology-sensitivity clDice exists for.
    gt = np.zeros((21, 21), bool); gt[8:13, 2:19] = True
    broken = gt.copy(); broken[8:13, 9:12] = False   # cut across the tube
    assert metrics.cldice(gt, gt) == 1.0
    assert metrics.cldice(broken, gt) < 1.0
