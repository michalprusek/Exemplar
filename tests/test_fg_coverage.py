import numpy as np

from active_segmenter.acquire.fg_coverage import fg_patch_novelty, fg_coverage_scores


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_novelty_zero_when_candidate_equals_bank():
    bank = _unit(np.eye(4)[:3])          # 3 orthonormal fg patches
    assert fg_patch_novelty(bank, bank) < 1e-6


def test_novelty_one_when_orthogonal_to_bank():
    bank = _unit(np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32))
    cand = _unit(np.array([[0, 0, 1, 0], [0, 0, 0, 1]], np.float32))  # orthogonal
    assert abs(fg_patch_novelty(cand, bank) - 1.0) < 1e-6


def test_empty_bank_is_all_novel_empty_candidate_is_zero():
    bank = np.zeros((0, 4), np.float32)
    cand = _unit(np.eye(4)[:2])
    assert fg_patch_novelty(cand, bank) == 1.0
    assert fg_patch_novelty(np.zeros((0, 4), np.float32), _unit(np.eye(4)[:2])) == 0.0


def test_fg_coverage_scores_prefers_the_novel_candidate():
    # bank fg lives near e0; candidate A overlaps it, candidate B is a new region (e2)
    bank_fg = _unit(np.array([[1, 0, 0], [0.9, 0.1, 0]], np.float32))
    cand_fg = {
        "A": _unit(np.array([[1, 0, 0]], np.float32)),   # already covered
        "B": _unit(np.array([[0, 0, 1]], np.float32)),   # new region
    }
    scores = fg_coverage_scores(cand_fg, bank_fg)
    assert scores["B"] > scores["A"]
