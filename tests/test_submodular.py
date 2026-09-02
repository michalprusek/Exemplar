import numpy as np

from active_segmenter.acquire.submodular import facility_location_greedy


def test_picks_highest_coverage_first():
    # pool of 3 items; candidate C covers all 3 strongly, A covers {0,1}, B covers {1,2}
    #             cand:  A     B     C
    sim = np.array([[0.9, 0.0, 0.8],    # pool 0
                    [0.9, 0.9, 0.8],    # pool 1
                    [0.0, 0.9, 0.8]], np.float32)  # pool 2
    w = np.ones(3, np.float32)
    picked = facility_location_greedy(sim, w, k=1)
    assert picked == [2]  # C maximizes total coverage


def test_greedy_diversifies_not_duplicates():
    # candidates 0 and 1 are identical (redundant); 2 covers a different region
    sim = np.array([[1.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]], np.float32)
    w = np.ones(3, np.float32)
    picked = facility_location_greedy(sim, w, k=2)
    # after taking one of {0,1}, the other adds ~0 gain -> the diverse candidate 2 is taken
    assert 2 in picked
    assert not (0 in picked and 1 in picked)


def test_weights_bias_selection_to_high_error_region():
    # pool item 2 has all the weight (error); candidate B covers it, A does not
    sim = np.array([[1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0]], np.float32)   # cand A covers {0,1}, B covers {2}
    w = np.array([0.0, 0.0, 1.0], np.float32)  # all error mass on pool item 2
    picked = facility_location_greedy(sim, w, k=1)
    assert picked == [1]  # B covers the high-error item


def test_k_capped_at_num_candidates():
    sim = np.ones((3, 2), np.float32)
    picked = facility_location_greedy(sim, np.ones(3, np.float32), k=5)
    assert len(picked) == 2 and sorted(picked) == [0, 1]
