import numpy as np
import pandas as pd
from occupancy.validate import chrono_split, scores, best_threshold, held_out_rooms


def test_split_is_chronological_with_no_label_overlap():
    idx = pd.date_range("2024-01-01", periods=2000, freq="15min", tz="Australia/Melbourne")
    ds = pd.DataFrame({"x": 1}, index=idx)
    h = 4
    tr, va, te = chrono_split(ds, h)
    gap = pd.Timedelta(minutes=15 * h)
    assert tr.index.max() + gap < va.index.min()
    assert (va.index.max() + gap) < te.index.min()
    assert len(tr) > len(va) > 0 and len(te) > 0


def test_scores_perfect_and_all_wrong():
    y = [0, 1, 1, 0]
    assert scores(y, y)["f1"] == 1.0
    assert scores(y, [1, 0, 0, 1])["f1"] == 0.0


def test_best_threshold_separates_classes():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = best_threshold(y, p)
    assert 0.3 < t <= 0.7


def test_held_out_rooms_reproducible_and_nonempty():
    rooms = [f"r{i}" for i in range(10)]
    assert held_out_rooms(rooms) == held_out_rooms(rooms)
    assert 1 <= len(held_out_rooms(rooms)) < 10
