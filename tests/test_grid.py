import numpy as np
import pandas as pd
from occupancy.grid import build_grid


def _events(times, states):
    return pd.DataFrame({"room": "r1", "time": pd.to_datetime(times, utc=True),
                         "occupied": np.array(states, dtype="int8")})


def test_short_gap_is_forward_filled():
    g = build_grid(_events(["2024-03-01 00:00", "2024-03-01 06:00"], [1, 0]))
    assert g["r1"].loc[g.index[4]] == 1.0
    assert g["r1"].iloc[-1] == 0.0


def test_long_gap_becomes_missing():
    g = build_grid(_events(["2024-03-01 00:00", "2024-03-10 00:00"], [1, 0]), max_gap_hours=72)
    mid = g["r1"].iloc[len(g) // 2]
    assert np.isnan(mid)
    assert g["r1"].iloc[0] == 1.0 and g["r1"].iloc[-1] == 0.0


def test_index_is_melbourne_time():
    g = build_grid(_events(["2024-03-01 00:00"], [1]))
    assert str(g.index.tz) == "Australia/Melbourne"
