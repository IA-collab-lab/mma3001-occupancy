import numpy as np
import pandas as pd
import pytest
from occupancy.features import make_dataset, FEATURES


def _grid(n=3 * 672, seed=0):
    idx = pd.date_range("2024-03-04", periods=n, freq="15min", tz="Australia/Melbourne")
    y = np.random.default_rng(seed).integers(0, 2, n).astype("float32")
    return pd.DataFrame({"r1": y}, index=idx)


def test_target_and_features_use_correct_times():
    g, h = _grid(), 4
    ds = make_dataset(g, h)
    t = ds.index[100]
    pos = g.index.get_loc(t)
    assert ds.loc[t, "now"] == g["r1"].iloc[pos]
    assert ds.loc[t, "lag2"] == g["r1"].iloc[pos - 2]
    assert ds.loc[t, "target"] == g["r1"].iloc[pos + h]
    assert ds.loc[t, "week_ago"] == g["r1"].iloc[pos + h - 672]
    assert ds.loc[t, "day_ago"] == g["r1"].iloc[pos + h - 96]


def test_features_never_look_past_origin_time():
    g, h = _grid(), 4
    ds = make_dataset(g, h)
    g2 = g.copy()
    cut = g.index[1000]
    g2.loc[g2.index > cut, "r1"] = 1 - g2.loc[g2.index > cut, "r1"]  # change the future only
    ds2 = make_dataset(g2, h)
    common = ds.index[(ds.index <= cut) & (ds.index.isin(ds2.index))]
    pd.testing.assert_frame_equal(ds.loc[common, FEATURES], ds2.loc[common, FEATURES])


def test_gaps_are_dropped_and_horizon_validated():
    g = _grid()
    g.iloc[900:910, 0] = np.nan
    ds = make_dataset(g, 4)
    assert ds.notna().all().all()
    with pytest.raises(ValueError):
        make_dataset(g, 0)
