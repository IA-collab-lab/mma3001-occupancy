import pandas as pd
from occupancy.io import clean_events, TZ


def _raw(rows):
    return pd.DataFrame(rows, columns=["floorspaceid", "collecteddate", "occupancystatus", "headcount"])


def test_bad_rows_dropped_and_counted():
    raw = _raw([
        ("a", "2023-11-24 00:36:52+11", "CurrentlyOccupied", 1),
        (None, "2023-11-24 00:37:00+11", "NotOccupied", 0),
        ("a", "not a date", "NotOccupied", 0),
        ("a", "2023-11-24 00:38:00+11", "Weird", 0),
    ])
    clean, rep = clean_events(raw)
    assert len(clean) == 1
    assert rep["null_room"] == 1 and rep["bad_time"] == 1 and rep["bad_status"] == 1


def test_status_mapping_and_headcount_checks():
    raw = _raw([
        ("a", "2023-11-24 00:00:00+11", "CurrentlyOccupied", 0),
        ("a", "2023-11-24 00:01:00+11", "RecentlyOccupied", 0),
        ("a", "2023-11-24 00:02:00+11", "NotOccupied", -3),
    ])
    clean, rep = clean_events(raw)
    assert clean["occupied"].tolist() == [1, 0, 0]
    assert rep["headcount_mismatch"] == 1
    assert rep["negative_headcount"] == 1


def test_timezone_conversion_across_daylight_saving():
    raw = _raw([
        ("a", "2023-11-24 00:36:52+11", "NotOccupied", 0),
        ("a", "2024-07-01 12:00:00+10", "NotOccupied", 0),
    ])
    clean, _ = clean_events(raw)
    local = clean["time"].dt.tz_convert(TZ)
    assert local.iloc[0].hour == 0 and local.iloc[0].utcoffset().total_seconds() == 11 * 3600
    assert local.iloc[1].hour == 12 and local.iloc[1].utcoffset().total_seconds() == 10 * 3600
