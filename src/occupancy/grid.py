"""Turn the event log into a regular occupancy grid."""
import numpy as np
import pandas as pd

from .io import TZ


def build_grid(events, freq="15min", max_gap_hours=72):
    """Convert state-change events to a regular 0/1 series per room.

    The state of a bin is the last event inside it, carried forward until the
    next event. A run of empty bins longer than ``max_gap_hours`` is set to NaN
    (treated as missing, excluded from training and testing).

    Args:
        events (pd.DataFrame): Output of ``clean_events`` (room, time, occupied).
        freq (str): Grid spacing.
        max_gap_hours (float): Longest gap that is still filled.

    Returns:
        pd.DataFrame: Index = Australia/Melbourne timestamps, one float32
        column per room, values 0, 1 or NaN.
    """
    limit = int(pd.Timedelta(hours=max_gap_hours) / pd.Timedelta(freq))
    cols = {}
    for room, g in events.groupby("room", observed=True):
        s = g.set_index("time")["occupied"].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        raw = s.resample(freq).last()
        is_na = raw.isna()
        gap_len = is_na.groupby((~is_na).cumsum()).transform("sum")
        filled = raw.ffill().astype("float32")
        filled[(is_na & (gap_len > limit)).to_numpy()] = np.nan
        cols[room] = filled
    grid = pd.DataFrame(cols)
    grid.index = grid.index.tz_convert(TZ)
    return grid
