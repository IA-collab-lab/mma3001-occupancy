"""Feature construction. Every feature uses only information available at time t."""
import numpy as np
import pandas as pd

STEP_MIN = 15
STEPS_PER_DAY = 96
STEPS_PER_WEEK = 672
FEATURES = ["now", "lag1", "lag2", "lag4", "lag8", "mean_1h", "mean_4h",
            "day_ago", "week_ago", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]


def make_dataset(grid, horizon=4):
    """Build a supervised dataset: predict occupancy ``horizon`` steps ahead.

    A row at time t holds features known at t and the target y[t + horizon].
    ``day_ago`` and ``week_ago`` are the states 24 h and 7 d before the target
    time. They lie in the past of t, so there is no leakage. Rows with any NaN
    (gaps, first week of each room) are dropped.

    Args:
        grid (pd.DataFrame): Output of ``build_grid``.
        horizon (int): Steps ahead (4 = 1 hour at 15 min). Range 1 to 96.

    Returns:
        pd.DataFrame: Index = origin time t; columns FEATURES, target, room.

    Raises:
        ValueError: If horizon is outside 1..96.
    """
    if not 1 <= horizon <= STEPS_PER_DAY:
        raise ValueError("horizon must be between 1 and 96 steps")
    tt = grid.index + pd.Timedelta(minutes=STEP_MIN * horizon)
    hour = (tt.hour + tt.minute / 60).to_numpy()
    dow = tt.dayofweek.to_numpy()
    cyc = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
    }, index=grid.index).astype("float32")
    frames = []
    for room in grid.columns:
        y = grid[room]
        d = pd.DataFrame({"now": y}, index=grid.index)
        for k in (1, 2, 4, 8):
            d[f"lag{k}"] = y.shift(k)
        d["mean_1h"] = y.rolling(4).mean()
        d["mean_4h"] = y.rolling(16).mean()
        d["day_ago"] = y.shift(STEPS_PER_DAY - horizon)
        d["week_ago"] = y.shift(STEPS_PER_WEEK - horizon)
        d = d.join(cyc)
        d["target"] = y.shift(-horizon)
        d["room"] = room
        frames.append(d.dropna())
    return pd.concat(frames)
