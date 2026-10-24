"""Baselines and learned models."""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES


def persistence(ds):
    """Baseline A: predict that the room stays in its current state."""
    return ds["now"].to_numpy().astype(int)


def same_time_last_week(ds):
    """Baseline B: predict the state seen at the target time one week earlier."""
    return ds["week_ago"].to_numpy().astype(int)


def fit_logreg(train):
    """Fit logistic regression (standardised features) on a training frame."""
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    return m.fit(train[FEATURES], train["target"].astype(int))


def fit_hgb(train, max_iter=200, max_depth=None, seed=0):
    """Fit histogram gradient boosting on a training frame."""
    m = HistGradientBoostingClassifier(max_iter=max_iter, max_depth=max_depth, random_state=seed)
    return m.fit(train[FEATURES], train["target"].astype(int))
