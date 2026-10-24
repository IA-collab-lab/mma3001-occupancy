"""Chronological splitting and metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)


def chrono_split(ds, horizon, train_frac=0.70, val_frac=0.15, step_min=15):
    """Split by time (never shuffled) with a label-leak buffer.

    Train labels all fall before the validation period, and validation labels
    before the test period.

    Args:
        ds (pd.DataFrame): Dataset indexed by origin time.
        horizon (int): Prediction horizon in steps.
        train_frac (float): Fraction of the time span used for training.
        val_frac (float): Fraction used for validation. The rest is test.
        step_min (int): Minutes per step.

    Returns:
        tuple: (train, val, test) DataFrames.
    """
    t0, t1 = ds.index.min(), ds.index.max()
    span = t1 - t0
    c1 = t0 + span * train_frac
    c2 = t0 + span * (train_frac + val_frac)
    gap = pd.Timedelta(minutes=step_min * horizon)
    train = ds[ds.index + gap < c1]
    val = ds[(ds.index >= c1) & (ds.index + gap < c2)]
    test = ds[ds.index >= c2]
    return train, val, test


def scores(y_true, y_pred, y_prob=None):
    """Binary classification metrics.

    Args:
        y_true: True 0/1 labels.
        y_pred: Predicted 0/1 labels.
        y_prob: Optional probabilities, used for PR-AUC.

    Returns:
        dict: n, prevalence, precision, recall, f1, pr_auc, tn, fp, fn, tp.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pr_auc = np.nan
    if y_prob is not None and y_true.any():
        pr_auc = average_precision_score(y_true, y_prob)
    return {"n": len(y_true), "prevalence": float(y_true.mean()),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0), "pr_auc": pr_auc,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def best_threshold(y_true, prob):
    """Probability threshold that maximises F1 (tune on validation data only)."""
    p, r, t = precision_recall_curve(y_true, prob)
    if len(t) == 0:
        return 0.5
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[int(np.argmax(f1))])


def held_out_rooms(rooms, frac=0.3, seed=0):
    """Pick a reproducible random subset of rooms to hold out entirely."""
    rooms = np.array(sorted(rooms))
    rng = np.random.default_rng(seed)
    k = max(1, int(round(len(rooms) * frac)))
    return list(rng.permutation(rooms)[:k])
