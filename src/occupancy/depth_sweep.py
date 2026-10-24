"""Tree depth sweep for gradient boosting.

Compares training and validation F1 at several tree depths to look for
overfitting. Uses validation data only. The test period is not touched.
"""
import argparse

import pandas as pd

from . import models
from .features import FEATURES, make_dataset
from .grid import build_grid
from .validate import best_threshold, chrono_split, scores


def run(parquet, horizon=4, depths=(2, 4, 8, None), out_dir="reports"):
    """Fit gradient boosting at each maximum tree depth and score it.

    For each depth the decision threshold is tuned on validation data, then
    the same threshold is used to score the training and validation sets.

    Args:
        parquet (str): Cleaned events from ``csv_to_parquet``.
        horizon (int): Steps ahead (4 = 1 hour).
        depths (tuple): Maximum tree depths to try. None means no limit.
        out_dir (str): Folder for the result CSV.

    Returns:
        pd.DataFrame: One row per depth with training and validation scores.
    """
    events = pd.read_parquet(parquet)
    ds = make_dataset(build_grid(events), horizon)
    train, val, _ = chrono_split(ds, horizon)
    y_tr = train["target"].astype(int)
    y_val = val["target"].astype(int)
    rows = []
    for d in depths:
        m = models.fit_hgb(train, max_depth=d)
        p_val = m.predict_proba(val[FEATURES])[:, 1]
        p_tr = m.predict_proba(train[FEATURES])[:, 1]
        thr = best_threshold(y_val, p_val)
        s_tr = scores(y_tr, (p_tr >= thr).astype(int), p_tr)
        s_val = scores(y_val, (p_val >= thr).astype(int), p_val)
        rows.append({"max_depth": "none" if d is None else d,
                     "trees_fitted": getattr(m, "n_iter_", None),
                     "threshold": thr,
                     "train_f1": s_tr["f1"], "val_f1": s_val["f1"],
                     "f1_gap": s_tr["f1"] - s_val["f1"],
                     "val_recall": s_val["recall"], "val_pr_auc": s_val["pr_auc"]})
    res = pd.DataFrame(rows)
    res.to_csv(f"{out_dir}/depth_sweep_h{horizon}.csv", index=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/processed/events.parquet")
    ap.add_argument("--horizon", type=int, default=4)
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    print(run(a.parquet, a.horizon).round(3))
