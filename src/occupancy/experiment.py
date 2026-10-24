"""End-to-end experiment: baselines, models, held-out-room test.

Run without --final to score on validation data only. Use --final once, after
all choices are fixed, to score on the test period.
"""
import argparse
import time

import pandas as pd

from . import models
from .features import FEATURES, make_dataset
from .grid import build_grid
from .validate import best_threshold, chrono_split, held_out_rooms, scores


def run(parquet, horizon=4, max_train_rows=2_000_000, final=False, out_dir="reports"):
    """Run the full comparison and save CSV tables.

    Args:
        parquet (str): Cleaned events from ``csv_to_parquet``.
        horizon (int): Steps ahead (4 = 1 hour).
        max_train_rows (int): Training rows are subsampled above this.
        final (bool): Score on test data if True, validation data otherwise.
        out_dir (str): Folder for result CSVs.

    Returns:
        pd.DataFrame: One row per model with metrics and timings.
    """
    events = pd.read_parquet(parquet)
    ds = make_dataset(build_grid(events), horizon)
    train, val, test = chrono_split(ds, horizon)
    if len(train) > max_train_rows:
        train = train.sample(max_train_rows, random_state=0)
    ev, split = (test, "test") if final else (val, "val")
    y = ev["target"].astype(int)
    rows = []

    def add(name, y_true, pred, prob=None, **extra):
        r = scores(y_true, pred, prob)
        r.update(model=name, split=split, **extra)
        rows.append(r)

    add("persistence", y, models.persistence(ev))
    add("last_week", y, models.same_time_last_week(ev))
    fitted, thr = {}, {}
    for name, fit in (("logreg", models.fit_logreg), ("hgb", models.fit_hgb)):
        t0 = time.perf_counter()
        m = fit(train)
        fit_s = time.perf_counter() - t0
        thr[name] = best_threshold(val["target"].astype(int), m.predict_proba(val[FEATURES])[:, 1])
        t0 = time.perf_counter()
        prob = m.predict_proba(ev[FEATURES])[:, 1]
        pred_s = time.perf_counter() - t0
        add(name, y, (prob >= thr[name]).astype(int), prob, fit_s=fit_s, predict_s=pred_s,
            threshold=thr[name])
        fitted[name] = (m, prob)

    # Held-out rooms: train without them, score on them only.
    hold = held_out_rooms(ds["room"].unique())
    sub = ev["room"].isin(hold).to_numpy()
    ysub = y[sub]
    add("persistence_heldout_rooms", ysub, models.persistence(ev)[sub])
    add("last_week_heldout_rooms", ysub, models.same_time_last_week(ev)[sub])
    add("hgb_seen_rooms_same_subset", ysub, (fitted["hgb"][1][sub] >= thr["hgb"]).astype(int),
        fitted["hgb"][1][sub])
    tr_h = train[~train["room"].isin(hold)]
    val_h = val[~val["room"].isin(hold)]
    mh = models.fit_hgb(tr_h)
    th = best_threshold(val_h["target"].astype(int), mh.predict_proba(val_h[FEATURES])[:, 1])
    ph = mh.predict_proba(ev.loc[sub, FEATURES])[:, 1]
    add("hgb_heldout_rooms", ysub, (ph >= th).astype(int), ph, threshold=th)

    res = pd.DataFrame(rows)
    per_room = []
    prob = fitted["hgb"][1]
    for room, idx in ev.reset_index(drop=True).groupby("room").groups.items():
        yt = y.to_numpy()[idx]
        if yt.sum() == 0:
            continue
        r = scores(yt, (prob[idx] >= thr["hgb"]).astype(int), prob[idx])
        r["room"] = room
        per_room.append(r)
    pd.DataFrame(per_room).to_csv(f"{out_dir}/per_room_{split}_h{horizon}.csv", index=False)
    res.to_csv(f"{out_dir}/results_{split}_h{horizon}.csv", index=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/processed/events.parquet")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    out = run(a.parquet, a.horizon, final=a.final)
    pd.set_option("display.width", 200)
    print(out[["model", "split", "n", "prevalence", "precision", "recall", "f1", "pr_auc"]].round(3))
