"""Load and clean the occupancy event log."""
import pandas as pd

TZ = "Australia/Melbourne"
VALID_STATUS = ("CurrentlyOccupied", "RecentlyOccupied", "NotOccupied")
USECOLS = ["floorspaceid", "collecteddate", "occupancystatus", "headcount"]


def clean_events(raw):
    """Validate raw occupancy events and reduce them to compact columns.

    Args:
        raw (pd.DataFrame): Columns floorspaceid, collecteddate,
            occupancystatus and headcount.

    Returns:
        tuple: (clean, report). ``clean`` has room (str), time (UTC datetime),
        occupied (int8, 1 only for CurrentlyOccupied) and headcount (float32,
        NaN when missing or negative). ``report`` is a dict of row counts
        dropped or flagged, by reason.
    """
    time = pd.to_datetime(raw["collecteddate"], utc=True, errors="coerce", format="ISO8601")
    bad_room = raw["floorspaceid"].isna()
    bad_time = time.isna()
    bad_status = ~raw["occupancystatus"].isin(VALID_STATUS)
    keep = ~(bad_room | bad_time | bad_status)
    hc = pd.to_numeric(raw["headcount"], errors="coerce")
    neg = hc < 0
    hc = hc.mask(neg | hc.isna())
    clean = pd.DataFrame({
        "room": raw.loc[keep, "floorspaceid"].astype(str),
        "time": time[keep],
        "occupied": (raw.loc[keep, "occupancystatus"] == "CurrentlyOccupied").astype("int8"),
        "headcount": hc[keep].astype("float32"),
    })
    mismatch = (((clean["occupied"] == 1) & (clean["headcount"] == 0))
                | ((clean["occupied"] == 0) & (clean["headcount"] > 0)))
    report = {
        "rows_in": len(raw),
        "null_room": int(bad_room.sum()),
        "bad_time": int(bad_time.sum()),
        "bad_status": int(bad_status.sum()),
        "negative_headcount": int(neg.sum()),
        "headcount_mismatch": int(mismatch.sum()),
        "rows_kept": len(clean),
    }
    return clean, report


def csv_to_parquet(csv_path, out_path, chunksize=1_000_000):
    """Stream a large occupancy CSV into one compact, sorted parquet file.

    Only four columns are read, in chunks, so memory stays low. Duplicate
    (room, time) rows are removed, keeping the last.

    Args:
        csv_path (str): Raw occupancy CSV.
        out_path (str): Destination parquet path.
        chunksize (int): Rows per chunk.

    Returns:
        dict: Summed cleaning report plus ``duplicates_removed``.
    """
    parts, total = [], {}
    for chunk in pd.read_csv(csv_path, usecols=USECOLS, chunksize=chunksize,
                             dtype={"floorspaceid": "string", "occupancystatus": "string"}):
        clean, rep = clean_events(chunk)
        parts.append(clean)
        for k, v in rep.items():
            total[k] = total.get(k, 0) + v
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(["room", "time"], keep="last").sort_values(["room", "time"])
    df["room"] = df["room"].astype("category")
    df.to_parquet(out_path)
    total["duplicates_removed"] = total["rows_kept"] - len(df)
    return total
