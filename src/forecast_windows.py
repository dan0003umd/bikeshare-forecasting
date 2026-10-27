"""Shared fixed-length validation windows: local-midnight days with 24 real hours."""
from __future__ import annotations

import pandas as pd

from . import config as cfg
from .splits import split_mask


def validation_window_mask(table: pd.DataFrame) -> pd.Series:
    """Mask rows in 24-hour local days of the locked 2025 validation split.

    Spring-forward (23-hour) and fall-back (25-hour) days are excluded so every
    model forecasts exactly 24 hourly values per station.  Fall-back hours are
    also excluded through the table's explicit dst_overlap flag.
    """
    ts = pd.to_datetime(table["ts_utc"], utc=True)
    local_date = ts.dt.tz_convert(cfg.LOCAL_TZ).dt.date
    candidates = split_mask(ts, "validation") & ~table["dst_overlap"].astype(bool)
    n_stations = table.loc[candidates, "station_id"].nunique()
    counts = table.loc[candidates].assign(local_date=local_date[candidates]).groupby("local_date").size()
    full_days = counts.index[counts.eq(24 * n_stations)]
    return candidates & local_date.isin(full_days)


def validation_window_summary(table: pd.DataFrame) -> dict:
    mask = validation_window_mask(table)
    n_stations = table.loc[mask, "station_id"].nunique()
    return {"validation_days": int(mask.sum() // (24 * n_stations)), "stations": int(n_stations),
            "rows": int(mask.sum()), "dst_or_non24_rows_excluded": int((~mask).sum())}
