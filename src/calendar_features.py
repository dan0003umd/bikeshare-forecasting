"""Local-time calendar and event features with a documented event-file schema."""
from __future__ import annotations

import pandas as pd
import holidays

from . import config as cfg


def load_events(path=cfg.EXTERNAL_DIR / "events.csv") -> pd.DataFrame:
    """Load verified events; an absent file means an all-zero event feature."""
    if not path.exists():
        return pd.DataFrame(columns=["date", "event_flag"])
    events = pd.read_csv(path, dtype={"date": "string"})
    needed = {"date", "event_flag"}
    if not needed <= set(events):
        raise ValueError(f"{path} needs columns {sorted(needed)}")
    events["event_flag"] = events["event_flag"].astype(bool)
    return events[["date", "event_flag"]].drop_duplicates("date")


def add_calendar_features(table: pd.DataFrame, events_path=cfg.EXTERNAL_DIR / "events.csv") -> pd.DataFrame:
    """Add calendar columns derived in America/New_York, never from UTC dates."""
    out = table.copy()
    local = pd.to_datetime(out["ts_utc"], utc=True).dt.tz_convert(cfg.LOCAL_TZ)
    out["hour"] = local.dt.hour.astype("int8")
    out["weekday"] = local.dt.day_of_week.astype("int8")
    out["month"] = local.dt.month.astype("int8")
    local_dates = local.dt.date
    years = range(int(local.dt.year.min()), int(local.dt.year.max()) + 1)
    us_holidays = holidays.US(years=years)
    out["is_us_holiday"] = local_dates.map(lambda d: d in us_holidays).astype(bool)
    out["local_date"] = local.dt.strftime("%Y-%m-%d")
    events = load_events(events_path)
    out = out.merge(events, how="left", left_on="local_date", right_on="date", validate="many_to_one")
    out["event_flag"] = out["event_flag"].fillna(False).astype(bool)
    return out.drop(columns=["date"])
