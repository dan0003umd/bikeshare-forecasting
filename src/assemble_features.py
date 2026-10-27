"""Join demand, observed weather, local calendar, and station metadata."""
from __future__ import annotations

import pandas as pd

from . import config as cfg
from .calendar_features import add_calendar_features
from .splits import development_end_utc


def assemble_model_table() -> tuple[pd.DataFrame, dict]:
    hourly_path = cfg.PROCESSED_DIR / "station_hourly.parquet"
    weather_path = cfg.PROCESSED_DIR / "weather_hourly.parquet"
    if not hourly_path.exists() or not weather_path.exists():
        raise FileNotFoundError("Run src.build_hourly and src.weather before assembling features.")
    hourly = pd.read_parquet(hourly_path)
    weather = pd.read_parquet(weather_path)
    table = hourly.merge(weather, on="ts_utc", how="left", validate="many_to_one")
    table = add_calendar_features(table)

    info = pd.read_csv(cfg.PROCESSED_DIR / "station_info.csv", dtype={"short_name": "string"})
    metadata = info[["short_name", "capacity", "lat", "lon"]].dropna(subset=["short_name"]).drop_duplicates("short_name")
    # Trip identifiers are Capital Bikeshare short_name values (e.g. 31000),
    # while GBFS station_id is a UUID.  This is the intentionally explicit map.
    metadata = metadata.rename(columns={"short_name": "station_id"})
    metadata["station_id"] = metadata["station_id"].astype("string")
    table["station_id"] = table["station_id"].astype("string")
    table = table.merge(metadata, on="station_id", how="left", validate="many_to_one")
    selected = table[["station_id", "capacity", "lat", "lon"]].drop_duplicates("station_id")
    selected.to_csv(cfg.PROCESSED_DIR / "station_metadata_mapping.csv", index=False)

    cutoff = development_end_utc()
    development = table[pd.to_datetime(table.ts_utc, utc=True) < cutoff].copy()
    locked_test = table[pd.to_datetime(table.ts_utc, utc=True) >= cutoff].copy()
    report = {
        "rows": len(development),
        "locked_test_rows": len(locked_test),
        "stations": int(table["station_id"].nunique()),
        "weather_missing_rows": int(table[["temperature_2m", "precipitation", "wind_speed_10m"]].isna().any(axis=1).sum()),
        "metadata_matched_stations": int(selected["capacity"].notna().sum()),
        "metadata_unmatched_stations": int(selected["capacity"].isna().sum()),
    }
    # Keep test rows physically separate so development loaders cannot inspect
    # them accidentally. The test file remains ignored by Git and is only for
    # an explicit final evaluation.
    cfg.LOCKED_TEST_DIR.mkdir(parents=True, exist_ok=True)
    development.to_parquet(cfg.MODEL_TABLE_PATH, index=False)
    locked_test.to_parquet(cfg.LOCKED_TEST_MODEL_TABLE_PATH, index=False)
    print(report)
    return development, report


if __name__ == "__main__":
    assemble_model_table()
