"""Download observed and (when complete) one-day-ahead DC weather in UTC."""
from __future__ import annotations

import argparse
from datetime import date

import pandas as pd
import requests

from . import config as cfg

WEATHER_COLUMNS = ("temperature_2m", "precipitation", "wind_speed_10m")


def _request(url: str, params: dict) -> pd.DataFrame:
    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()
    hourly = response.json().get("hourly")
    if not hourly or "time" not in hourly:
        raise ValueError(f"Open-Meteo response has no hourly payload: {response.text[:300]}")
    out = pd.DataFrame(hourly)
    out["ts_utc"] = pd.to_datetime(out.pop("time"), utc=True)
    return out


def observed_weather(start: str, end: str) -> pd.DataFrame:
    """Observed/reanalysis weather; timestamps returned and persisted as UTC."""
    out = _request(cfg.OPEN_METEO_ARCHIVE_URL, {
        "latitude": cfg.DC_LATITUDE, "longitude": cfg.DC_LONGITUDE,
        "start_date": start, "end_date": end, "timezone": "GMT",
        "hourly": ",".join(WEATHER_COLUMNS),
    })
    return out[["ts_utc", *WEATHER_COLUMNS]]


def previous_run_weather(start: str, end: str) -> pd.DataFrame | None:
    """Return fixed one-day-lead forecasts, or None when the archive is incomplete.

    Open-Meteo's documented previous-run columns are the only acceptable source
    here: asking for the ordinary column would silently use a near-valid-time
    forecast and leak information.
    """
    requested = [f"{name}_previous_day1" for name in WEATHER_COLUMNS]
    out = _request(cfg.OPEN_METEO_PREVIOUS_RUNS_URL, {
        "latitude": cfg.DC_LATITUDE, "longitude": cfg.DC_LONGITUDE,
        "start_date": start, "end_date": end, "timezone": "GMT",
        "hourly": ",".join(requested),
    })
    if any(c not in out or out[c].isna().any() for c in requested):
        return None
    return out.rename(columns={f"{name}_previous_day1": f"{name}_fc1d" for name in WEATHER_COLUMNS})[
        ["ts_utc", *[f"{name}_fc1d" for name in WEATHER_COLUMNS]]
    ]


def build_weather(start: str = cfg.TRAIN_START, end: str | None = None) -> tuple[pd.DataFrame, bool]:
    """Build the weather table, deliberately falling back to observed-only data.

    The January-2023 one-day-lead probe is made first.  It has to be complete for
    all requested variables before forecast columns are retained for the range.
    """
    end = end or date.today().isoformat()
    # Required coverage test: GFS temperature exists in 2023, but precipitation
    # and wind do not, so partial columns would be misleading.
    probe = previous_run_weather("2023-01-01", "2023-01-02")
    observed = observed_weather(start, end)
    forecast_available = probe is not None
    if forecast_available:
        fc = previous_run_weather(start, end)
        forecast_available = fc is not None
        if forecast_available:
            observed = observed.merge(fc, on="ts_utc", how="left", validate="one_to_one")
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    observed.to_parquet(cfg.PROCESSED_DIR / "weather_hourly.parquet", index=False)
    print(f"{len(observed):,} UTC weather hours; forecast_weather_available={forecast_available}")
    if not forecast_available:
        print("Previous Runs forecast variables are incomplete for January 2023; using observed weather only.")
    return observed, forecast_available


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=cfg.TRAIN_START)
    parser.add_argument("--end", default=None, help="inclusive ISO date; defaults to today")
    args = parser.parse_args()
    build_weather(args.start, args.end)


if __name__ == "__main__":
    main()
