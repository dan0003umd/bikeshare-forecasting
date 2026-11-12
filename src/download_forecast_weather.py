from __future__ import annotations

import time

import pandas as pd
import requests

from . import config as cfg

VARIABLES = ("temperature_2m", "precipitation", "wind_speed_10m")
LEADS = (1, 2)
OUTPUT = cfg.PROCESSED_DIR / "weather_forecast_2025_2026.parquet"

def forecast_month(month: pd.Period) -> pd.DataFrame:
    "Request one month of forecast data and return as a DataFrame."
    columns = [f"{variable}_previous_day{lead}" for lead in LEADS for variable in VARIABLES]
    response = requests.get(cfg.OPEN_METEO_PREVIOUS_RUNS_URL, params={
        "latitude": cfg.DC_LATITUDE,
        "longitude": cfg.DC_LONGITUDE,
        "start_date": month.start_time.date().isoformat(),
        "end_date": month.end_time.date().isoformat(),
        "timezone": "GMT", "hourly": ",".join(columns)
    }, timeout=120)
    response.raise_for_status()
    out = pd.DataFrame(response.json()["hourly"])
    out["ts_utc"] = pd.to_datetime(out.pop("time"), utc=True)
    return out.rename(columns={f"{variable}_previous_day{lead}": f"{variable}_fc{lead}d"
                               for lead in LEADS for variable in VARIABLES})

def main() -> None:
    last_full_month = pd.Timestamp.today().to_period("M") - 1
    months = pd.period_range(cfg.VALID_START[:7], last_full_month, freq="M")
    pieces = []
    for month in months:
        print(f"downloading {month}...")
        pieces.append(forecast_month(month))
        time.sleep(1)
    weather = pd.concat(pieces, ignore_index=True).sort_values("ts_utc").reset_index(drop=True)

    expected = pd.date_range(weather.ts_utc.min(), weather.ts_utc.max(), freq="h")
    if len (weather) != len(expected) or not weather.ts_utc.reset_index(drop = True).equals(pd.Series(expected)):
        raise RuntimeError("forecast hours are missing or duplicated")
    missing = weather.drop(columns="ts_utc").isna().sum()
    if missing.any():
        raise RuntimeError(f"forecast values are missing: {missing[missing > 0].to_dict()}") 

    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(OUTPUT, index=False)
    print(f"\n{len(weather):,} hours, {weather.ts_utc.min()} to {weather.ts_utc.max()}")
    print(weather.head(3).to_string(index=False))
    print(f"\nsaved {OUTPUT}")


if __name__ == "__main__":
    main()