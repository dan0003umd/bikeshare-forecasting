# Extending weather probe coverage

from __future__ import annotations

import time

import pandas as pd
import requests

from . import config as cfg

VARIABLES = ("temperature_2m", "precipitation", "wind_speed_10m")
LEADS = (1, 2)
FIRST_MONTH = ("2023-01")


def month_coverage(month: pd.Period) -> list[dict]:
    "Request one month full coverage and count non null values for each variable and lead time."
    columns = [f"{variable}_previous_day{lead}" for lead in LEADS for variable in VARIABLES]
    response = requests.get(
        cfg.OPEN_METEO_PREVIOUS_RUNS_URL,
        params={
            "latitude": cfg.DC_LATITUDE,
            "longitude": cfg.DC_LONGITUDE,
            "start_date": month.start_time.date().isoformat(),
            "end_date": month.end_time.date().isoformat(),
            "timezone": "GMT",
            "hourly": ",".join(columns),
        }, timeout=120)
    response.raise_for_status()
    hourly = response.json()["hourly"]
    requested = len(hourly["time"])
    rows = []
    for lead in LEADS:
        for variable in VARIABLES:
            values = hourly.get(f"{variable}_previous_day{lead}", [])
            nonnull = sum(value is not None for value in values)
            rows.append({"month": str(month), "variable": variable, "lead_days": lead, "requested_hours": requested, "nonnull": nonnull,
                         "pct_complete": round(100 * nonnull / requested, 1)})
    return rows

def main() -> None:
    last_full_month = pd.Timestamp.today().to_period("M") - 1
    months = pd.period_range(FIRST_MONTH, last_full_month, freq="M")
    rows = []
    for month in months:
        print(f"Requesting coverage for {month}...")
        rows.extend(month_coverage(month))
        time.sleep(1)

    results = pd.DataFrame(rows)
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.REPORTS_DIR / "weather_coverage_by_month.csv"
    results.to_csv(path, index=False)

    table = results.pivot_table(index="month", columns=["lead_days", "variable"], values="pct_complete")
    print("\nPercent of hours present, by month:")
    print(table.to_string())

    needed = results[results.month >= cfg.VALID_START[:7]]
    for lead in LEADS:
        complete = (needed.loc[needed.lead_days == lead, "pct_complete"] == 100).all()
        print(f"\nLead {lead} day(s): 2025 onward fully complete = {complete}")
    print(f"\nSaved {path}")

if __name__ == "__main__":
    main()