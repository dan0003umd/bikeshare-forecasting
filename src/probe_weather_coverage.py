"""Document Previous Runs lead-1 weather coverage by year and variable."""
from __future__ import annotations

import pandas as pd
import requests

from . import config as cfg


VARIABLES = ("temperature_2m", "precipitation", "wind_speed_10m")


def main() -> None:
    rows = []
    for year in range(2023, 2027):
        requested = [f"{variable}_previous_day1" for variable in VARIABLES]
        response = requests.get(cfg.OPEN_METEO_PREVIOUS_RUNS_URL, params={
            "latitude": cfg.DC_LATITUDE, "longitude": cfg.DC_LONGITUDE,
            "start_date": f"{year}-01-01", "end_date": f"{year}-01-02", "timezone": "GMT",
            "hourly": ",".join(requested),
        }, timeout=120)
        response.raise_for_status()
        hourly = response.json()["hourly"]
        expected = len(hourly["time"])
        for variable, column in zip(VARIABLES, requested):
            nonnull = sum(value is not None for value in hourly.get(column, []))
            rows.append({"year": year, "variable": variable, "nonnull_hours": nonnull,
                         "requested_hours": expected, "complete": nonnull == expected})
    result = pd.DataFrame(rows)
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv = cfg.REPORTS_DIR / "weather_previous_runs_probe.csv"
    result.to_csv(csv, index=False)
    markdown = ["# Previous Runs Weather Coverage Probe", "",
                "Each row is a 48-hour Jan 1--2 UTC request for the `*_previous_day1` series.",
                "A series is usable only when all requested values are present.", "",
                "| Year | Variable | Non-null / requested hours | Complete |", "|---:|---|---:|---|"]
    markdown.extend(f"| {r.year} | {r.variable} | {r.nonnull_hours} / {r.requested_hours} | {r.complete} |"
                    for r in result.itertuples(index=False))
    markdown.extend(["", "The 2023 precipitation and wind gaps make a full 2023--2025 forecast-weather design invalid. "
                     "Observed weather is joined in the model table, but the reported first-run models use calendar features only; any future observed-weather ablation must be labelled ORACLE."])
    (cfg.REPORTS_DIR / "weather_probe.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
