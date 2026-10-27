# Previous Runs Weather Coverage Probe

Each row is a 48-hour Jan 1--2 UTC request for the `*_previous_day1` series.
A series is usable only when all requested values are present.

| Year | Variable | Non-null / requested hours | Complete |
|---:|---|---:|---|
| 2023 | temperature_2m | 48 / 48 | True |
| 2023 | precipitation | 0 / 48 | False |
| 2023 | wind_speed_10m | 0 / 48 | False |
| 2024 | temperature_2m | 0 / 48 | False |
| 2024 | precipitation | 0 / 48 | False |
| 2024 | wind_speed_10m | 0 / 48 | False |
| 2025 | temperature_2m | 48 / 48 | True |
| 2025 | precipitation | 48 / 48 | True |
| 2025 | wind_speed_10m | 48 / 48 | True |
| 2026 | temperature_2m | 48 / 48 | True |
| 2026 | precipitation | 48 / 48 | True |
| 2026 | wind_speed_10m | 48 / 48 | True |

The 2023 precipitation and wind gaps make a full 2023--2025 forecast-weather design invalid. Observed weather is joined in the model table, but the reported first-run models use calendar features only; any future observed-weather ablation must be labelled ORACLE.
