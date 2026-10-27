import numpy as np
import pandas as pd

from src.baselines import seasonal_naive_frozen_quantiles, seasonal_naive_rolling_quantiles


def test_seasonal_naive_does_not_use_validation_demand():
    # All rows share local Monday 00:00; the 2025 value must not enter its own
    # (or any later) forecast distribution.
    table = pd.DataFrame({
        "station_id": ["A", "A", "A"],
        "ts_utc": pd.to_datetime(["2024-12-16T05:00Z", "2024-12-23T05:00Z", "2025-01-06T05:00Z"]),
        "hour": [0, 0, 0], "weekday": [0, 0, 0], "checkouts": [1, 2, 999],
    })
    prediction = seasonal_naive_frozen_quantiles(table)
    assert np.nanmax(prediction[2]) <= 2


def test_rolling_seasonal_naive_uses_only_prior_same_hour_observations():
    table = pd.DataFrame({
        "station_id": ["A", "A", "A", "A"],
        "ts_utc": pd.to_datetime(["2024-12-09T05:00Z", "2024-12-16T05:00Z", "2024-12-23T05:00Z", "2025-01-06T05:00Z"]),
        "hour": [0, 0, 0, 0], "weekday": [0, 0, 0, 0], "checkouts": [1, 2, 3, 999],
    })
    prediction = seasonal_naive_rolling_quantiles(table)
    # The Jan 6 origin uses [1, 2, 3], never its own 999 actual value.
    assert np.nanmax(prediction[3]) <= 3
