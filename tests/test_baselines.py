import numpy as np
import pandas as pd

from src.baselines import seasonal_naive_quantiles


def test_seasonal_naive_does_not_use_validation_demand():
    # All rows share local Monday 00:00; the 2025 value must not enter its own
    # (or any later) forecast distribution.
    table = pd.DataFrame({
        "station_id": ["A", "A", "A"],
        "ts_utc": pd.to_datetime(["2024-12-16T05:00Z", "2024-12-23T05:00Z", "2025-01-06T05:00Z"]),
        "hour": [0, 0, 0], "weekday": [0, 0, 0], "checkouts": [1, 2, 999],
    })
    prediction = seasonal_naive_quantiles(table)
    assert np.nanmax(prediction[2]) <= 2
