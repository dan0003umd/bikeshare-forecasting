"""Origin-safety tests for the daily XGBoost feature design."""
import numpy as np
import pandas as pd

from src import config as cfg
from src.xgboost_baseline import (
    LEAK_FREE_LAGS,
    SHORT_LAG_ABLATION_LAGS,
    demand_feature_source_timestamps,
    lag_leakage_mask,
    xgboost_features,
)


def _station_hours_through_fall_dst() -> pd.DataFrame:
    # This UTC sequence contains all 25 local hours on the 2025 fall-back day,
    # including both 01:00 observations, as well as enough prior demand for 168h.
    ts = pd.date_range("2025-10-20T04:00:00Z", "2025-11-03T05:00:00Z", freq="h")
    local = ts.tz_convert(cfg.LOCAL_TZ)
    return pd.DataFrame({
        "station_id": "A", "ts_utc": ts, "checkouts": np.arange(len(ts), dtype=float),
        "hour": local.hour, "weekday": local.weekday, "month": local.month,
        "is_us_holiday": False, "event_flag": False,
    })


def test_leak_free_lags_are_strictly_before_every_daily_origin_including_dst():
    table = _station_hours_through_fall_dst()
    local = table.ts_utc.dt.tz_convert(cfg.LOCAL_TZ)
    dst_rows = table.loc[local.dt.date == pd.Timestamp("2025-11-02").date()]
    # The fall-back date exercises every displayed target hour 0..23, with an
    # additional repeated 01:00 observation.
    assert set(local.loc[dst_rows.index].dt.hour) == set(range(24))

    source = demand_feature_source_timestamps(table, LEAK_FREE_LAGS)
    origins = local.dt.normalize()
    for hour in range(24):
        rows = dst_rows.index[local.loc[dst_rows.index].dt.hour == hour]
        assert len(rows) >= 1
        for lag in LEAK_FREE_LAGS:
            used = source.loc[rows, f"lag_{lag}"].dropna()
            assert (used < origins.loc[used.index]).all(), f"hour={hour}, lag={lag} crossed origin"
    assert not lag_leakage_mask(table, LEAK_FREE_LAGS).to_numpy().any()
    features, columns = xgboost_features(table)
    assert [c for c in columns if c.startswith("lag_")] == ["lag_24", "lag_48", "lag_168"]
    assert features[columns].notna().any().all()


def test_historical_short_lag_ablation_fails_the_origin_check():
    table = _station_hours_through_fall_dst()
    leakage = lag_leakage_mask(table, SHORT_LAG_ABLATION_LAGS, short_lag_ablation=True)
    local = table.ts_utc.dt.tz_convert(cfg.LOCAL_TZ)
    dst_rows = leakage.loc[local.dt.date == pd.Timestamp("2025-11-02").date()]
    # On a forecast-day target after midnight, lag 1/2/3 consume realised
    # same-day demand, so the old ablation must fail this meaningful check.
    assert dst_rows[["lag_1", "lag_2", "lag_3"]].to_numpy().any()
    _, columns = xgboost_features(table, short_lag_ablation=True)
    assert [c for c in columns if c.startswith("lag_")] == ["lag_1", "lag_2", "lag_3", "lag_24", "lag_168"]
