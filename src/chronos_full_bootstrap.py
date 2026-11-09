"""Paired day-bootstrap comparisons involving full-window Chronos-2 (validation only)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .splits import assert_no_locked_test_rows

Q = np.asarray(cfg.QUANTILES)
QCOLS = [f"q{int(q * 100):02d}" for q in Q]
MEDIAN = QCOLS.index("q50")
FILES = {
    "chronos2_zero_shot": "chronos2_zero_shot_50stations_363days.parquet",
    "xgboost_shared_quantile": "xgboost_shared_quantile_validation_2025.parquet",
    "hf_transformer_daily_embed_noweather": "hf_transformer_daily_embed_noweather_validation_2025.parquet",
    "seasonal_naive_rolling_8week": "seasonal_naive_rolling_8week_validation_2025.parquet",
}
PAIRS = (
    ("xgboost_shared_quantile", "chronos2_zero_shot"),
    ("hf_transformer_daily_embed_noweather", "chronos2_zero_shot"),
    ("chronos2_zero_shot", "seasonal_naive_rolling_8week"),
)


def daily_metrics(name: str) -> tuple[pd.MultiIndex, pd.DataFrame]:
    frame = pd.read_parquet(cfg.PREDICTIONS_DIR / FILES[name])
    assert_no_locked_test_rows(frame)
    frame = frame.sort_values(["station_id", "ts_utc"]).reset_index(drop=True)
    if len(frame) != 435_600 or frame.duplicated(["station_id", "ts_utc"]).any():
        raise AssertionError(f"{name}: unexpected rows")
    actual = frame.actual.to_numpy(float)
    pred = frame[QCOLS].to_numpy(float)
    diff = actual[:, None] - pred
    pinball = np.maximum(Q * diff, (Q - 1) * diff).mean(axis=1)
    mae = np.abs(actual - pred[:, MEDIAN])
    date = pd.to_datetime(frame.ts_utc, utc=True).dt.tz_convert(cfg.LOCAL_TZ).dt.date
    daily = pd.DataFrame({"date": date, "pinball": pinball, "median_mae": mae}).groupby("date").mean()
    key = pd.MultiIndex.from_frame(frame[["station_id", "ts_utc"]])
    print(f"{name}: pinball {pinball.mean():.4f}, median MAE {mae.mean():.4f}, days {len(daily)}")
    return key, daily


def main(draws: int = 10_000) -> None:
    keys, daily = {}, {}
    for name in FILES:
        keys[name], daily[name] = daily_metrics(name)
    base = next(iter(keys.values()))
    for name, key in keys.items():
        if not key.equals(base):
            raise AssertionError(f"{name} rows do not align with the others")
    rng = np.random.default_rng(cfg.SEED)
    n_days = len(next(iter(daily.values())))
    resample = rng.integers(0, n_days, size=(draws, n_days))
    out = []
    for a, b in PAIRS:
        if not daily[a].index.equals(daily[b].index):
            raise AssertionError("day sets differ")
        for metric in ("pinball", "median_mae"):
            diff = (daily[a][metric] - daily[b][metric]).to_numpy()
            boot = diff[resample].mean(axis=1)
            low, high = np.percentile(boot, [2.5, 97.5])
            out.append({"comparison": f"{a}_minus_{b}", "metric": metric, "difference_a_minus_b": diff.mean(),
                        "ci95_low": low, "ci95_high": high, "days": n_days, "bootstrap_draws": draws,
                        "ci_includes_zero": bool(low <= 0 <= high)})
    frame = pd.DataFrame(out)
    path = cfg.REPORTS_DIR / "chronos_full_window_bootstrap.csv"
    frame.to_csv(path, index=False)
    print(frame.round(5).to_string(index=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()