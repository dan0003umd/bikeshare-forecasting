"""One shared, origin-safe 19-quantile XGBoost baseline fitted through 2024."""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from . import config as cfg
from .forecast_windows import validation_window_mask, validation_window_summary
from .metrics import score
from .splits import add_final_argument, load_development_model_table, load_table_through_test, split_mask


# A daily origin predicts all 24 hours of the forecast day from information
# strictly before local midnight.  Lags shorter than 24 hours would expose
# realised demand from earlier target hours to later target-hour rows.
LEAK_FREE_LAGS = (24, 48, 168)
SHORT_LAG_ABLATION_LAGS = (1, 2, 3, 24, 168)
LEAK_FREE_MODEL = "xgboost_shared_quantile"
SHORT_LAG_ABLATION_MODEL = "xgboost_short_lag_not_deployable"


def load_pretest_table() -> pd.DataFrame:
    """Use the shared guarded loader, then establish deterministic station order."""
    return load_development_model_table().sort_values(["station_id", "ts_utc"]).reset_index(drop=True)


def _daily_aligned_lags(table: pd.DataFrame, lags: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return lag values and source timestamps at the same local hour on prior days.

    This uses local calendar days, not a fixed count of UTC rows. That distinction
    is necessary at fall-back: a 24-row shift for 23:00 can point to 00:00 of
    the *same* 25-hour local day. If a source day has a repeated local hour, its
    later physical observation is selected deterministically; either candidate
    is still strictly before the following forecast-day origin.
    """
    work = pd.DataFrame({
        "row": np.arange(len(table)),
        "station_id": table["station_id"].astype(str).to_numpy(),
        "ts_utc": pd.to_datetime(table["ts_utc"], utc=True).to_numpy(),
        "value": table["checkouts"].to_numpy(),
    })
    local = work["ts_utc"].dt.tz_convert(cfg.LOCAL_TZ)
    work["local_date"] = local.dt.tz_localize(None).dt.normalize()
    work["hour"] = local.dt.hour
    lookup = (work.sort_values("ts_utc")
              .drop_duplicates(["station_id", "local_date", "hour"], keep="last")
              [["station_id", "local_date", "hour", "ts_utc", "value"]])
    values, timestamps = {}, {}
    for lag in lags:
        if lag % 24:
            raise ValueError(f"daily origin-safe lag must be a multiple of 24, got {lag}")
        keys = work[["row", "station_id", "hour"]].copy()
        keys["local_date"] = work["local_date"] - pd.Timedelta(days=lag // 24)
        merged = keys.merge(lookup, on=["station_id", "local_date", "hour"], how="left", sort=False)
        merged = merged.sort_values("row")
        values[f"lag_{lag}"] = merged["value"].to_numpy()
        timestamps[f"lag_{lag}"] = pd.to_datetime(merged["ts_utc"], utc=True).to_numpy()
    return pd.DataFrame(values, index=table.index), pd.DataFrame(timestamps, index=table.index)


def demand_feature_source_timestamps(table: pd.DataFrame, lags: tuple[int, ...], *,
                                     short_lag_ablation: bool = False) -> pd.DataFrame:
    """Return timestamp provenance for station-level demand-lag features."""
    timestamps = pd.to_datetime(table["ts_utc"], utc=True)
    if not short_lag_ablation:
        _, source = _daily_aligned_lags(table, lags)
        return source
    grouped = timestamps.groupby(table["station_id"], sort=False)
    return pd.DataFrame({f"lag_{lag}": grouped.shift(lag) for lag in lags}, index=table.index)


def lag_leakage_mask(table: pd.DataFrame, lags: tuple[int, ...], *,
                     short_lag_ablation: bool = False) -> pd.DataFrame:
    """Flag demand-lag sources at/after their target local-midnight origin.

    The returned table is intentionally public/testable: it makes the daily
    forecasting provenance check independent of XGBoost itself.
    """
    source = demand_feature_source_timestamps(table, lags, short_lag_ablation=short_lag_ablation)
    target_ts = pd.to_datetime(table["ts_utc"], utc=True)
    origins = target_ts.dt.tz_convert(cfg.LOCAL_TZ).dt.normalize()
    out = pd.DataFrame(index=table.index)
    for column in source:
        out[column] = source[column].notna() & (source[column] >= origins)
    return out


def xgboost_features(table: pd.DataFrame, *, short_lag_ablation: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Origin-safe demand lags plus calendar/holiday/event/station features.

    ``short_lag_ablation`` preserves the historical, leaky row-wise lag design
    strictly for reproducibility. It must never be used as the deployable
    default because target-hour rows after midnight can see same-day demand.
    """
    lags = SHORT_LAG_ABLATION_LAGS if short_lag_ablation else LEAK_FREE_LAGS
    out = table.copy()
    if short_lag_ablation:
        for lag in lags:
            out[f"lag_{lag}"] = out.groupby("station_id", sort=False).checkouts.shift(lag)
    else:
        values, _ = _daily_aligned_lags(out, lags)
        out[[f"lag_{lag}" for lag in lags]] = values
    hour, weekday, month = out.hour.to_numpy(float), out.weekday.to_numpy(float), out.month.to_numpy(float)
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    out["weekday_sin"], out["weekday_cos"] = np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7)
    out["month_sin"], out["month_cos"] = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
    # One shared model still needs station identity; unlike a per-station fit, it
    # shares every split/leaf structure across all 50 station series.
    out["station_code"] = pd.Categorical(out.station_id).codes.astype("int16")
    columns = [*[f"lag_{lag}" for lag in lags], "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
               "month_sin", "month_cos", "is_us_holiday", "event_flag", "station_code"]
    if not short_lag_ablation and lag_leakage_mask(out, lags).to_numpy().any():
        raise AssertionError("leak-free XGBoost features reference data at or after forecast origin")
    return out, columns


def deadline_callback(started: float, max_minutes: float):
    """Create the XGBoost-only deadline callback without burdening feature tests."""
    from xgboost.callback import TrainingCallback

    class DeadlineStop(TrainingCallback):
        def __init__(self):
            self.deadline = started + 60 * max_minutes
            self.stopped = False

        def after_iteration(self, model, epoch, evals_log):
            self.stopped = time.monotonic() >= self.deadline
            return self.stopped

    return DeadlineStop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-minutes", type=float, default=60)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--short-lag-ablation", action="store_true",
                        help="reproduce the historical non-deployable 1/2/3/24/168-hour lag ablation")
    add_final_argument(parser)
    args = parser.parse_args()
    import xgboost as xgb
    from .baselines import save_predictions, save_result

    split = "test" if args.final else "validation"
    tag = "test_2026" if args.final else "validation_2025"
    started = time.monotonic()
    loaded = load_table_through_test(final=True) if args.final else load_pretest_table()
    table, columns = xgboost_features(
        loaded.sort_values(["station_id", "ts_utc"]).reset_index(drop=True),
        short_lag_ablation=args.short_lag_ablation,
    )
    train = split_mask(table.ts_utc, "train") & table[columns].notna().all(axis=1)
    eval_mask = validation_window_mask(table, split, args.final)
    validation = eval_mask & table[columns].notna().all(axis=1)
    expected = int(eval_mask.sum()) if args.final else 435_600
    if validation.sum() != expected:
        raise RuntimeError(f"expected {expected:,} {split} observations, found {validation.sum():,}")
    deadline = deadline_callback(started, args.max_minutes)
    model = xgb.XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=list(cfg.QUANTILES), n_estimators=args.n_estimators,
        max_depth=8, learning_rate=.05, min_child_weight=20, subsample=.8, colsample_bytree=.8,
        tree_method="hist", device=args.device, n_jobs=1, random_state=cfg.SEED, callbacks=[deadline],
    )
    model.fit(table.loc[train, columns].astype("float32"), table.loc[train, "checkouts"], verbose=False)
    if args.final and deadline.stopped:
        raise RuntimeError("time budget reached: the refit differs from the validation fit; raise --max-minutes")
    prediction = model.predict(table.loc[validation, columns].astype("float32"))
    if prediction.shape != (int(validation.sum()), len(cfg.QUANTILES)):
        raise RuntimeError(f"unexpected XGBoost prediction shape {prediction.shape}")
    model_name = SHORT_LAG_ABLATION_MODEL if args.short_lag_ablation else LEAK_FREE_MODEL
    result = {"model": model_name, **score(table.loc[validation, "checkouts"], prediction)}
    result.update(validation_window_summary(table, split, args.final))
    result["excluded_warmup_rows"] = int(eval_mask.sum() - validation.sum())
    result["fit_minutes"] = (time.monotonic() - started) / 60
    result["time_budget_reached"] = deadline.stopped
    save_predictions(table, validation, prediction, result["model"], tag)
    if args.final:
        save_result(result, tag=tag, per_job=True)
    else:
        save_result(result)
    print(f"fit_minutes={result['fit_minutes']:.2f}; time_budget_reached={deadline.stopped}")


if __name__ == "__main__":
    main()
