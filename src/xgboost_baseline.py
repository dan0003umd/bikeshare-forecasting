"""One shared 19-quantile XGBoost baseline, fitted only through 2024."""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from xgboost.callback import TrainingCallback

from . import config as cfg
from .baselines import save_predictions, save_result
from .forecast_windows import validation_window_mask, validation_window_summary
from .metrics import score
from .splits import add_final_argument, load_development_model_table, load_table_through_test, split_mask


LAGS = (1, 2, 3, 24, 168)


def load_pretest_table() -> pd.DataFrame:
    """Use the shared guarded loader, then establish deterministic station order."""
    return load_development_model_table().sort_values(["station_id", "ts_utc"]).reset_index(drop=True)


def xgboost_features(table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Five demand lags plus exactly the transformer's calendar covariates and station ID."""
    out = table.copy()
    for lag in LAGS:
        out[f"lag_{lag}"] = out.groupby("station_id", sort=False).checkouts.shift(lag)
    hour, weekday, month = out.hour.to_numpy(float), out.weekday.to_numpy(float), out.month.to_numpy(float)
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    out["weekday_sin"], out["weekday_cos"] = np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7)
    out["month_sin"], out["month_cos"] = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
    # One shared model still needs station identity; unlike a per-station fit, it
    # shares every split/leaf structure across all 50 station series.
    out["station_code"] = pd.Categorical(out.station_id).codes.astype("int16")
    columns = [*[f"lag_{lag}" for lag in LAGS], "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
               "month_sin", "month_cos", "is_us_holiday", "event_flag", "station_code"]
    return out, columns


class DeadlineStop(TrainingCallback):
    """Stop boosting at the user-specified wall-clock budget."""
    def __init__(self, started: float, max_minutes: float):
        self.deadline = started + 60 * max_minutes
        self.stopped = False

    def after_iteration(self, model, epoch, evals_log):
        self.stopped = time.monotonic() >= self.deadline
        return self.stopped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-minutes", type=float, default=60)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    add_final_argument(parser)
    args = parser.parse_args()
    import xgboost as xgb

    split = "test" if args.final else "validation"
    tag = "test_2026" if args.final else "validation_2025"
    started = time.monotonic()
    loaded = load_table_through_test(final=True) if args.final else load_pretest_table()
    table, columns = xgboost_features(loaded.sort_values(["station_id", "ts_utc"]).reset_index(drop=True))
    train = split_mask(table.ts_utc, "train") & table[columns].notna().all(axis=1)
    eval_mask = validation_window_mask(table, split, args.final)
    validation = eval_mask & table[columns].notna().all(axis=1)
    expected = int(eval_mask.sum()) if args.final else 435_600
    if validation.sum() != expected:
        raise RuntimeError(f"expected {expected:,} {split} observations, found {validation.sum():,}")
    deadline = DeadlineStop(started, args.max_minutes)
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
    result = {"model": "xgboost_shared_quantile", **score(table.loc[validation, "checkouts"], prediction)}
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
