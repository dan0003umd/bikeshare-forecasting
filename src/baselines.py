"""Probabilistic 2025 baselines evaluated on the common fixed windows."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import nbinom
import statsmodels.api as sm

from . import config as cfg
from .forecast_windows import validation_window_mask, validation_window_summary
from .metrics import score
from .splits import split_mask


def seasonal_naive_quantiles(table: pd.DataFrame, history_weeks: int = 8) -> np.ndarray:
    """Same-hour-of-week empirical quantiles fit only on the 2023--24 split."""
    if not {"hour", "weekday", "checkouts"} <= set(table):
        raise ValueError("model table needs hour, weekday, and checkouts")
    data = table.sort_values(["station_id", "ts_utc"]).copy()
    data["hour_of_week"] = data["weekday"].astype(int) * 24 + data["hour"].astype(int)
    grid = np.asarray(cfg.QUANTILES)
    prediction = np.full((len(data), len(grid)), np.nan, dtype="float32")
    train = split_mask(data["ts_utc"], "train").to_numpy()
    # Freeze each station/hour-of-week distribution at the training cutoff.  In
    # particular, no early-2025 demand can improve later 2025 validation scores.
    grouped_indices = data.groupby(["station_id", "hour_of_week"], sort=False).indices
    for idx in grouped_indices.values():
        train_idx = np.asarray(idx)[train[np.asarray(idx)]]
        if len(train_idx):
            history = data["checkouts"].to_numpy(dtype=float)[train_idx[-history_weeks:]]
            prediction[np.asarray(idx), :] = np.quantile(history, grid)
    data[[f"q{int(q * 100):02d}" for q in grid]] = prediction
    return data.sort_index()[[f"q{int(q * 100):02d}" for q in grid]].to_numpy()


def evaluate_seasonal_naive(table: pd.DataFrame) -> dict:
    pred = seasonal_naive_quantiles(table)
    mask = validation_window_mask(table)
    valid = mask & np.isfinite(pred).all(axis=1)
    result = {"model": "seasonal_naive_8week", **score(table.loc[valid, "checkouts"], pred[valid])}
    result.update(validation_window_summary(table))
    result["excluded_warmup_rows"] = int(mask.sum() - valid.sum())
    return result


def _time_design(frame: pd.DataFrame) -> np.ndarray:
    """Compact cyclic calendar design shared by the count baselines."""
    hour = frame["hour"].to_numpy(float)
    weekday = frame["weekday"].to_numpy(float)
    month = frame["month"].to_numpy(float)
    return np.column_stack([
        np.ones(len(frame)), np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7),
        np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
    ])


def evaluate_negative_binomial(table: pd.DataFrame) -> dict:
    """Independent statsmodels NB2 GLMs, fit only to 2023--24 demand."""
    train_mask = split_mask(table["ts_utc"], "train")
    window_mask = validation_window_mask(table)
    pred = np.full((len(table), len(cfg.QUANTILES)), np.nan, dtype="float32")
    for station, idx in table.groupby("station_id", sort=False).indices.items():
        idx = np.asarray(idx)
        train_idx = idx[train_mask.iloc[idx].to_numpy()]
        valid_idx = idx[window_mask.iloc[idx].to_numpy()]
        y = table["checkouts"].iloc[train_idx].to_numpy(float)
        mean, variance = y.mean(), y.var(ddof=1)
        alpha = max((variance - mean) / max(mean * mean, 1e-9), 1e-6)
        model = sm.GLM(y, _time_design(table.iloc[train_idx]), family=sm.families.NegativeBinomial(alpha=alpha))
        fit = model.fit(maxiter=100, disp=0)
        mu = np.maximum(fit.predict(_time_design(table.iloc[valid_idx])), 1e-8)
        size = 1 / alpha
        probability = size / (size + mu)
        pred[valid_idx] = nbinom.ppf(np.asarray(cfg.QUANTILES)[None, :], size, probability[:, None])
    valid = window_mask & np.isfinite(pred).all(axis=1)
    result = {"model": "per_station_negative_binomial", **score(table.loc[valid, "checkouts"], pred[valid])}
    result.update(validation_window_summary(table))
    result["excluded_warmup_rows"] = int(window_mask.sum() - valid.sum())
    return result


def save_result(result: dict) -> pd.DataFrame:
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.REPORTS_DIR / "results_validation_2025.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    old = old[old.get("model", pd.Series(dtype=str)) != result["model"]] if len(old) else old
    out = pd.concat([old, pd.DataFrame([result])], ignore_index=True)
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["seasonal_naive", "negative_binomial"], default="seasonal_naive")
    args = parser.parse_args()
    table = pd.read_parquet(cfg.PROCESSED_DIR / "model_table.parquet")
    if args.model == "seasonal_naive":
        save_result(evaluate_seasonal_naive(table))
    else:
        save_result(evaluate_negative_binomial(table))


if __name__ == "__main__":
    main()
