"""Reproducible interim-report figures, created only from saved data/results."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg
from .baselines import seasonal_naive_quantiles
from .forecast_windows import validation_window_mask


def _style():
    plt.style.use("seaborn-v0_8-whitegrid")


def eda_hour_of_week(table: pd.DataFrame) -> None:
    _style()
    train = table[table.ts_utc < pd.Timestamp("2025-01-01", tz="America/New_York").tz_convert("UTC")]
    mean = train.groupby(["weekday", "hour"]).checkouts.mean().unstack(0)
    ax = mean.plot(figsize=(10, 4), linewidth=1.4)
    ax.set(xlabel="Local hour", ylabel="Mean check-outs per station-hour", title="Training demand by local hour of week")
    ax.legend(title="Weekday (Mon=0)", ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(cfg.REPORTS_DIR / "eda_hour_of_week.png", dpi=160)
    plt.close()


def calibration_chart(results: pd.DataFrame) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(results.model, results.coverage_90, color="#4c78a8")
    ax.axhline(.90, color="#e45756", linestyle="--", label="nominal 90%")
    ax.set(ylim=(0, 1), ylabel="Empirical coverage", title="Validation 90% interval coverage")
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    plt.tight_layout()
    plt.savefig(cfg.REPORTS_DIR / "calibration_coverage.png", dpi=160)
    plt.close()


def seasonal_sample_forecasts(table: pd.DataFrame) -> None:
    """Plot weekday, weekend, and rainy 24-hour samples for the busy station."""
    prediction = seasonal_naive_quantiles(table)
    mask = validation_window_mask(table)
    local_date = table.ts_utc.dt.tz_convert(cfg.LOCAL_TZ).dt.date
    train = table.ts_utc < pd.Timestamp("2025-01-01", tz=cfg.LOCAL_TZ).tz_convert("UTC")
    station = table.loc[train].groupby("station_id").checkouts.sum().idxmax()
    daily = table.loc[mask & table.station_id.eq(station)].assign(local_date=local_date[mask & table.station_id.eq(station)])
    daily = daily.groupby("local_date").agg(weekday=("weekday", "first"), precipitation=("precipitation", "sum"))
    weekday_day = daily.index[daily.weekday.lt(5)][0]
    weekend_day = daily.index[daily.weekday.ge(5)][0]
    rainy_day = daily[ daily.precipitation.gt(0)].index[0]
    samples = [(weekday_day, "weekday"), (weekend_day, "weekend"), (rainy_day, "rainy")]
    _style()
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, (sample_day, label) in zip(axes, samples):
        take = mask & table.station_id.eq(station) & local_date.eq(sample_day)
        rows = table.loc[take].sort_values("ts_utc")
        pos = rows.index.to_numpy()
        qs = prediction[pos]
        hours = rows.ts_utc.dt.tz_convert(cfg.LOCAL_TZ).dt.hour
        ax.fill_between(hours, qs[:, 0], qs[:, -1], color="#72b7b2", alpha=.35, label="5--95%" if ax is axes[0] else None)
        ax.plot(hours, qs[:, 9], color="#4c78a8", label="median" if ax is axes[0] else None)
        ax.scatter(hours, rows.checkouts, s=12, color="#e45756", label="actual" if ax is axes[0] else None)
        ax.set_ylabel(f"{station}\ncheck-outs")
        ax.set_title(f"{label.title()} day: {sample_day}", fontsize=10)
    axes[0].legend(loc="upper right")
    axes[-1].set(xlabel="Local hour")
    fig.suptitle("Frozen-training seasonal-naive 24-hour validation forecasts", y=1.01)
    plt.tight_layout()
    plt.savefig(cfg.REPORTS_DIR / "seasonal_naive_sample_forecasts.png", dpi=160)
    plt.close()


def error_by_hour_available_predictions(table: pd.DataFrame) -> None:
    """Plot retained per-row error evidence; transformer trajectories were not saved."""
    prediction = seasonal_naive_quantiles(table)
    mask = validation_window_mask(table)
    frame = table.loc[mask, ["hour", "checkouts"]].copy()
    frame["seasonal_naive_8week"] = np.abs(frame.checkouts.to_numpy() - prediction[mask.to_numpy(), 9])
    _style()
    ax = frame.groupby("hour")["seasonal_naive_8week"].mean().plot(marker="o", figsize=(8, 4))
    ax.set(title="Validation median absolute error by local hour (retained predictions)",
           xlabel="Local hour", ylabel="MAE")
    plt.tight_layout()
    plt.savefig(cfg.REPORTS_DIR / "error_by_hour_available_predictions.png", dpi=160)
    plt.close()


def loss_curve_if_present() -> None:
    path = cfg.REPORTS_DIR / "transformer_loss.csv"
    if not path.exists():
        return
    log = pd.read_csv(path)
    _style()
    ax = log.plot(x="epoch", y=["train_loss", "validation_loss"], marker="o", figsize=(7, 4))
    ax.set(title="Transformer training and validation loss", ylabel="Negative log likelihood")
    plt.tight_layout()
    plt.savefig(cfg.REPORTS_DIR / "transformer_loss.png", dpi=160)
    plt.close()


def main() -> None:
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(cfg.PROCESSED_DIR / "model_table.parquet")
    results = pd.read_csv(cfg.REPORTS_DIR / "results_validation_2025.csv")
    eda_hour_of_week(table)
    calibration_chart(results)
    seasonal_sample_forecasts(table)
    error_by_hour_available_predictions(table)
    loss_curve_if_present()
    print(f"figures written to {cfg.REPORTS_DIR}")


if __name__ == "__main__":
    main()
