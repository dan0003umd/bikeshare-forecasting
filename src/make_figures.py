"""Reproducible interim-report figures, created only from saved data/results."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg
from .splits import load_development_model_table


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
    path = cfg.PREDICTIONS_DIR / "seasonal_naive_frozen_train_8week_validation_2025.parquet"
    prediction = pd.read_parquet(path)
    prediction = prediction.merge(table[["station_id", "ts_utc", "precipitation"]], on=["station_id", "ts_utc"],
                                  how="left", validate="one_to_one")
    local_date = prediction.ts_utc.dt.tz_convert(cfg.LOCAL_TZ).dt.date
    train = table.ts_utc < pd.Timestamp("2025-01-01", tz=cfg.LOCAL_TZ).tz_convert("UTC")
    station = table.loc[train].groupby("station_id").checkouts.sum().idxmax()
    daily = prediction.loc[prediction.station_id.eq(station)].assign(local_date=local_date[prediction.station_id.eq(station)])
    daily["weekday"] = daily.ts_utc.dt.tz_convert(cfg.LOCAL_TZ).dt.day_of_week
    daily = daily.groupby("local_date").agg(weekday=("weekday", "first"), precipitation=("precipitation", "sum"))
    weekday_day = daily.index[daily.weekday.lt(5)][0]
    weekend_day = daily.index[daily.weekday.ge(5)][0]
    rainy_day = daily[ daily.precipitation.gt(0)].index[0]
    samples = [(weekday_day, "weekday"), (weekend_day, "weekend"), (rainy_day, "rainy")]
    _style()
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, (sample_day, label) in zip(axes, samples):
        rows = prediction.loc[prediction.station_id.eq(station) & local_date.eq(sample_day)].sort_values("ts_utc")
        hours = rows.ts_utc.dt.tz_convert(cfg.LOCAL_TZ).dt.hour
        ax.fill_between(hours, rows.q05, rows.q95, color="#72b7b2", alpha=.35, label="5--95%" if ax is axes[0] else None)
        ax.plot(hours, rows.q50, color="#4c78a8", label="median" if ax is axes[0] else None)
        ax.scatter(hours, rows.actual, s=12, color="#e45756", label="actual" if ax is axes[0] else None)
        ax.set_ylabel(f"{station}\ncheck-outs")
        ax.set_title(f"{label.title()} day: {sample_day}", fontsize=10)
    axes[0].legend(loc="upper right")
    axes[-1].set(xlabel="Local hour")
    fig.suptitle("Frozen-training seasonal-naive 24-hour validation forecasts", y=1.01)
    plt.tight_layout()
    plt.savefig(cfg.REPORTS_DIR / "seasonal_naive_sample_forecasts.png", dpi=160)
    plt.close()


def error_by_hour_from_saved_predictions() -> None:
    """Use persisted per-row predictions for every evaluated model."""
    paths = [path for path in sorted(cfg.PREDICTIONS_DIR.glob("*_validation_2025.parquet"))
             if "_seed43_probe" not in path.name]
    if not paths:
        raise FileNotFoundError("No saved predictions. Run the evaluators before making figures.")
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["model"] = path.name.removesuffix("_validation_2025.parquet")
        frame["hour"] = frame.ts_utc.dt.tz_convert(cfg.LOCAL_TZ).dt.hour
        frame["absolute_error"] = np.abs(frame.actual - frame.q50)
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    _style()
    summary = frame.groupby(["hour", "model"]).absolute_error.mean().unstack("model")
    ax = summary.plot(marker="o", figsize=(9, 4))
    ax.set(title="Validation mean absolute error by local hour", xlabel="Local hour", ylabel="MAE")
    plt.tight_layout()
    plt.savefig(cfg.REPORTS_DIR / "error_by_hour_available_predictions.png", dpi=160)
    plt.close()


def loss_curve_if_present() -> None:
    for stem, title, ylabel in [
        ("transformer", "Initial weekly-stride transformer loss", "Negative log likelihood"),
        ("transformer_daily_embed_noweather", "Daily-stride transformer loss (embeddings, no weather)", "Negative log likelihood"),
        ("transformer_daily_embed_observed_weather_oracle", "Daily-stride transformer loss (observed-weather ORACLE)", "Negative log likelihood"),
        ("transformer_xgbfeatures_longcontext_nb", "Longer-context transformer loss (NB head)", "Negative log likelihood"),
        ("transformer_xgbfeatures_longcontext_quantile", "Longer-context transformer loss (direct quantile head)", "Pinball loss"),
    ]:
        path = cfg.REPORTS_DIR / f"{stem}_loss.csv"
        if not path.exists():
            continue
        log = pd.read_csv(path)
        _style()
        ax = log.plot(x="epoch", y=["train_loss", "validation_loss"], marker="o", figsize=(7, 4))
        ax.set(title=title, ylabel=ylabel)
        plt.tight_layout()
        plt.savefig(cfg.REPORTS_DIR / f"{stem}_loss.png", dpi=160)
        plt.close()


def main() -> None:
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table = load_development_model_table()
    results = pd.read_csv(cfg.REPORTS_DIR / "results_validation_2025.csv")
    eda_hour_of_week(table)
    calibration_chart(results)
    seasonal_sample_forecasts(table)
    error_by_hour_from_saved_predictions()
    loss_curve_if_present()
    print(f"figures written to {cfg.REPORTS_DIR}")


if __name__ == "__main__":
    main()
