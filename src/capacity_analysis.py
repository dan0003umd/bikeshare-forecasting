"""High-capacity station check: do model accuracy and rankings differ by station size?

Small stations can run out of bikes, so their observed checkouts undercount true demand
(right-censoring). Large stations rarely empty, so their counts sit closer to true demand.
We cannot observe empty stations directly (trip data only, no availability feed), so dock
capacity is the proxy. Stations are split at the median capacity, every main model is scored
within each group, and a paired day bootstrap tests whether gaps between models differ by group.

Run: python -m src.capacity_analysis
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as cfg
from .bootstrap_comparisons import _row_metric
from .metrics import score

MODELS = [
    "seasonal_naive_rolling_8week",
    "xgboost_shared_quantile",
    "hf_transformer_daily_embed_noweather",
    "hf_transformer_daily_embed_forecast_weather_lead1",
    "hf_transformer_daily_embed_observed_weather_oracle",
    "lstm_quantile",
]
# (label, model A, model B): is the A minus B gap different at high vs low capacity stations?
COMPARISONS = [
    ("transformer_minus_xgboost", "hf_transformer_daily_embed_noweather", "xgboost_shared_quantile"),
    ("lstm_minus_transformer", "lstm_quantile", "hf_transformer_daily_embed_noweather"),
    ("forecast_weather_minus_noweather", "hf_transformer_daily_embed_forecast_weather_lead1",
     "hf_transformer_daily_embed_noweather"),
]
QUANTILE_COLUMNS = [f"q{int(q * 100):02d}" for q in cfg.QUANTILES]
KEYS = ["station_id", "ts_utc"]


def capacity_groups() -> pd.DataFrame:
    """Label each station high (above median dock capacity) or low (at or below it)."""
    mapping = pd.read_csv(cfg.PROCESSED_DIR / "station_metadata_mapping.csv", dtype={"station_id": "string"})
    if mapping.capacity.isna().any():
        raise RuntimeError("some stations have no dock capacity")
    median = mapping.capacity.median()
    mapping["capacity_group"] = np.where(mapping.capacity > median, "high", "low")
    print(f"median capacity = {median:g} docks")
    print(mapping.groupby("capacity_group").capacity.agg(stations="size", min_docks="min", max_docks="max").to_string())
    return mapping[["station_id", "capacity", "capacity_group"]]


def load_predictions(model: str, groups: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_parquet(cfg.PREDICTIONS_DIR / f"{model}_validation_2025.parquet")
    frame["station_id"] = frame.station_id.astype("string")
    frame = frame.merge(groups, on="station_id", how="left", validate="many_to_one")
    if frame.capacity_group.isna().any():
        raise RuntimeError(f"{model}: prediction rows for stations with no capacity group")
    return frame


def group_scores(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every model scored separately within each capacity group, plus a volume-scaled pinball."""
    rows = []
    for model, frame in predictions.items():
        for group, part in frame.groupby("capacity_group"):
            result = score(part.actual, part[QUANTILE_COLUMNS])
            mean_actual = float(part.actual.mean())
            rows.append({"model": model, "capacity_group": group, "stations": part.station_id.nunique(),
                         **result, "mean_actual": mean_actual,
                         "pinball_per_bike": result["pinball_loss"] / mean_actual})
    out = pd.DataFrame(rows)
    out["rank_in_group"] = out.groupby("capacity_group").pinball_loss.rank().astype(int)
    return out.sort_values(["capacity_group", "rank_in_group"]).reset_index(drop=True)


def gap_by_group_bootstrap(first: pd.DataFrame, second: pd.DataFrame, draws: int, seed: int) -> list[dict]:
    """Bootstrap the A minus B pinball gap in each group and its high minus low difference.

    Resamples whole local dates (never single rows), the same unit as bootstrap_comparisons.
    """
    a = first[[*KEYS, "capacity_group", "actual"]].assign(loss_a=_row_metric(first, "pinball"))
    b = second[[*KEYS, "actual"]].assign(loss_b=_row_metric(second, "pinball"))
    rows = a.merge(b, on=KEYS, suffixes=("_a", "_b"), validate="one_to_one")
    if not np.array_equal(rows.actual_a.to_numpy(), rows.actual_b.to_numpy()):
        raise ValueError("paired prediction files do not share the same actual outcomes")
    rows["difference"] = rows.loss_a - rows.loss_b
    rows["local_date"] = pd.to_datetime(rows.ts_utc, utc=True).dt.tz_convert(cfg.LOCAL_TZ).dt.date
    daily = rows.pivot_table(index="local_date", columns="capacity_group", values="difference", aggfunc="mean").dropna()
    quantities = {"gap_low": daily["low"].to_numpy(), "gap_high": daily["high"].to_numpy()}
    quantities["gap_high_minus_low"] = quantities["gap_high"] - quantities["gap_low"]
    draws_index = np.random.default_rng(seed).integers(0, len(daily), size=(draws, len(daily)))
    out = []
    for name, values in quantities.items():
        low, high = np.quantile(values[draws_index].mean(axis=1), [.025, .975])
        out.append({"quantity": name, "estimate": float(values.mean()), "ci95_low": float(low),
                    "ci95_high": float(high), "ci_includes_zero": bool(low <= 0 <= high), "days": len(daily)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=10_000)
    args = parser.parse_args()
    groups = capacity_groups()
    predictions = {model: load_predictions(model, groups) for model in MODELS}

    scores = group_scores(predictions)
    scores_path = cfg.REPORTS_DIR / "capacity_group_scores_validation_2025.csv"
    scores.to_csv(scores_path, index=False)
    columns = ["capacity_group", "rank_in_group", "model", "pinball_loss", "pinball_per_bike", "coverage_90", "median_mae", "mean_actual"]
    print("\n" + scores[columns].to_string(index=False, float_format="%.4f"))

    rows = []
    for label, first, second in COMPARISONS:
        for result in gap_by_group_bootstrap(predictions[first], predictions[second], args.draws, cfg.SEED):
            rows.append({"comparison": label, "model_a": first, "model_b": second, **result})
    bootstrap = pd.DataFrame(rows)
    bootstrap_path = cfg.REPORTS_DIR / "capacity_gap_bootstrap_validation_2025.csv"
    bootstrap.to_csv(bootstrap_path, index=False)
    print("\n" + bootstrap[["comparison", "quantity", "estimate", "ci95_low", "ci95_high", "ci_includes_zero"]]
          .to_string(index=False, float_format="%.4f"))
    print(f"\nwrote {scores_path}\nwrote {bootstrap_path}")


if __name__ == "__main__":
    main()