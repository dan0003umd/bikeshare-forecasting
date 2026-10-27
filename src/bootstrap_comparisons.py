"""Paired day-bootstrap comparisons from persisted 2025 prediction artifacts."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as cfg


COMPARISONS = {
    "daily_embed_noweather_minus_rolling_seasonal": (
        "hf_transformer_daily_embed_noweather", "seasonal_naive_rolling_8week"
    ),
    "daily_embed_noweather_minus_daily_noembed_noweather": (
        "hf_transformer_daily_embed_noweather", "hf_transformer_daily_noembed_noweather"
    ),
    "daily_embed_noweather_minus_observed_weather_oracle": (
        "hf_transformer_daily_embed_noweather", "hf_transformer_daily_embed_observed_weather_oracle"
    ),
}


def _row_pinball(frame: pd.DataFrame) -> np.ndarray:
    quantiles = np.asarray(cfg.QUANTILES)
    predicted = frame[[f"q{int(q * 100):02d}" for q in quantiles]].to_numpy(float)
    error = frame.actual.to_numpy(float)[:, None] - predicted
    return np.maximum(quantiles * error, (quantiles - 1) * error).mean(axis=1)


def paired_day_bootstrap(first: pd.DataFrame, second: pd.DataFrame, draws: int = 10_000,
                         seed: int = cfg.SEED) -> dict[str, float | int | bool]:
    """Bootstrap A-B by resampling complete local dates, never individual rows."""
    keys = ["station_id", "ts_utc"]
    merged = first.merge(second, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    if not np.array_equal(merged.actual_a.to_numpy(), merged.actual_b.to_numpy()):
        raise ValueError("paired prediction files do not share the same actual outcomes")
    renamed_a = merged[["actual_a", *[f"q{int(q * 100):02d}_a" for q in cfg.QUANTILES]]].rename(
        columns=lambda c: c.removesuffix("_a") if c != "actual_a" else "actual")
    renamed_b = merged[["actual_b", *[f"q{int(q * 100):02d}_b" for q in cfg.QUANTILES]]].rename(
        columns=lambda c: c.removesuffix("_b") if c != "actual_b" else "actual")
    merged["local_date"] = pd.to_datetime(merged.ts_utc, utc=True).dt.tz_convert(cfg.LOCAL_TZ).dt.date
    day_difference = pd.DataFrame({
        "local_date": merged.local_date,
        "difference": _row_pinball(renamed_a) - _row_pinball(renamed_b),
    }).groupby("local_date", sort=True).difference.mean().to_numpy()
    rng = np.random.default_rng(seed)
    sampled_means = day_difference[rng.integers(0, len(day_difference), size=(draws, len(day_difference)))].mean(axis=1)
    low, high = np.quantile(sampled_means, [.025, .975])
    return {
        "difference_pinball_a_minus_b": float(day_difference.mean()),
        "ci95_low": float(low), "ci95_high": float(high), "days": int(len(day_difference)),
        "bootstrap_draws": int(draws), "ci_includes_zero": bool(low <= 0 <= high),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=10_000)
    args = parser.parse_args()
    rows = []
    for label, (first_name, second_name) in COMPARISONS.items():
        first = pd.read_parquet(cfg.PREDICTIONS_DIR / f"{first_name}_validation_2025.parquet")
        second = pd.read_parquet(cfg.PREDICTIONS_DIR / f"{second_name}_validation_2025.parquet")
        rows.append({"comparison": label, "model_a": first_name, "model_b": second_name,
                     **paired_day_bootstrap(first, second, draws=args.draws)})
    output = cfg.REPORTS_DIR / "paired_bootstrap_validation_2025.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
