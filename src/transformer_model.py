"""Data plumbing and small Hugging Face negative-binomial transformer config."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


PAST_LENGTH = 168 + 168  # context plus largest configured lag
PREDICTION_LENGTH = 24


def known_feature_columns(table: pd.DataFrame) -> list[str]:
    """Calendar plus *only* genuinely available forecast weather covariates."""
    columns = ["hour", "weekday", "month", "is_us_holiday", "event_flag"]
    columns.extend(c for c in ("temperature_2m_fc1d", "precipitation_fc1d", "wind_speed_10m_fc1d") if c in table)
    return columns


def time_features(table: pd.DataFrame, include_observed_weather: bool = False) -> np.ndarray:
    hour = table.hour.to_numpy(float)
    weekday = table.weekday.to_numpy(float)
    month = table.month.to_numpy(float)
    features = [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7),
                np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
                table.is_us_holiday.to_numpy(float), table.event_flag.to_numpy(float)]
    # Forecast weather is safe only when previous-runs coverage was complete.
    for column in ("temperature_2m_fc1d", "precipitation_fc1d", "wind_speed_10m_fc1d"):
        if column in table:
            values = table[column].to_numpy(float)
            features.append((values - values.mean()) / max(values.std(), 1e-6))
    if include_observed_weather:
        # ORACLE ONLY: observations are unavailable at a real 24-hour origin.
        # Scale using the locked training period to avoid validation distribution
        # information entering a covariate transform.
        from .splits import split_mask
        train = split_mask(table["ts_utc"], "train").to_numpy()
        for column in ("temperature_2m", "precipitation", "wind_speed_10m"):
            values = table[column].to_numpy(float)
            mean, std = values[train].mean(), values[train].std()
            features.append((values - mean) / max(std, 1e-6))
    return np.column_stack(features).astype("float32")


def transformer_config(num_stations: int, num_features: int, use_station_embedding: bool = True):
    from transformers import TimeSeriesTransformerConfig
    return TimeSeriesTransformerConfig(
        prediction_length=PREDICTION_LENGTH, context_length=168,
        lags_sequence=[1, 2, 3, 24, 168], input_size=1,
        distribution_output="negative_binomial", num_parallel_samples=100,
        num_time_features=num_features, num_dynamic_real_features=0,
        num_static_categorical_features=1 if use_station_embedding else 0,
        cardinality=[num_stations] if use_station_embedding else None,
        embedding_dimension=[16] if use_station_embedding else None,
        d_model=32, encoder_layers=2, decoder_layers=2,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=64, decoder_ffn_dim=64, dropout=.1,
    )


def daily_starts(table: pd.DataFrame, split: str, every_n_days: int = 1) -> np.ndarray:
    """UTC positions starting local midnight, requiring a complete non-DST 24h target."""
    from .splits import split_mask
    local_date = table.ts_utc.dt.tz_convert(cfg.LOCAL_TZ).dt.date
    eligible = split_mask(table.ts_utc, split)
    counts = table[eligible].assign(local_date=local_date[eligible]).groupby("local_date").size()
    # Input is a single station frame here, so complete target days have 24 rows.
    dates = counts.index[counts.eq(24)][::every_n_days]
    starts = table.index[eligible & local_date.isin(dates) & table.hour.eq(0)].to_numpy()
    return starts[starts >= PAST_LENGTH]
