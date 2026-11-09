"""Validation-only prediction sanity checks and a labelled non-negativity clip sensitivity."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .metrics import score
from .splits import assert_no_locked_test_rows

MODELS = (
    "seasonal_naive_rolling_8week", "seasonal_naive_frozen_train_8week", "per_station_negative_binomial",
    "hf_transformer_negative_binomial", "hf_transformer_daily_embed_noweather",
    "hf_transformer_daily_noembed_noweather", "hf_transformer_daily_embed_observed_weather_oracle",
    "xgboost_shared_quantile", "transformer_xgbfeatures_longcontext_nb",
    "transformer_xgbfeatures_longcontext_quantile",
)
QCOLS = [f"q{int(q * 100):02d}" for q in cfg.QUANTILES]


def main() -> None:
    out = []
    for model in MODELS:
        df = pd.read_parquet(cfg.PREDICTIONS_DIR / f"{model}_validation_2025.parquet")
        assert_no_locked_test_rows(df)
        if len(df) != 435_600:
            raise AssertionError(f"{model}: {len(df):,} rows")
        pred = df[QCOLS].to_numpy(float)
        actual = df.actual.to_numpy(float)
        raw = score(df.actual, pred)
        clipped = score(df.actual, np.maximum(pred, 0.0))
        out.append({
            "model": model,
            "zero_actual_share": float((actual == 0).mean()),
            "rows_with_negative_quantile": float((pred < 0).any(axis=1).mean()),
            "q05_negative_share": float((pred[:, 0] < 0).mean()),
            "rows_with_crossing_quantiles": float((np.diff(pred, axis=1) < 0).any(axis=1).mean()),
            "pinball_raw": raw["pinball_loss"], "pinball_clipped_at_0": clipped["pinball_loss"],
            "coverage90_raw": raw["coverage_90"], "coverage90_clipped": clipped["coverage_90"],
            "p_actual_le_q90": float((actual <= pred[:, 17]).mean()),
            "p_actual_le_q95": float((actual <= pred[:, 18]).mean()),
        })
    frame = pd.DataFrame(out)
    path = cfg.REPORTS_DIR / "prediction_sanity_2025.csv"
    frame.to_csv(path, index=False)
    print(frame.round(4).to_string(index=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()