"""Quantify stochastic transformer quantile variation across two saved seeds."""
from __future__ import annotations

import pandas as pd

from . import config as cfg
from .metrics import score


def main() -> None:
    primary = pd.read_parquet(cfg.PREDICTIONS_DIR / "hf_transformer_daily_embed_noweather_validation_2025.parquet")
    replicate = pd.read_parquet(cfg.PREDICTIONS_DIR / "hf_transformer_daily_embed_noweather_seed43_probe_validation_2025.parquet")
    keys = ["station_id", "ts_utc"]
    joined = primary.merge(replicate, on=keys, suffixes=("_seed42", "_seed43"), validate="one_to_one")
    columns = [f"q{int(q * 100):02d}" for q in cfg.QUANTILES]
    if not (joined.actual_seed42 == joined.actual_seed43).all():
        raise ValueError("sampling audit prediction files do not have identical actuals")
    scores_a = score(joined.actual_seed42, joined[[f"{c}_seed42" for c in columns]].set_axis(columns, axis=1))
    scores_b = score(joined.actual_seed43, joined[[f"{c}_seed43" for c in columns]].set_axis(columns, axis=1))
    difference = joined[[f"{c}_seed42" for c in columns]].to_numpy(float) - joined[[f"{c}_seed43" for c in columns]].to_numpy(float)
    report = pd.DataFrame([{
        "model": "hf_transformer_daily_embed_noweather", "seed_a": cfg.SEED, "seed_b": 43,
        "rows": len(joined), "mean_absolute_quantile_difference": abs(difference).mean(),
        "q50_mean_absolute_difference": abs(joined.q50_seed42 - joined.q50_seed43).mean(),
        "pinball_seed42": scores_a["pinball_loss"], "pinball_seed43": scores_b["pinball_loss"],
        "pinball_difference_seed43_minus_seed42": scores_b["pinball_loss"] - scores_a["pinball_loss"],
        "median_mae_seed42": scores_a["median_mae"], "median_mae_seed43": scores_b["median_mae"],
        "median_mae_difference_seed43_minus_seed42": scores_b["median_mae"] - scores_a["median_mae"],
    }])
    output = cfg.REPORTS_DIR / "transformer_sampling_audit.csv"
    report.to_csv(output, index=False)
    print(report.to_string(index=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
