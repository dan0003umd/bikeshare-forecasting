from src.event_day_analysis import (
    assert_event_partition,
    event_dates_2025,
    event_day_labels,
    label_prediction_rows,
    month_stratified_difference_in_differences,
    monthly_event_counts,
)
from src import config as cfg
import pandas as pd


def test_event_and_non_event_rows_add_to_fixed_validation_window():
    """The persisted evaluation window must partition into 435,600 event labels."""
    prediction = pd.read_parquet(cfg.PREDICTIONS_DIR / "xgboost_shared_quantile_validation_2025.parquet")
    rows = label_prediction_rows(prediction, event_day_labels())
    assert len(rows) == 435_600
    assert_event_partition(rows)
    assert int(rows.any_event.sum()) + int((~rows.any_event).sum()) == 435_600


def test_event_dates_are_unique_and_monthly_partition_matches_scored_dates():
    labels = event_day_labels()
    dates = event_dates_2025(labels)
    assert dates.local_date.is_unique
    assert dates.any_event.all()
    prediction = pd.read_parquet(cfg.PREDICTIONS_DIR / "xgboost_shared_quantile_validation_2025.parquet")
    scored = label_prediction_rows(prediction, labels)
    counts = monthly_event_counts(labels, set(scored.local_date.unique()))
    assert int(counts.event_days.sum()) + int(counts.non_event_days.sum()) == 363
    assert int(counts.event_days.sum()) * 50 * 24 == int(scored.any_event.sum())


def test_month_stratified_difference_in_differences_is_finite():
    labels = event_day_labels()
    xgb = label_prediction_rows(pd.read_parquet(cfg.PREDICTIONS_DIR / "xgboost_shared_quantile_validation_2025.parquet"), labels)
    daily = label_prediction_rows(pd.read_parquet(
        cfg.PREDICTIONS_DIR / "hf_transformer_daily_embed_noweather_validation_2025.parquet"), labels)
    counts = monthly_event_counts(labels, set(xgb.local_date.unique()))
    result = month_stratified_difference_in_differences(xgb, daily, counts.loc[counts.event_days.gt(0)],
                                                         "relative_pinball", draws=10)
    assert result["metric"] == "relative_pinball"
    assert result["ci95_low"] <= result["ci95_high"]
