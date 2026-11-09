"""Event-day validation slices from saved 2025 prediction artifacts.

This diagnostic deliberately uses ``load_development_model_table`` for event
labels, so it cannot read a local-2026 row.  It does not retrain or overwrite
any model result.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .bootstrap_comparisons import _row_metric
from .metrics import interval_coverage_90, median_mae, pinball_loss
from .splits import assert_no_locked_test_rows, load_development_model_table, split_mask


MODEL_NAMES = (
    "seasonal_naive_rolling_8week",
    "seasonal_naive_frozen_train_8week",
    "per_station_negative_binomial",
    "hf_transformer_negative_binomial",
    "hf_transformer_daily_embed_noweather",
    "hf_transformer_daily_noembed_noweather",
    "hf_transformer_daily_embed_observed_weather_oracle",
    "xgboost_shared_quantile",
    "transformer_xgbfeatures_longcontext_nb",
    "transformer_xgbfeatures_longcontext_quantile",
)
QUANTILE_COLUMNS = [f"q{int(q * 100):02d}" for q in cfg.QUANTILES]
EVENT_CATEGORIES = ("nationals_home_game", "cherry_blossom_festival", "us_holiday", "marathon")


def _local_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.tz_convert(cfg.LOCAL_TZ).dt.date


def _event_dates() -> dict[str, set[object]]:
    """Read verified category dates; absence of a category means no such flag exists."""
    events_path = cfg.EXTERNAL_DIR / "events.csv"
    if not events_path.exists():
        return {name: set() for name in EVENT_CATEGORIES}
    events = pd.read_csv(events_path, dtype={"date": "string", "event_type": "string"})
    events["date"] = pd.to_datetime(events["date"], errors="raise").dt.date
    return {
        category: set(events.loc[events.event_type.eq(category), "date"])
        for category in EVENT_CATEGORIES if category != "us_holiday"
    } | {"us_holiday": set()}


def event_day_labels() -> pd.DataFrame:
    """Return one guarded, local-date label row for every local 2025 calendar day."""
    columns = ["ts_utc", "is_us_holiday", "event_flag"]
    table = load_development_model_table(columns=columns)
    assert_no_locked_test_rows(table)
    validation = table.loc[split_mask(table.ts_utc, "validation")].copy()
    validation["local_date"] = _local_date(validation.ts_utc)
    daily = validation.groupby("local_date", sort=True).agg(
        us_holiday=("is_us_holiday", "max"), event_flag=("event_flag", "max"),
    ).reset_index()
    event_dates = _event_dates()
    for category in EVENT_CATEGORIES:
        if category == "us_holiday":
            daily[category] = daily.us_holiday.astype(bool)
        else:
            daily[category] = daily.local_date.isin(event_dates[category])
    daily["any_event"] = daily[list(EVENT_CATEGORIES)].any(axis=1)
    # The only sources of the model-table event flag are the typed event CSV.
    typed_event = daily[["nationals_home_game", "cherry_blossom_festival", "marathon"]].any(axis=1)
    if not np.array_equal(daily.event_flag.astype(bool), typed_event.to_numpy(bool)):
        raise AssertionError("typed event dates do not reproduce model-table event_flag")
    # ``us_holiday`` has intentionally been replaced by the boolean category
    # of the same name above; retain it alongside the other slice labels.
    return daily.drop(columns=["event_flag"])


def label_prediction_rows(prediction: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Attach the aggregate event partition to one prediction artifact."""
    assert_no_locked_test_rows(prediction)
    out = prediction.copy()
    out["local_date"] = _local_date(out.ts_utc)
    out = out.merge(labels[["local_date", "any_event"]], on="local_date", how="left", validate="many_to_one")
    if out.any_event.isna().any():
        raise AssertionError("prediction artifact has a date outside the guarded 2025 labels")
    out["any_event"] = out.any_event.astype(bool)
    return out


def assert_event_partition(rows: pd.DataFrame, expected_total: int = 435_600) -> None:
    """Ensure the event/non-event slices form exactly the fixed evaluation window."""
    event_rows = int(rows.any_event.sum())
    non_event_rows = int((~rows.any_event).sum())
    if event_rows + non_event_rows != expected_total:
        raise AssertionError(f"event ({event_rows}) + non-event ({non_event_rows}) != {expected_total}")


def _score_slice(rows: pd.DataFrame) -> dict[str, float | int]:
    pred = rows[QUANTILE_COLUMNS].to_numpy(float)
    mean_actual = float(rows.actual.mean())
    pinball = pinball_loss(rows.actual, pred)
    return {
        "pinball_loss": pinball,
        "mean_actual_demand": mean_actual,
        "relative_pinball": pinball / mean_actual if mean_actual else np.nan,
        "median_mae": median_mae(rows.actual, pred),
        "coverage_90": interval_coverage_90(rows.actual, pred),
        "n_observations": len(rows),
        "days": rows.local_date.nunique(),
    }


def event_dates_2025(labels: pd.DataFrame) -> pd.DataFrame:
    """List each aggregate event date once, with its possibly overlapping flags."""
    out = labels.loc[labels.any_event, ["local_date", *EVENT_CATEGORIES, "any_event"]].copy()
    out.insert(1, "weekday", pd.to_datetime(out.local_date).dt.day_name())
    if out.local_date.duplicated().any() or not out.any_event.all():
        raise AssertionError("aggregate event dates must be unique and flagged")
    return out.sort_values("local_date").reset_index(drop=True)


def monthly_event_counts(labels: pd.DataFrame, valid_dates: set[object]) -> pd.DataFrame:
    """Evaluation-window event/non-event counts, retaining the calendar denominator."""
    daily = labels.copy()
    daily["month"] = pd.to_datetime(daily.local_date).dt.strftime("%Y-%m")
    daily["is_valid_evaluation_day"] = daily.local_date.isin(valid_dates)
    rows = []
    for month, frame in daily.groupby("month", sort=True):
        valid = frame.loc[frame.is_valid_evaluation_day]
        rows.append({
            "month": month,
            "calendar_days": int(len(frame)),
            "valid_evaluation_days": int(len(valid)),
            "event_days": int(valid.any_event.sum()),
            "non_event_days": int((~valid.any_event).sum()),
            **{category: int(valid[category].sum()) for category in EVENT_CATEGORIES},
        })
    output = pd.DataFrame(rows)
    if int(output.event_days.sum() + output.non_event_days.sum()) != len(valid_dates):
        raise AssertionError("monthly event partition does not equal the valid-date count")
    return output


def month_matched_scores(rows: pd.DataFrame, counts: pd.DataFrame) -> dict[str, float | int]:
    """Compare within calendar months, weighting every month by its event-day count."""
    work = rows.copy()
    work["month"] = pd.to_datetime(work.local_date).dt.strftime("%Y-%m")
    weighted: dict[str, dict[str, float]] = {"event": {}, "non_event": {}}
    total_weight = float(counts.event_days.sum())
    metric_names = ("pinball_loss", "relative_pinball", "median_mae", "coverage_90", "mean_actual_demand")
    for slice_name, is_event in (("event", True), ("non_event", False)):
        values = {metric: 0.0 for metric in metric_names}
        for count in counts.itertuples(index=False):
            subset = work.loc[work.month.eq(count.month) & work.any_event.eq(is_event)]
            if subset.empty:
                raise AssertionError(f"{slice_name} has no rows for event-containing month {count.month}")
            score = _score_slice(subset)
            for metric in metric_names:
                values[metric] += count.event_days * float(score[metric])
        weighted[slice_name] = {metric: value / total_weight for metric, value in values.items()}
    result: dict[str, float | int] = {
        "event_months": int(len(counts)), "event_days_weight": int(total_weight),
        "event_days": int(counts.event_days.sum()), "matched_non_event_days": int(counts.non_event_days.sum()),
    }
    for metric in metric_names:
        result[f"event_{metric}"] = weighted["event"][metric]
        result[f"non_event_{metric}"] = weighted["non_event"][metric]
        result[f"event_minus_non_event_{metric}"] = weighted["event"][metric] - weighted["non_event"][metric]
    return result


def paired_slice_bootstrap(first: pd.DataFrame, second: pd.DataFrame, event: bool, metric: str,
                           draws: int = 10_000, seed: int = cfg.SEED) -> dict[str, float | int | bool]:
    """Paired day bootstrap for XGBoost minus the embedded daily transformer."""
    keys = ["station_id", "ts_utc"]
    second_cols = [*keys, "actual", *QUANTILE_COLUMNS]
    merged = first.merge(second[second_cols], on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    if not np.array_equal(merged.actual_a.to_numpy(), merged.actual_b.to_numpy()):
        raise ValueError("paired artifacts do not share actual outcomes")
    subset = merged.loc[merged.any_event.eq(event)].copy()
    if subset.empty:
        raise ValueError("cannot bootstrap an empty event slice")
    a = subset[["actual_a", *[f"{column}_a" for column in QUANTILE_COLUMNS]]].rename(
        columns={"actual_a": "actual", **{f"{column}_a": column for column in QUANTILE_COLUMNS}})
    b = subset[["actual_b", *[f"{column}_b" for column in QUANTILE_COLUMNS]]].rename(
        columns={"actual_b": "actual", **{f"{column}_b": column for column in QUANTILE_COLUMNS}})
    difference = _row_metric(a, metric) - _row_metric(b, metric)
    daily = pd.DataFrame({"local_date": subset.local_date, "difference": difference}).groupby(
        "local_date", sort=True).difference.mean().to_numpy()
    rng = np.random.default_rng(seed)
    sampled = daily[rng.integers(0, len(daily), size=(draws, len(daily)))].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {
        "metric": metric, "slice": "event" if event else "non_event",
        "difference_xgboost_minus_daily_embed": float(daily.mean()),
        "ci95_low": float(low), "ci95_high": float(high), "ci_includes_zero": bool(low <= 0 <= high),
        "days": int(len(daily)), "n_observations": int(len(subset)), "bootstrap_draws": draws, "seed": seed,
    }


def paired_month_matched_bootstrap(first: pd.DataFrame, second: pd.DataFrame, counts: pd.DataFrame,
                                   event: bool, metric: str, draws: int = 10_000,
                                   seed: int = cfg.SEED) -> dict[str, float | int | bool]:
    """Stratified day bootstrap with event-day weights for the calendar-month match."""
    keys = ["station_id", "ts_utc"]
    second_cols = [*keys, "actual", *QUANTILE_COLUMNS]
    merged = first.merge(second[second_cols], on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    if not np.array_equal(merged.actual_a.to_numpy(), merged.actual_b.to_numpy()):
        raise ValueError("paired artifacts do not share actual outcomes")
    merged["month"] = pd.to_datetime(merged.local_date).dt.strftime("%Y-%m")
    a = merged[["actual_a", *[f"{column}_a" for column in QUANTILE_COLUMNS]]].rename(
        columns={"actual_a": "actual", **{f"{column}_a": column for column in QUANTILE_COLUMNS}})
    b = merged[["actual_b", *[f"{column}_b" for column in QUANTILE_COLUMNS]]].rename(
        columns={"actual_b": "actual", **{f"{column}_b": column for column in QUANTILE_COLUMNS}})
    merged["difference"] = _row_metric(a, metric) - _row_metric(b, metric)
    rng = np.random.default_rng(seed)
    sampled = np.zeros(draws, dtype=float)
    point = 0.0
    total_weight = float(counts.event_days.sum())
    n_days = 0
    n_rows = 0
    for count in counts.itertuples(index=False):
        daily = merged.loc[merged.month.eq(count.month) & merged.any_event.eq(event)].groupby(
            "local_date", sort=True).difference.mean().to_numpy()
        if not len(daily):
            raise AssertionError(f"month {count.month} has no rows for requested slice")
        weight = count.event_days / total_weight
        point += weight * daily.mean()
        sampled += weight * daily[rng.integers(0, len(daily), size=(draws, len(daily)))].mean(axis=1)
        n_days += len(daily)
        n_rows += len(daily) * 50 * 24
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {
        "metric": metric, "slice": "event" if event else "non_event",
        "difference_xgboost_minus_daily_embed": float(point),
        "ci95_low": float(low), "ci95_high": float(high), "ci_includes_zero": bool(low <= 0 <= high),
        "event_months": int(len(counts)), "event_days_weight": int(total_weight),
        "days": int(n_days), "n_observations": int(n_rows), "bootstrap_draws": draws, "seed": seed,
    }


def month_stratified_difference_in_differences(first: pd.DataFrame, second: pd.DataFrame, counts: pd.DataFrame,
                                               metric: str, draws: int = 10_000,
                                               seed: int = cfg.SEED) -> dict[str, float | int | bool]:
    """Event-minus-month-matched-non-event XGBoost-minus-transformer difference.

    For ``relative_pinball``, each bootstrap sample divides the model pinball
    difference by its own sampled mean actual demand before forming the
    event-minus-non-event contrast.
    """
    if metric not in {"pinball", "relative_pinball"}:
        raise ValueError(f"unsupported difference-in-differences metric: {metric}")
    keys = ["station_id", "ts_utc"]
    second_cols = [*keys, "actual", *QUANTILE_COLUMNS]
    merged = first.merge(second[second_cols], on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    if not np.array_equal(merged.actual_a.to_numpy(), merged.actual_b.to_numpy()):
        raise ValueError("paired artifacts do not share actual outcomes")
    merged["month"] = pd.to_datetime(merged.local_date).dt.strftime("%Y-%m")
    a = merged[["actual_a", *[f"{column}_a" for column in QUANTILE_COLUMNS]]].rename(
        columns={"actual_a": "actual", **{f"{column}_a": column for column in QUANTILE_COLUMNS}})
    b = merged[["actual_b", *[f"{column}_b" for column in QUANTILE_COLUMNS]]].rename(
        columns={"actual_b": "actual", **{f"{column}_b": column for column in QUANTILE_COLUMNS}})
    merged["difference"] = _row_metric(a, "pinball") - _row_metric(b, "pinball")
    rng = np.random.default_rng(seed)
    total_weight = float(counts.event_days.sum())
    sampled = np.zeros(draws, dtype=float)
    point = 0.0
    for count in counts.itertuples(index=False):
        weight = count.event_days / total_weight
        sampled_slices: dict[bool, np.ndarray] = {}
        point_slices: dict[bool, float] = {}
        for is_event in (True, False):
            daily = merged.loc[merged.month.eq(count.month) & merged.any_event.eq(is_event)].groupby(
                "local_date", sort=True).agg(difference=("difference", "mean"), actual=("actual_a", "mean"))
            if daily.empty:
                raise AssertionError(f"month {count.month} has no rows for requested slice")
            point_slices[is_event] = (daily.difference.mean() if metric == "pinball" else
                                      daily.difference.mean() / daily.actual.mean())
            indices = rng.integers(0, len(daily), size=(draws, len(daily)))
            sampled_difference = daily.difference.to_numpy()[indices].mean(axis=1)
            if metric == "relative_pinball":
                sampled_difference /= daily.actual.to_numpy()[indices].mean(axis=1)
            sampled_slices[is_event] = sampled_difference
        point += weight * (point_slices[True] - point_slices[False])
        sampled += weight * (sampled_slices[True] - sampled_slices[False])
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {
        "metric": metric, "contrast": "(xgboost_minus_daily_embed)_event_minus_month_matched_non_event",
        "difference_in_differences": float(point), "ci95_low": float(low), "ci95_high": float(high),
        "ci_excludes_zero": bool(high < 0 or low > 0), "event_months": int(len(counts)),
        "event_days_weight": int(total_weight), "bootstrap_draws": draws, "seed": seed,
    }


def _render_markdown(category_counts: pd.DataFrame, monthly_counts: pd.DataFrame, scores: pd.DataFrame,
                     month_matched: pd.DataFrame, bootstrap: pd.DataFrame, month_bootstrap: pd.DataFrame,
                     difference_in_differences: pd.DataFrame, dates: pd.DataFrame, elapsed_minutes: float,
                     command: str) -> str:
    def table(frame: pd.DataFrame) -> str:
        columns = list(frame.columns)
        header = "| " + " | ".join(columns) + " |"
        rule = "|" + "|".join(["---"] * len(columns)) + "|"
        body = []
        for row in frame.itertuples(index=False, name=None):
            body.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join([header, rule, *body])
    event_days = int(category_counts.loc[category_counts.category.eq("any_event"), "valid_evaluation_days"].iloc[0])
    overlap_days = int(dates[list(EVENT_CATEGORIES)].sum(axis=1).gt(1).sum())
    text = [
        "# 2025 event-day validation slices", "",
        "This is a validation-only, post-hoc diagnostic: no model was retrained. Event labels came from "
        "`load_development_model_table()` (which filters at the local-2026 boundary) and the typed event calendar.", "",
        "## Event-day counts", "",
        table(category_counts), "",
        f"`any_event` is the union of Nationals home games, Cherry Blossom Festival days, and US holidays. "
        f"It contains {event_days} of the 363 valid fixed-24-hour evaluation days. Categories with fewer than five "
        "days are marked **too few days for inference**; the metrics are descriptive only. Marathon is **not in the "
        "event file**, rather than a verified zero-event category.", "",
        f"{overlap_days} flagged dates have more than one category. Every date is included exactly once in the aggregate "
        "`any_event` slice; see `event_dates_2025.csv` for the underlying flags.", "",
        "## Seasonal composition by month", "",
        "Counts use the fixed 363-day evaluation window; `calendar_days` is the unfiltered calendar length.", "",
        table(monthly_counts), "",
        "## Model slices", "",
        "Each model has exactly 435,600 rows when its event and non-event rows are combined. Pinball is averaged over "
        "the shared 19 quantiles; coverage is the q05--q95 interval.", "",
        table(scores), "",
        "## Month-matched event comparison", "",
        "For every month with event days, event and non-event scores are calculated separately and then averaged with "
        "weights equal to the month's event-day count. Relative pinball is pinball divided by mean actual demand on "
        "the same slice.", "",
        table(month_matched), "",
        "## Paired XGBoost minus embedded daily transformer", "",
        "Intervals use 10,000 seed-42 resamples of complete local days within each slice, rather than resampling rows. "
        "Negative differences favour XGBoost; an interval containing zero is not distinguishable from zero by this diagnostic.", "",
        table(bootstrap), "",
        "## Month-matched paired XGBoost minus embedded daily transformer", "",
        "Intervals use 10,000 seed-42 day resamples stratified within event-containing months, with the same event-day "
        "weights. Negative differences favour XGBoost.", "",
        table(month_bootstrap), "",
        "## Month-stratified difference-in-differences", "",
        "This is `(XGBoost minus daily embedded transformer on event days) - (the same difference on month-matched "
        "non-event days)`. The bootstrap resamples complete days separately within each calendar-month/slice stratum. "
        "For relative pinball, each sampled model difference is divided by its sampled mean actual demand before the "
        "contrast is formed.", "",
        table(difference_in_differences), "",
        "## Reproduction", "",
        f"Command: `{command}`", "",
        f"Wall-clock time: {elapsed_minutes:.2f} minutes.",
    ]
    return "\n".join(text) + "\n"


def run(draws: int = 10_000, seed: int = cfg.SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = event_day_labels()
    valid_dates = set()
    score_rows: list[dict[str, object]] = []
    month_matched_rows: list[dict[str, object]] = []
    xgb_rows = daily_rows = None
    for model in MODEL_NAMES:
        path = cfg.PREDICTIONS_DIR / f"{model}_validation_2025.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing required saved predictions: {path}")
        prediction = pd.read_parquet(path)
        rows = label_prediction_rows(prediction, labels)
        if len(rows) != 435_600:
            raise AssertionError(f"{model} has {len(rows):,}, expected 435,600 rows")
        assert_event_partition(rows)
        valid_dates.update(rows.local_date.unique())
        for is_event, name in ((True, "event"), (False, "non_event")):
            score_rows.append({"model": model, "slice": name, **_score_slice(rows.loc[rows.any_event.eq(is_event)])})
        # Every artifact has the same fixed dates, so the month weights are shared.
        if len(valid_dates) == 363:
            counts = monthly_event_counts(labels, valid_dates)
            month_matched_rows.append({"model": model, **month_matched_scores(rows, counts.loc[counts.event_days.gt(0)])})
        if model == "xgboost_shared_quantile":
            xgb_rows = rows
        elif model == "hf_transformer_daily_embed_noweather":
            daily_rows = rows
    if len(valid_dates) != 363:
        raise AssertionError(f"expected 363 valid local dates, found {len(valid_dates)}")
    assert xgb_rows is not None and daily_rows is not None
    bootstrap = pd.DataFrame([
        paired_slice_bootstrap(xgb_rows, daily_rows, event, metric, draws=draws, seed=seed)
        for event in (True, False) for metric in ("pinball", "median_mae")
    ])
    monthly = monthly_event_counts(labels, valid_dates)
    event_months = monthly.loc[monthly.event_days.gt(0)].reset_index(drop=True)
    month_bootstrap = pd.DataFrame([
        paired_month_matched_bootstrap(xgb_rows, daily_rows, event_months, event, metric, draws=draws, seed=seed)
        for event in (True, False) for metric in ("pinball", "median_mae")
    ])
    difference_in_differences = pd.DataFrame([
        month_stratified_difference_in_differences(xgb_rows, daily_rows, event_months, metric, draws=draws, seed=seed)
        for metric in ("pinball", "relative_pinball")
    ])
    category_rows = []
    valid_set = valid_dates
    for category in (*EVENT_CATEGORIES, "any_event"):
        all_days = int(labels[category].sum())
        valid_days = int(labels.loc[labels.local_date.isin(valid_set), category].sum())
        category_rows.append({
            "category": "marathon (not in event file)" if category == "marathon" else category,
            "calendar_2025_days": all_days, "valid_evaluation_days": valid_days,
            "inference_note": "not in event file" if category == "marathon" else (
                "too few days for inference" if valid_days < 5 else "descriptive slice; not causal inference"),
        })
    return (pd.DataFrame(category_rows), monthly, pd.DataFrame(score_rows), pd.DataFrame(month_matched_rows),
            bootstrap, month_bootstrap, difference_in_differences)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    args = parser.parse_args()
    start = time.perf_counter()
    categories, monthly, scores, month_matched, bootstrap, month_bootstrap, difference_in_differences = run(
        draws=args.draws, seed=args.seed)
    elapsed = (time.perf_counter() - start) / 60
    category_out = categories.assign(record_type="event_day_count")
    monthly_out = monthly.assign(record_type="monthly_day_count")
    score_out = scores.assign(record_type="model_slice")
    month_matched_out = month_matched.assign(record_type="month_matched_model_slice")
    bootstrap_out = bootstrap.assign(record_type="paired_bootstrap")
    month_bootstrap_out = month_bootstrap.assign(record_type="month_matched_paired_bootstrap")
    did_out = difference_in_differences.assign(record_type="month_stratified_difference_in_differences")
    output = pd.concat([category_out, monthly_out, score_out, month_matched_out, bootstrap_out, month_bootstrap_out, did_out],
                       ignore_index=True, sort=False)
    csv_path = cfg.REPORTS_DIR / "event_day_results.csv"
    md_path = cfg.REPORTS_DIR / "event_day_results.md"
    event_dates_path = cfg.REPORTS_DIR / "event_dates_2025.csv"
    output.to_csv(csv_path, index=False)
    dates = event_dates_2025(event_day_labels())
    dates.to_csv(event_dates_path, index=False)
    command = f"python -m src.event_day_analysis --draws {args.draws} --seed {args.seed}"
    md_path.write_text(_render_markdown(categories, monthly, scores, month_matched, bootstrap, month_bootstrap,
                                        difference_in_differences, dates, elapsed, command), encoding="utf-8")
    print(categories.to_string(index=False))
    print(monthly.to_string(index=False))
    print(scores.to_string(index=False))
    print(bootstrap.to_string(index=False))
    print(month_bootstrap.to_string(index=False))
    print(difference_in_differences.to_string(index=False))
    print(f"wrote {csv_path}, {md_path}, and {event_dates_path}; wall-clock {elapsed:.2f} minutes")


if __name__ == "__main__":
    main()
