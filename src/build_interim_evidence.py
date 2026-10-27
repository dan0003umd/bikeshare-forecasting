"""Write the factual interim evidence package from saved experiment outputs."""
from __future__ import annotations

import json

import pandas as pd

from . import config as cfg


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for row in frame[columns].itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join([header, rule, *rows])


FEATURES = {
    "seasonal_naive_rolling_8week": "Previous 8 same-station weekday/hour demand values; no weather",
    "seasonal_naive_frozen_train_8week": "Last 8 pre-2025 same-station weekday/hour demand values; no weather",
    "per_station_negative_binomial": "Cyclic hour, weekday, and month calendar features; no weather",
    "hf_transformer_negative_binomial": "336 check-out history values, local calendar/holiday/event features, station embedding; no weather",
    "hf_transformer_daily_embed_noweather": "336 check-out history values, local calendar/holiday/event features, station embedding; no weather",
    "hf_transformer_daily_noembed_noweather": "336 check-out history values, local calendar/holiday/event features; no station embedding or weather",
    "hf_transformer_daily_embed_observed_weather_oracle": "336 check-out history values, local calendar/holiday/event features, station embedding, realized temperature/precipitation/wind (ORACLE)",
    "xgboost_shared_quantile": "Lagged check-outs (1, 2, 3, 24, 168 hours), local calendar/holiday/event features, station code; no observed or forecast weather",
}


def main() -> None:
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(cfg.REPORTS_DIR / "results_validation_2025.csv")
    if not results.n_observations.eq(435600).all():
        raise ValueError("all interim models must use the common 435,600-observation window")
    results = results.sort_values("pinball_loss").reset_index(drop=True)
    results.to_csv(cfg.REPORTS_DIR / "results_table.csv", index=False)
    display = results.assign(
        pinball_loss=results.pinball_loss.map("{:.4f}".format), crps=results.crps.map("{:.4f}".format),
        coverage_90=results.coverage_90.map("{:.3f}".format), median_mae=results.median_mae.map("{:.4f}".format),
        features=results.model.map(FEATURES),
    )
    results_md = ["# Validation Results (2025)", "",
                  "All full-window rows below are 2025 validation results on the same 435,600 check-out station-hours: 50 stations x 363 local days x 24 hours. This validation period was also used for transformer early stopping and tuning; these are not 2026 test results.", "",
                  _table(display, ["model", "features", "pinball_loss", "crps", "coverage_90", "median_mae", "n_observations"]), "",
                  "Pinball loss is averaged over the fixed 19 quantiles (0.05--0.95 by 0.05). CRPS is approximated as twice the trapezoidal integral of pinball loss across that grid, so it is related to, not independent of, pinball loss.",
                  "", "## Quantile construction", "",
                  "Seasonal-naive quantiles are empirical quantiles of their eight historical demand values. The per-station Negative Binomial uses analytic inverse-CDF quantiles. XGBoost is one multi-quantile `reg:quantileerror` model trained directly for all 19 levels. Transformer quantiles are empirical quantiles of 100 generated Negative Binomial samples. Chronos-2 supplies native quantiles, interpolated to the common 19-level grid when necessary.",
                  "", "`observed_weather_oracle` rows use realized future weather, not an operational forecast; they are explicitly non-deployable. Only check-outs have been modeled and scored so far; return-demand results are not available.",
                  "", "No saved scored prediction target falls in the local 2026 test period (the final local-2025 evening has a 2026 UTC timestamp). However, the earlier baseline/transformer runs loaded an assembled table that contains later archives before applying their split masks; their scores did not use local-2026 labels, but a literal claim that those historical jobs never read a 2026 row would be false. The new XGBoost and Chronos scripts use Parquet filters that prevent local-2026 rows from being read."]
    bootstrap_path = cfg.REPORTS_DIR / "paired_bootstrap_validation_2025.csv"
    if bootstrap_path.exists():
        bootstrap = pd.read_csv(bootstrap_path).assign(
            difference_a_minus_b=lambda x: x.difference_a_minus_b.map("{:.4f}".format),
            ci95_low=lambda x: x.ci95_low.map("{:.4f}".format),
            ci95_high=lambda x: x.ci95_high.map("{:.4f}".format),
            interpretation=lambda x: x.ci_includes_zero.map(
                lambda includes: "not distinguishable from zero" if includes else "distinguishable from zero"),
        )
        results_md += ["", "## Paired day-bootstrap comparisons", "",
                       "Each comparison is A minus B for the named metric. The 95% percentile intervals use 10,000 paired bootstrap resamples of the 363 local dates (not individual station-hours). Negative values favour A.", "",
                       _table(bootstrap, ["comparison", "metric", "difference_a_minus_b", "ci95_low", "ci95_high", "days", "interpretation"])]
    sampling_path = cfg.REPORTS_DIR / "transformer_sampling_audit.csv"
    if sampling_path.exists():
        sampling = pd.read_csv(sampling_path).iloc[0]
        results_md += ["", "## Transformer sampling variation", "",
                       f"The sampled transformer was regenerated over all {int(sampling.rows):,} validation rows with seed {int(sampling.seed_b)} instead of seed {int(sampling.seed_a)}. Across all 19 quantiles, the mean absolute prediction difference was {sampling.mean_absolute_quantile_difference:.4f}; for the median it was {sampling.q50_mean_absolute_difference:.4f}. Pinball changed by {sampling.pinball_difference_seed43_minus_seed42:+.6f}, and median MAE by {sampling.median_mae_difference_seed43_minus_seed42:+.6f}."]
    chronos_path = cfg.REPORTS_DIR / "chronos_subsample_validation_2025.csv"
    if chronos_path.exists():
        chronos = pd.read_csv(chronos_path).sort_values("pinball_loss").assign(
            pinball_loss=lambda x: x.pinball_loss.map("{:.4f}".format),
            crps=lambda x: x.crps.map("{:.4f}".format),
            coverage_90=lambda x: x.coverage_90.map("{:.3f}".format),
            median_mae=lambda x: x.median_mae.map("{:.4f}".format),
        )
        chronos_text = ["# Chronos-2 Fixed-Subsample Validation (2025)", "",
                        "Chronos-2 zero-shot and every full-window model were scored on the identical deterministic 14,400-row slice: the first 10 sorted station IDs and first 60 complete 2025 local days (2025-01-01 through 2025-03-01). This is a validation-only comparison, not a 2026 test result.", "",
                        _table(chronos, ["model", "pinball_loss", "crps", "coverage_90", "median_mae", "n_observations"])]
        (cfg.REPORTS_DIR / "chronos_subsample_results.md").write_text("\n".join(chronos_text) + "\n", encoding="utf-8")
        results_md += ["", "## Chronos-2 fixed subsample", "", *chronos_text[1:]]
    (cfg.REPORTS_DIR / "results_table.md").write_text("\n".join(results_md) + "\n", encoding="utf-8")

    ablations = results[results.model.str.contains("daily_", regex=False)].copy()
    if len(ablations):
        ablations = ablations.assign(
            pinball_loss=ablations.pinball_loss.map("{:.4f}".format), crps=ablations.crps.map("{:.4f}".format),
            coverage_90=ablations.coverage_90.map("{:.3f}".format), median_mae=ablations.median_mae.map("{:.4f}".format),
        )
        text = ["# Daily-Stride Transformer Ablations", "",
                _table(ablations, ["model", "pinball_loss", "crps", "coverage_90", "median_mae", "n_observations"]), "",
                "All rows use the common 435,600-row 2025 validation window. `observed_weather_oracle` uses realized future weather and is not deployable; no model is described as using forecast weather."]
        (cfg.REPORTS_DIR / "ablation_summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")

    report = json.loads((cfg.PROCESSED_DIR / "build_report.json").read_text())
    missing_start_share = 100 * report["missing_start_station"] / report["rows_read"]
    missing_end_share = 100 * report["missing_end_station"] / report["rows_read"]
    quality = f"""# Data Quality\n\n- Input monthly archives: {len(report['months'])} ({report['months'][0]} through {report['months'][-1]}).\n- Rows read: {report['rows_read']:,}; table rows: {report['table_rows']:,}.\n- Selected stations: {report['stations_selected']} of {report['stations_seen']} seen ({report['stations_active_whole_period']} active across the full observed window).\n- Rows with unparseable timestamps: {report['bad_timestamps']:,}; within-file duplicate rides removed: {report['within_file_duplicates']:,}; cross-file boundary duplicates removed: {report['cross_file_duplicates']:,}.\n- Trips lacking a start station: {report['missing_start_station']:,} ({missing_start_share:.2f}% of rows read); lacking an end station: {report['missing_end_station']:,} ({missing_end_share:.2f}%). Demand counts only trip sides with a station: check-outs require a start station and returns require an end station.\n- Trips with end before start: {report['end_before_start']:,}; these are reported for audit but not discarded by the present aggregation.\n- Daylight saving: the repeated November local hour is assigned to the first occurrence and both physical hours receive `dst_overlap=True`. There are 300 flagged station-hours in the built table; evaluation excludes DST/non-24-hour local days.\n\nStations are ranked by check-outs strictly before 2025-01-01, after requiring activity near both ends of the downloaded period. The top 50 are then zero-filled at every UTC hour.\n"""
    (cfg.REPORTS_DIR / "data_quality.md").write_text(quality, encoding="utf-8")

    locked_rows = "not yet separated"
    if cfg.LOCKED_TEST_MODEL_TABLE_PATH.exists():
        locked_rows = f"{len(pd.read_parquet(cfg.LOCKED_TEST_MODEL_TABLE_PATH, columns=['ts_utc'])):,}"
    integrity = f"""# Split Integrity and Locked Test Data

- Station selection uses only rows before `{cfg.SELECTION_END}`; this is stricter than the 2026 boundary.
- The development `model_table.parquet` contains only timestamps before local `{cfg.TEST_START}`. `{locked_rows}` local-2026 rows are physically stored in the ignored locked file `data/processed/locked_test_2026/model_table.parquet`.
- Every training, tuning, scoring, and figure entrypoint loads this table through `load_development_model_table`, which applies a Parquet predicate and raises if a local-2026 row is present.
- Transformer observed-weather standardization computes mean and standard deviation from the 2023--2024 training mask only. Calendar features need no fitted scaler; XGBoost uses unscaled lag/calendar inputs; Chronos-2 is zero-shot and fits no project-level normalization.
- `tests/test_metrics_splits.py` verifies both the predicate/guard and that each development entrypoint routes through the guarded loader.
"""
    (cfg.REPORTS_DIR / "split_integrity.md").write_text(integrity, encoding="utf-8")

    history = """# Forecast-Origin History Available to Each Model

| Model | Fitted parameters/history | Context available at a 2025 local-midnight origin | Earlier 2025 observed demand? |
|---|---|---|---|
| Seasonal naive, rolling 8-week | No fitted parameters; empirical 19 quantiles | The previous eight observations for the same station, weekday, and local hour | Yes, when those observations precede the origin |
| Seasonal naive, frozen training | No fitted parameters; empirical 19 quantiles frozen at 2024 cutoff | The last eight training-period observations for the same station, weekday, and local hour | No |
| Per-station Negative Binomial | GLM fit on 2023--2024 only | Calendar covariates only; no demand lag/context | No |
| HF Transformer, first run | Parameters fit on 2023--2024 weekly-stride windows | 336 prior hourly values (168 context + maximum 168-hour lag) and local calendar features | Yes, where those prior values precede the origin |
| HF Transformer, daily stride + embeddings | Parameters fit on 2023--2024 daily-stride windows | 336 prior hourly values, local calendar features, and station embedding | Yes, where those prior values precede the origin |
| HF Transformer, daily stride without embeddings | Parameters fit on 2023--2024 daily-stride windows | 336 prior hourly values and local calendar features; no station embedding | Yes, where those prior values precede the origin |
| HF Transformer, observed-weather ORACLE | Parameters fit on 2023--2024 daily-stride windows | 336 prior hourly values, station embedding, local calendar features, and realized future weather | Yes, where those prior values precede the origin |

All models are scored on the same 24 future hours for each of 363 eligible local days. The rolling seasonal baseline and transformer are causal with respect to an origin, but they are not frozen-history comparators; both can consume demand observed earlier in 2025 during sequential validation.
"""
    (cfg.REPORTS_DIR / "forecast_history.md").write_text(history, encoding="utf-8")

    challenges = """# Challenges and Changes\n\n| Date | Status | Issue and response |\n|---|---|---|\n| 2026-10-26 | solved | Current trip archives used the expected `ride_id`, timestamps, and station-id columns; the builder verifies those columns and fails loudly if they change. |\n| 2026-10-26 | solved | 205 duplicate rides appeared across month boundaries; the builder de-duplicated boundary ride IDs. |\n| 2026-10-26 | worked around | 3,813,? start/end station fields are missing in raw trips (exact side-specific counts are in `data_quality.md`); each available trip side remains usable. |\n| 2026-10-26 | solved | Repeated November local hours are ambiguous. They are flagged `dst_overlap`; the two DST transition days are excluded from fixed-24-hour evaluation. |\n| 2026-10-26 | worked around | Previous Runs lead-1 weather is incomplete in early years. Observed weather is joined but not yet passed to a reported model; any future observed-weather result will be labelled ORACLE. |\n| 2026-10-27 | solved | Seasonal naive initially rolled through earlier 2025 demand. It was changed to fit only 2023--2024 history, with a regression test. |\n| 2026-10-27 | open | The first transformer is a five-epoch weekly-stride run, so it is undertrained; daily-stride, early-stopped training and embedding/weather ablations remain. |\n| 2026-10-27 | open | Per-row transformer samples were not persisted after evaluation, so the current hourly-error figure only includes the retained seasonal-naive predictions. |\n""".replace("3,813,? start/end station fields are missing", f"{report['missing_start_station']:,} start and {report['missing_end_station']:,} end station fields are missing")
    challenges = """# Challenges and Changes\n\n| Date | Status | Issue and response |\n|---|---|---|\n| 2026-10-26 | solved | Current trip archives used the expected `ride_id`, timestamps, and station-id columns; the builder verifies those columns and fails loudly if they change. |\n| 2026-10-26 | solved | 205 duplicate rides appeared across month boundaries; the builder de-duplicated boundary ride IDs. |\n| 2026-10-26 | worked around | 3,813,? start/end station fields are missing in raw trips (exact side-specific counts are in `data_quality.md`); each available trip side remains usable. |\n| 2026-10-26 | solved | Repeated November local hours are ambiguous. They are flagged `dst_overlap`; the two DST transition days are excluded from fixed-24-hour evaluation. |\n| 2026-10-26 | worked around | Previous Runs lead-1 weather is incomplete in early years. Observed weather is joined but not yet passed to a reported model; any future observed-weather result will be labelled ORACLE. |\n| 2026-10-27 | solved | Review showed the original rolling seasonal naive was causal: it used the preceding eight same-weekday/hour observations, all before each forecast origin. It is restored as `seasonal_naive_rolling_8week`; the frozen 2023--2024-only version is retained as a separate stricter comparator. |\n| 2026-10-27 | open | The first transformer is a five-epoch weekly-stride run, so it is undertrained; daily-stride, early-stopped training and embedding/weather ablations remain. |\n| 2026-10-27 | solved | Every evaluator now persists actuals and all 19 quantiles by station/hour under ignored `data/predictions/`, so figures can be rebuilt without re-evaluating. |\n""".replace("3,813,? start/end station fields are missing", f"{report['missing_start_station']:,} start and {report['missing_end_station']:,} end station fields are missing")
    challenges = challenges.replace(
        "| 2026-10-27 | open | The first transformer is a five-epoch weekly-stride run, so it is undertrained; daily-stride, early-stopped training and embedding/weather ablations remain. |",
        "| 2026-10-27 | solved | Daily-stride transformer runs used 35,650 train and 18,150 validation windows with seed 42 and patience-3 early stopping. Embedding, no-embedding, and observed-weather ORACLE ablations were evaluated on the common window. |",
    )
    challenges += "| 2026-10-27 | worked around | Observed future weather produces an ORACLE comparison only; incomplete 2023--2024 Previous Runs coverage prevents a deployable forecast-weather experiment. |\n"
    (cfg.REPORTS_DIR / "challenges.md").write_text(challenges, encoding="utf-8")

    rolling = results.loc[results.model == "seasonal_naive_rolling_8week"].iloc[0]
    daily = results.loc[results.model == "hf_transformer_daily_embed_noweather"].iloc[0]
    oracle = results.loc[results.model == "hf_transformer_daily_embed_observed_weather_oracle"].iloc[0]
    summary = f"""# Interim Evidence Summary\n\n- Built a 50-station hourly check-out/return table from {len(report['months'])} monthly archives; it contains {report['table_rows']:,} station-hours and no nulls after feature assembly.\n- The train/validation boundary is fixed at local 2025-01-01; all seven reported models use the same 435,600 2025 check-out observations. 2026 remains locked.\n- The two DST transition days are excluded because a local day has 23 or 25 physical hours; this leaves 363 valid fixed-24-hour days.\n- The restored causal rolling seasonal naive has pinball loss {rolling.pinball_loss:.4f} and median MAE {rolling.median_mae:.4f}. The daily-stride transformer with station embeddings and no weather improves these to {daily.pinball_loss:.4f} and {daily.median_mae:.4f}.\n- The observed-weather ORACLE reaches {oracle.pinball_loss:.4f} pinball loss and {oracle.median_mae:.4f} median MAE, but it uses realized future weather and is not deployable. Previous Runs forecast-weather coverage is incomplete before 2025.\n- Daily-stride transformer ablations used seed 42 and patience-3 early stopping. XGBoost, Chronos-2, and LSTM have not yet been evaluated.\n"""
    summary = summary.replace(
        "all seven reported models use the same 435,600 2025 check-out observations. 2026 remains locked.",
        "all full-window numbers use the same 435,600 2025 check-out observations. This period was also used for early stopping and tuning, so these are not test results. The 2026 test split remains locked.",
    )
    summary = summary.replace(
        "XGBoost, Chronos-2, and LSTM have not yet been evaluated.",
        "XGBoost and Chronos-2 have now been evaluated; the LSTM is intentionally deferred to Dylan.",
    )
    xgboost = results.loc[results.model == "xgboost_shared_quantile"]
    if len(xgboost):
        xgb = xgboost.iloc[0]
        summary += (f"- The shared XGBoost quantile baseline, fit only on 2023--2024 data, has pinball loss "
                    f"{xgb.pinball_loss:.4f}, CRPS {xgb.crps:.4f}, and median MAE {xgb.median_mae:.4f}; its fit took "
                    f"{xgb.fit_minutes:.2f} minutes.\n")
    if (cfg.REPORTS_DIR / "chronos_subsample_validation_2025.csv").exists():
        chronos_score = pd.read_csv(cfg.REPORTS_DIR / "chronos_subsample_validation_2025.csv").query("model == 'chronos2_zero_shot'").iloc[0]
        summary += (f"- Chronos-2 zero-shot was evaluated only on the fixed 14,400-row 2025 validation subsample; "
                    f"its pinball loss was {chronos_score.pinball_loss:.4f}.\n")
    sampling_path = cfg.REPORTS_DIR / "transformer_sampling_audit.csv"
    if sampling_path.exists():
        sampling = pd.read_csv(sampling_path).iloc[0]
        summary += (f"- Across the seed-{int(sampling.seed_a)} and seed-{int(sampling.seed_b)} transformer sampling runs, the mean absolute "
                    f"difference across quantile values was {sampling.mean_absolute_quantile_difference:.4f}; pinball changed by "
                    f"{sampling.pinball_difference_seed43_minus_seed42:+.4f} and median MAE by {sampling.median_mae_difference_seed43_minus_seed42:+.4f}.\n")
    summary += ("- No saved scored target falls in the local 2026 test period (the final local-2025 evening has a 2026 UTC timestamp). "
                "Historical baseline/transformer scripts did load an assembled table containing later archives before masking to 2025, "
                "so the project cannot claim that no 2026 row was ever read; the new XGBoost and Chronos loaders use pre-2026 Parquet predicates.\n")
    (cfg.REPORTS_DIR / "interim_evidence.md").write_text(summary, encoding="utf-8")
    print("wrote interim evidence markdown and results table")


if __name__ == "__main__":
    main()
