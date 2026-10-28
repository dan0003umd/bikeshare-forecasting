# Consolidated Interim Evidence

## Seed-43 sampling audit

The seed-43 audit did **not** retrain the transformer. It loaded the existing daily embedded/no-weather checkpoint and changed only the stochastic generation seed during the 100-sample quantile evaluation.

## Source: `reports/interim_evidence/results_table.md`

# Validation Results (2025)

All full-window rows below are 2025 validation results on the same 435,600 check-out station-hours: 50 stations x 363 local days x 24 hours. This validation period was also used for transformer early stopping and tuning; these are not 2026 test results.

| model | features | pinball_loss | crps | coverage_90 | median_mae | n_observations |
|---|---|---|---|---|---|---|
| xgboost_shared_quantile | Lagged check-outs (1, 2, 3, 24, 168 hours), local calendar/holiday/event features, station code; no observed or forecast weather | 0.5644 | 1.0520 | 0.900 | 1.5148 | 435600 |
| hf_transformer_daily_embed_observed_weather_oracle | 336 check-out history values, local calendar/holiday/event features, station embedding, realized temperature/precipitation/wind (ORACLE) | 0.5653 | 1.0536 | 0.949 | 1.5108 | 435600 |
| hf_transformer_daily_embed_noweather | 336 check-out history values, local calendar/holiday/event features, station embedding; no weather | 0.5755 | 1.0727 | 0.939 | 1.5426 | 435600 |
| hf_transformer_daily_noembed_noweather | 336 check-out history values, local calendar/holiday/event features; no station embedding or weather | 0.5770 | 1.0753 | 0.949 | 1.5419 | 435600 |
| transformer_xgbfeatures_longcontext_nb | 504 past check-out hours (formal lags 1, 2, 3, 24, 168), local calendar/holiday/event features, station embedding/code; no weather; Negative Binomial head | 0.5842 | 1.0891 | 0.945 | 1.5672 | 435600 |
| transformer_xgbfeatures_longcontext_quantile | 504 past check-out hours (formal lags 1, 2, 3, 24, 168), local calendar/holiday/event features, station embedding/code; no weather; direct ordered 19-quantile pinball head | 0.5981 | 1.1149 | 0.912 | 1.6046 | 435600 |
| hf_transformer_negative_binomial | 336 check-out history values, local calendar/holiday/event features, station embedding; no weather | 0.6345 | 1.1819 | 0.933 | 1.6922 | 435600 |
| seasonal_naive_rolling_8week | Previous 8 same-station weekday/hour demand values; no weather | 0.6659 | 1.2358 | 0.783 | 1.7306 | 435600 |
| seasonal_naive_frozen_train_8week | Last 8 pre-2025 same-station weekday/hour demand values; no weather | 0.7492 | 1.3884 | 0.761 | 1.9293 | 435600 |
| per_station_negative_binomial | Cyclic hour, weekday, and month calendar features; no weather | 0.8157 | 1.5178 | 0.960 | 2.1441 | 435600 |

Pinball loss is averaged over the fixed 19 quantiles (0.05--0.95 by 0.05). CRPS is approximated as twice the trapezoidal integral of pinball loss across that grid, so it is related to, not independent of, pinball loss.

## Quantile construction

Seasonal-naive quantiles are empirical quantiles of their eight historical demand values. The per-station Negative Binomial and longer-context NB transformer use analytic inverse-CDF quantiles. XGBoost is one multi-quantile `reg:quantileerror` model trained directly for all 19 levels. The original Hugging Face transformers use empirical quantiles of 100 generated Negative Binomial samples. The longer-context direct-head transformer outputs sorted 19 quantiles and is optimized with pinball loss. Chronos-2 supplies native quantiles, interpolated to the common 19-level grid when necessary.

`observed_weather_oracle` rows use realized future weather, not an operational forecast; they are explicitly non-deployable. Only check-outs have been modeled and scored so far; return-demand results are not available.

No saved scored prediction target falls in the local 2026 test period (the final local-2025 evening has a 2026 UTC timestamp). However, the earlier baseline/transformer runs loaded an assembled table that contains later archives before applying their split masks; their scores did not use local-2026 labels, but a literal claim that those historical jobs never read a 2026 row would be false. The new XGBoost and Chronos scripts use Parquet filters that prevent local-2026 rows from being read.

## Paired day-bootstrap comparisons

Each comparison is A minus B for the named metric. The 95% percentile intervals use 10,000 paired bootstrap resamples of the 363 local dates (not individual station-hours). Negative values favour A.

| comparison | metric | difference_a_minus_b | ci95_low | ci95_high | days | interpretation |
|---|---|---|---|---|---|---|
| daily_embed_noweather_minus_rolling_seasonal | pinball | -0.0904 | -0.1019 | -0.0796 | 363 | distinguishable from zero |
| daily_embed_noweather_minus_daily_noembed_noweather | pinball | -0.0015 | -0.0046 | 0.0016 | 363 | not distinguishable from zero |
| daily_embed_noweather_minus_observed_weather_oracle | pinball | 0.0102 | 0.0040 | 0.0172 | 363 | distinguishable from zero |
| xgboost_minus_daily_embed_noweather | pinball | -0.0112 | -0.0166 | -0.0063 | 363 | distinguishable from zero |
| xgboost_minus_daily_embed_noweather | median_mae | -0.0278 | -0.0422 | -0.0149 | 363 | distinguishable from zero |
| xgboost_minus_longcontext_nb_transformer | pinball | -0.0198 | -0.0260 | -0.0143 | 363 | distinguishable from zero |
| xgboost_minus_longcontext_nb_transformer | median_mae | -0.0524 | -0.0685 | -0.0378 | 363 | distinguishable from zero |
| xgboost_minus_longcontext_quantile_transformer | pinball | -0.0337 | -0.0383 | -0.0294 | 363 | distinguishable from zero |
| xgboost_minus_longcontext_quantile_transformer | median_mae | -0.0899 | -0.1022 | -0.0783 | 363 | distinguishable from zero |

## Targeted longer-context transformer variants

Both variants use the no-weather XGBoost covariate set and were tuned with 2025 validation early stopping. They are compared against XGBoost on the same 435,600 rows; their paired bootstrap rows appear above.

| model | pinball_loss | crps | coverage_90 | median_mae | n_observations |
|---|---|---|---|---|---|
| transformer_xgbfeatures_longcontext_nb | 0.5842 | 1.0891 | 0.945 | 1.5672 | 435600 |
| transformer_xgbfeatures_longcontext_quantile | 0.5981 | 1.1149 | 0.912 | 1.6046 | 435600 |

Selected on 2025: `xgboost_shared_quantile` (pinball 0.5644); neither targeted transformer beat it. Selecting among several variants on this same tuning period makes the selected validation score slightly optimistic and it is not a 2026 test result.

## Transformer sampling variation

The sampled transformer was regenerated over all 435,600 validation rows with seed 43 instead of seed 42. Across all 19 quantiles, the mean absolute prediction difference was 0.3061; for the median it was 0.2578. Pinball changed by +0.000022, and median MAE by -0.000715.

## Chronos-2 fixed subsample


Chronos-2 zero-shot and every full-window model were scored on the identical deterministic 14,400-row slice: the first 10 sorted station IDs and first 60 complete 2025 local days (2025-01-01 through 2025-03-01). This is a validation-only comparison, not a 2026 test result.

| model | pinball_loss | crps | coverage_90 | median_mae | n_observations |
|---|---|---|---|---|---|
| hf_transformer_daily_embed_observed_weather_oracle | 0.4658 | 0.8681 | 0.950 | 1.2480 | 14400 |
| xgboost_shared_quantile | 0.4790 | 0.8931 | 0.898 | 1.2883 | 14400 |
| hf_transformer_daily_noembed_noweather | 0.4929 | 0.9188 | 0.946 | 1.3246 | 14400 |
| hf_transformer_daily_embed_noweather | 0.4961 | 0.9245 | 0.936 | 1.3305 | 14400 |
| chronos2_zero_shot | 0.5329 | 0.9932 | 0.858 | 1.4296 | 14400 |
| hf_transformer_negative_binomial | 0.5428 | 1.0111 | 0.923 | 1.4488 | 14400 |
| seasonal_naive_rolling_8week | 0.5909 | 1.0974 | 0.796 | 1.5411 | 14400 |
| seasonal_naive_frozen_train_8week | 0.5925 | 1.1026 | 0.821 | 1.5704 | 14400 |
| per_station_negative_binomial | 0.6428 | 1.1961 | 0.957 | 1.6785 | 14400 |

## Source: `reports/interim_evidence/chronos_subsample_results.md`

# Chronos-2 Fixed-Subsample Validation (2025)

Chronos-2 zero-shot and every full-window model were scored on the identical deterministic 14,400-row slice: the first 10 sorted station IDs and first 60 complete 2025 local days (2025-01-01 through 2025-03-01). This is a validation-only comparison, not a 2026 test result.

| model | pinball_loss | crps | coverage_90 | median_mae | n_observations |
|---|---|---|---|---|---|
| hf_transformer_daily_embed_observed_weather_oracle | 0.4658 | 0.8681 | 0.950 | 1.2480 | 14400 |
| xgboost_shared_quantile | 0.4790 | 0.8931 | 0.898 | 1.2883 | 14400 |
| hf_transformer_daily_noembed_noweather | 0.4929 | 0.9188 | 0.946 | 1.3246 | 14400 |
| hf_transformer_daily_embed_noweather | 0.4961 | 0.9245 | 0.936 | 1.3305 | 14400 |
| chronos2_zero_shot | 0.5329 | 0.9932 | 0.858 | 1.4296 | 14400 |
| hf_transformer_negative_binomial | 0.5428 | 1.0111 | 0.923 | 1.4488 | 14400 |
| seasonal_naive_rolling_8week | 0.5909 | 1.0974 | 0.796 | 1.5411 | 14400 |
| seasonal_naive_frozen_train_8week | 0.5925 | 1.1026 | 0.821 | 1.5704 | 14400 |
| per_station_negative_binomial | 0.6428 | 1.1961 | 0.957 | 1.6785 | 14400 |

## Source: `reports/interim_evidence/split_integrity.md`

# Split Integrity and Locked Test Data

- Station selection uses only rows before `2025-01-01`; this is stricter than the 2026 boundary.
- The development `model_table.parquet` contains only timestamps before local `2026-01-01`. `327,550` local-2026 rows are physically stored in the ignored locked file `data/processed/locked_test_2026/model_table.parquet`.
- Every training, tuning, scoring, and figure entrypoint loads this table through `load_development_model_table`, which applies a Parquet predicate and raises if a local-2026 row is present.
- Transformer observed-weather standardization computes mean and standard deviation from the 2023--2024 training mask only. Calendar features need no fitted scaler; XGBoost uses unscaled lag/calendar inputs; Chronos-2 is zero-shot and fits no project-level normalization.
- `tests/test_metrics_splits.py` verifies both the predicate/guard and that each development entrypoint routes through the guarded loader.

## Source: `reports/interim_evidence/interim_evidence.md`

# Interim Evidence Summary

- Built a 50-station hourly check-out/return table from 45 monthly archives; it contains 1,642,750 station-hours and no nulls after feature assembly.
- The train/validation boundary is fixed at local 2025-01-01; all full-window numbers use the same 435,600 2025 check-out observations. This period was also used for early stopping and tuning, so these are not test results. The 2026 test split remains locked.
- The two DST transition days are excluded because a local day has 23 or 25 physical hours; this leaves 363 valid fixed-24-hour days.
- The restored causal rolling seasonal naive has pinball loss 0.6659 and median MAE 1.7306. The daily-stride transformer with station embeddings and no weather improves these to 0.5755 and 1.5426.
- The observed-weather ORACLE reaches 0.5653 pinball loss and 1.5108 median MAE, but it uses realized future weather and is not deployable. Previous Runs forecast-weather coverage is incomplete before 2025.
- Daily-stride transformer ablations used seed 42 and patience-3 early stopping. XGBoost and Chronos-2 have now been evaluated; the LSTM is intentionally deferred to Dylan.
- The shared XGBoost quantile baseline, fit only on 2023--2024 data, has pinball loss 0.5644, CRPS 1.0520, and median MAE 1.5148; its fit took 3.19 minutes.
- The targeted longer-context no-weather transformer variants used the XGBoost covariate set. The best was `transformer_xgbfeatures_longcontext_nb` at 0.5842 pinball loss, so XGBoost (0.5644) remains selected on 2025; this several-variant choice makes its validation score slightly optimistic and is not a test result.
- Chronos-2 zero-shot was evaluated only on the fixed 14,400-row 2025 validation subsample; its pinball loss was 0.5329.
- Across the seed-42 and seed-43 transformer sampling runs, the mean absolute difference across quantile values was 0.3061; pinball changed by +0.0000 and median MAE by -0.0007.
- No saved scored target falls in the local 2026 test period (the final local-2025 evening has a 2026 UTC timestamp). Historical baseline/transformer scripts did load an assembled table containing later archives before masking to 2025, so the project cannot claim that no 2026 row was ever read; the new XGBoost and Chronos loaders use pre-2026 Parquet predicates.

## Source: `reports/interim_evidence/challenges.md`

# Challenges and Changes

| Date | Status | Issue and response |
|---|---|---|
| 2026-10-26 | solved | Current trip archives used the expected `ride_id`, timestamps, and station-id columns; the builder verifies those columns and fails loudly if they change. |
| 2026-10-26 | solved | 205 duplicate rides appeared across month boundaries; the builder de-duplicated boundary ride IDs. |
| 2026-10-26 | worked around | 3,812,905 start and 3,974,624 end station fields are missing in raw trips (exact side-specific counts are in `data_quality.md`); each available trip side remains usable. |
| 2026-10-26 | solved | Repeated November local hours are ambiguous. They are flagged `dst_overlap`; the two DST transition days are excluded from fixed-24-hour evaluation. |
| 2026-10-26 | worked around | Previous Runs lead-1 weather is incomplete in early years. Observed weather is joined but not yet passed to a reported model; any future observed-weather result will be labelled ORACLE. |
| 2026-10-27 | solved | Review showed the original rolling seasonal naive was causal: it used the preceding eight same-weekday/hour observations, all before each forecast origin. It is restored as `seasonal_naive_rolling_8week`; the frozen 2023--2024-only version is retained as a separate stricter comparator. |
| 2026-10-27 | solved | Daily-stride transformer runs used 35,650 train and 18,150 validation windows with seed 42 and patience-3 early stopping. Embedding, no-embedding, and observed-weather ORACLE ablations were evaluated on the common window. |
| 2026-10-27 | solved | Every evaluator now persists actuals and all 19 quantiles by station/hour under ignored `data/predictions/`, so figures can be rebuilt without re-evaluating. |
| 2026-10-27 | worked around | Observed future weather produces an ORACLE comparison only; incomplete 2023--2024 Previous Runs coverage prevents a deployable forecast-weather experiment. |

## Source: `reports/interim_evidence/data_quality.md`

# Data Quality

- Input monthly archives: 45 (202301 through 202609).
- Rows read: 22,072,596; table rows: 1,642,750.
- Selected stations: 50 of 911 seen (678 active across the full observed window).
- Rows with unparseable timestamps: 0; within-file duplicate rides removed: 0; cross-file boundary duplicates removed: 205.
- Trips lacking a start station: 3,812,905 (17.27% of rows read); lacking an end station: 3,974,624 (18.01%). Demand counts only trip sides with a station: check-outs require a start station and returns require an end station.
- Trips with end before start: 986; these are reported for audit but not discarded by the present aggregation.
- Daylight saving: the repeated November local hour is assigned to the first occurrence and both physical hours receive `dst_overlap=True`. There are 300 flagged station-hours in the built table; evaluation excludes DST/non-24-hour local days.

Stations are ranked by check-outs strictly before 2025-01-01, after requiring activity near both ends of the downloaded period. The top 50 are then zero-filled at every UTC hour.

## Source: `reports/interim_evidence/weather_probe.md`

# Previous Runs Weather Coverage Probe

Each row is a 48-hour Jan 1--2 UTC request for the `*_previous_day1` series.
A series is usable only when all requested values are present.

| Year | Variable | Non-null / requested hours | Complete |
|---:|---|---:|---|
| 2023 | temperature_2m | 48 / 48 | True |
| 2023 | precipitation | 0 / 48 | False |
| 2023 | wind_speed_10m | 0 / 48 | False |
| 2024 | temperature_2m | 0 / 48 | False |
| 2024 | precipitation | 0 / 48 | False |
| 2024 | wind_speed_10m | 0 / 48 | False |
| 2025 | temperature_2m | 48 / 48 | True |
| 2025 | precipitation | 48 / 48 | True |
| 2025 | wind_speed_10m | 48 / 48 | True |
| 2026 | temperature_2m | 48 / 48 | True |
| 2026 | precipitation | 48 / 48 | True |
| 2026 | wind_speed_10m | 48 / 48 | True |

The 2023 precipitation and wind gaps make a full 2023--2025 forecast-weather design invalid. Observed weather is joined in the model table, but the reported first-run models use calendar features only; any future observed-weather ablation must be labelled ORACLE.

## Source: `reports/interim_evidence/forecast_history.md`

# Forecast-Origin History Available to Each Model

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

## Source: `reports/interim_evidence/reproduce.md`

# Evidence Reproduction

Run from the repository root after activating `.\.venv\Scripts\Activate.ps1`:

```powershell
python -m src.check_trip_availability
python -m src.probe_weather_coverage
python -m src.baselines --model seasonal_rolling
python -m src.baselines --model seasonal_frozen
python -m src.baselines --model negative_binomial
python -m src.xgboost_baseline --max-minutes 60
python -m src.train_transformer --daily-stride --patience 3 --run-name transformer_daily_embed_noweather
python -m src.evaluate_transformer --checkpoint checkpoints/transformer_daily_embed_noweather_best.pt --model-name hf_transformer_daily_embed_noweather --batch-size 32
python -m src.train_transformer --daily-stride --no-station-embedding --patience 3 --run-name transformer_daily_noembed_noweather
python -m src.evaluate_transformer --checkpoint checkpoints/transformer_daily_noembed_noweather_best.pt --model-name hf_transformer_daily_noembed_noweather --batch-size 32
python -m src.train_transformer --daily-stride --observed-weather-oracle --patience 3 --run-name transformer_daily_embed_observed_weather_oracle
python -m src.evaluate_transformer --checkpoint checkpoints/transformer_daily_embed_observed_weather_oracle_best.pt --model-name hf_transformer_daily_embed_observed_weather_oracle --observed-weather-oracle --batch-size 32
python -m src.bootstrap_comparisons
python -m src.chronos_subsample --batch-size 8
python -m src.make_figures
python -m src.build_interim_evidence
python -m pytest -q
```

The checked-in experiment settings are `src/config.py`: seed 42, training 2023--2024, validation 2025, test locked behind `--final`, 50 stations, and the 19-quantile 0.05--0.95 grid. The committed scripts regenerate the tables and figures; `data/` and `checkpoints/` are intentionally ignored and must be supplied/generated locally.

The initial transformer checkpoint and loss log came from a five-epoch weekly-stride run. The daily-stride commands above produce the ablation rows and their loss CSVs. The observed-weather command is an ORACLE diagnostic: it passes realized future observed weather and is not a deployable model. XGBoost has a 60-minute wall-clock training cap and uses a parquet predicate excluding 2026. Chronos downloads `amazon/chronos-2` on first run and is intentionally scored only on the recorded 10-station x 60-day validation subsample.

## Source: `reports/handoff_one_pager.md`

# Dylan Handoff: Interim Forecasting Evidence

## Current evidence

All full-window metrics are **2025 validation**, not test metrics: the same 2025 period was used for transformer early stopping and tuning. The held-out 2026 test remains locked. The strongest deployable full-window model is the shared XGBoost quantile baseline (pinball 0.5644, CRPS 1.0520, 90% coverage 0.900, median MAE 1.5148). The observed-weather transformer Oracle is non-deployable because it uses realized future weather.

| Model | Pinball | Median MAE | Scope |
|---|---:|---:|---|
| Shared XGBoost, 19 quantiles | 0.5644 | 1.5148 | Full 435,600-row 2025 validation window |
| Transformer, observed-weather Oracle | 0.5653 | 1.5108 | Full window; non-deployable |
| Transformer, embeddings, no weather | 0.5755 | 1.5426 | Full window |
| Rolling 8-week seasonal naive | 0.6659 | 1.7306 | Full window, causal rolling history |
| Chronos-2 zero-shot | 0.5329 | 1.4296 | Fixed 14,400-row 2025 subsample only |

The embedding versus no-embedding transformer difference is not distinguishable from zero with a paired 363-day bootstrap. The no-weather transformer is clearly better than rolling seasonal naive; the realized-weather Oracle is clearly better than the no-weather transformer, but is an Oracle diagnostic only.

## Artifacts and reproduction

- Full results and bootstrap intervals: `reports/interim_evidence/results_table.md`.
- Chronos comparison: `reports/interim_evidence/chronos_subsample_results.md`; slice definition: `chronos_subsample_definition.csv`.
- Rebuild figures from ignored per-row artifacts: `python -m src.make_figures`.
- Each ignored `data/predictions/*validation_2025.parquet` has `station_id`, `ts_utc`, `actual`, and `q05` through `q95` for all 435,600 full-window rows.
- Reproduce XGBoost (pre-2026 parquet predicate, 60-minute hard budget): `python -m src.xgboost_baseline --max-minutes 60`.
- Reproduce Chronos: `python -m src.chronos_subsample --batch-size 8`. It downloads `amazon/chronos-2` on first run.

## Important caveats

- Scores use check-outs only; return-demand modeling and event-day evaluation remain unfinished.
- The raw assembled table contains post-2025 archives. Historical baseline/transformer scripts loaded that table before applying their split masks, so no score uses a local-2026 label but we cannot claim historical jobs never read a 2026 row. New XGBoost and Chronos scripts use a Parquet filter and never read local-2026 rows.
- Previous Runs forecast-weather data are incomplete in 2023--2024; do not label observed-weather results as deployable weather forecasts.
- Chronos was run zero-shot only on the fixed subsample, not on all 435,600 rows.

## Suggested next work for Dylan

1. **LSTM is intentionally skipped for this checkpoint.** Implement it only if it is needed for the final report, using the same 2023--2024 train / 2025 validation guard and the common quantile scorer.
2. Add the return-demand target and repeat the model comparison without touching the 2026 test set.
3. Add event-day slices and calibration plots from the saved predictions.
4. Resolve forecast-weather coverage or state it as unavailable; never substitute realized weather for a production feature.
5. Run 2026 exactly once only after final model choice and report it separately from tuning metrics.
