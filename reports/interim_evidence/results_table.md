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
