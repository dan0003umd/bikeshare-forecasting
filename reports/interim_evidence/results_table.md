# Validation Results (2025)

All full-window rows below are 2025 validation results on the same 435,600 check-out station-hours: 50 stations x 363 local days x 24 hours. This validation period was also used for transformer early stopping and tuning; these are not 2026 test results.

| model | pinball_loss | crps | coverage_90 | median_mae | n_observations |
|---|---|---|---|---|---|
| xgboost_shared_quantile | 0.5644 | 1.0520 | 0.900 | 1.5148 | 435600 |
| hf_transformer_daily_embed_observed_weather_oracle | 0.5653 | 1.0536 | 0.949 | 1.5108 | 435600 |
| hf_transformer_daily_embed_noweather | 0.5755 | 1.0727 | 0.939 | 1.5426 | 435600 |
| hf_transformer_daily_noembed_noweather | 0.5770 | 1.0753 | 0.949 | 1.5419 | 435600 |
| hf_transformer_negative_binomial | 0.6345 | 1.1819 | 0.933 | 1.6922 | 435600 |
| seasonal_naive_rolling_8week | 0.6659 | 1.2358 | 0.783 | 1.7306 | 435600 |
| seasonal_naive_frozen_train_8week | 0.7492 | 1.3884 | 0.761 | 1.9293 | 435600 |
| per_station_negative_binomial | 0.8157 | 1.5178 | 0.960 | 2.1441 | 435600 |

Pinball loss is averaged over the fixed 19 quantiles (0.05--0.95 by 0.05). CRPS is approximated as twice the trapezoidal integral of pinball loss across that grid, so it is related to, not independent of, pinball loss.

`observed_weather_oracle` rows use realized future weather, not an operational forecast; they are explicitly non-deployable. Only check-outs have been modeled and scored so far; return-demand results are not available.

No saved scored prediction target falls in the local 2026 test period (the final local-2025 evening has a 2026 UTC timestamp). However, the earlier baseline/transformer runs loaded an assembled table that contains later archives before applying their split masks; their scores did not use local-2026 labels, but a literal claim that those historical jobs never read a 2026 row would be false. The new XGBoost and Chronos scripts use Parquet filters that prevent local-2026 rows from being read.

## Paired day-bootstrap comparisons

Each comparison is A minus B in pinball loss. The 95% percentile intervals use 10,000 paired bootstrap resamples of the 363 local dates (not individual station-hours). Negative values favour A.

| comparison | difference_pinball_a_minus_b | ci95_low | ci95_high | days | interpretation |
|---|---|---|---|---|---|
| daily_embed_noweather_minus_rolling_seasonal | -0.0904 | -0.1019 | -0.0796 | 363 | distinguishable from zero |
| daily_embed_noweather_minus_daily_noembed_noweather | -0.0015 | -0.0046 | 0.0016 | 363 | not distinguishable from zero |
| daily_embed_noweather_minus_observed_weather_oracle | 0.0102 | 0.0040 | 0.0172 | 363 | distinguishable from zero |

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
