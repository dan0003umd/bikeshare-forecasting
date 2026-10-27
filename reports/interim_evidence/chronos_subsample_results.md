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
