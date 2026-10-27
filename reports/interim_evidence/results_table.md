# Validation Results (2025)

All rows use the same 435,600 held-out 2025 check-out station-hours: 50 stations x 363 local days x 24 hours.

| model | pinball_loss | crps | coverage_90 | median_mae | n_observations |
|---|---|---|---|---|---|
| hf_transformer_daily_embed_observed_weather_oracle | 0.5653 | 1.0536 | 0.949 | 1.5108 | 435600 |
| hf_transformer_daily_embed_noweather | 0.5755 | 1.0727 | 0.939 | 1.5426 | 435600 |
| hf_transformer_daily_noembed_noweather | 0.5770 | 1.0753 | 0.949 | 1.5419 | 435600 |
| hf_transformer_negative_binomial | 0.6345 | 1.1819 | 0.933 | 1.6922 | 435600 |
| seasonal_naive_rolling_8week | 0.6659 | 1.2358 | 0.783 | 1.7306 | 435600 |
| seasonal_naive_frozen_train_8week | 0.7492 | 1.3884 | 0.761 | 1.9293 | 435600 |
| per_station_negative_binomial | 0.8157 | 1.5178 | 0.960 | 2.1441 | 435600 |

Pinball loss is averaged over the fixed 19 quantiles (0.05--0.95 by 0.05). CRPS is approximated as twice the trapezoidal integral of pinball loss across that grid, so it is related to, not independent of, pinball loss.

`observed_weather_oracle` rows use realized future weather, not an operational forecast; they are explicitly non-deployable. Only check-outs have been modeled and scored so far; return-demand results are not available.
