# Validation Results (2025)

All rows use the same 435,600 held-out 2025 check-out station-hours: 50 stations x 363 local days x 24 hours.

| model | pinball_loss | crps | coverage_90 | median_mae | n_observations |
|---|---|---|---|---|---|
| hf_transformer_negative_binomial | 0.6349 | 1.1826 | 0.932 | 1.6942 | 435600 |
| seasonal_naive_8week | 0.7492 | 1.3884 | 0.761 | 1.9293 | 435600 |
| per_station_negative_binomial | 0.8157 | 1.5178 | 0.960 | 2.1441 | 435600 |

Pinball loss is averaged over the fixed 19 quantiles (0.05--0.95 by 0.05). CRPS is approximated as twice the trapezoidal integral of pinball loss across that grid, so it is related to, not independent of, pinball loss.

Only check-outs have been modeled and scored so far; return-demand results are not available.
