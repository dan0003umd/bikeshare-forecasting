# Interim Evidence Summary

- Built a 50-station hourly check-out/return table from 45 monthly archives; it contains 1,642,750 station-hours and no nulls after feature assembly.
- The train/validation boundary is fixed at local 2025-01-01; all three reported models use the same 435,600 2025 check-out observations. 2026 remains locked.
- The two DST transition days are excluded because a local day has 23 or 25 physical hours; this leaves 363 valid fixed-24-hour days.
- With the strict 2023--2024-only seasonal-naive fit, `hf_transformer_negative_binomial` has the lowest pinball loss (0.6349) and median MAE (1.6942) among the evaluated models.
- The five-epoch transformer validation NLL fell from 1.7968 to a best 1.7606 (epoch 4), then 1.7597 at epoch 5. It is preliminary and was trained on weekly-stride windows.
- Previous Runs forecast-weather coverage is incomplete before 2025. Observed weather is available in the model table but is not used in these first reported models; a future observed-weather ablation will be labelled ORACLE.
- The next compute priority is daily-stride early-stopped transformer training plus the station-embedding and weather ablations; XGBoost, Chronos-2, and LSTM are not yet interim prerequisites.
