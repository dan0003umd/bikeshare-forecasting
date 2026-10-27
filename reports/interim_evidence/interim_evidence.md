# Interim Evidence Summary

- Built a 50-station hourly check-out/return table from 45 monthly archives; it contains 1,642,750 station-hours and no nulls after feature assembly.
- The train/validation boundary is fixed at local 2025-01-01; all full-window numbers use the same 435,600 2025 check-out observations. This period was also used for early stopping and tuning, so these are not test results. The 2026 test split remains locked.
- The two DST transition days are excluded because a local day has 23 or 25 physical hours; this leaves 363 valid fixed-24-hour days.
- The restored causal rolling seasonal naive has pinball loss 0.6659 and median MAE 1.7306. The daily-stride transformer with station embeddings and no weather improves these to 0.5755 and 1.5426.
- The observed-weather ORACLE reaches 0.5653 pinball loss and 1.5108 median MAE, but it uses realized future weather and is not deployable. Previous Runs forecast-weather coverage is incomplete before 2025.
- Daily-stride transformer ablations used seed 42 and patience-3 early stopping. XGBoost and Chronos-2 have now been evaluated; the LSTM is intentionally deferred to Dylan.
- The shared XGBoost quantile baseline, fit only on 2023--2024 data, has pinball loss 0.5644, CRPS 1.0520, and median MAE 1.5148; its fit took 3.19 minutes.
- Chronos-2 zero-shot was evaluated only on the fixed 14,400-row 2025 validation subsample; its pinball loss was 0.5329.
- No saved scored target falls in the local 2026 test period (the final local-2025 evening has a 2026 UTC timestamp). Historical baseline/transformer scripts did load an assembled table containing later archives before masking to 2025, so the project cannot claim that no 2026 row was ever read; the new XGBoost and Chronos loaders use pre-2026 Parquet predicates.
