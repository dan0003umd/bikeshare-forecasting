# Interim Evidence Summary

- Built a 50-station hourly check-out/return table from 45 monthly archives; it contains 1,642,750 station-hours and no nulls after feature assembly.
- The train/validation boundary is fixed at local 2025-01-01; all seven reported models use the same 435,600 2025 check-out observations. 2026 remains locked.
- The two DST transition days are excluded because a local day has 23 or 25 physical hours; this leaves 363 valid fixed-24-hour days.
- The restored causal rolling seasonal naive has pinball loss 0.6659 and median MAE 1.7306. The daily-stride transformer with station embeddings and no weather improves these to 0.5755 and 1.5426.
- The observed-weather ORACLE reaches 0.5653 pinball loss and 1.5108 median MAE, but it uses realized future weather and is not deployable. Previous Runs forecast-weather coverage is incomplete before 2025.
- Daily-stride transformer ablations used seed 42 and patience-3 early stopping. XGBoost, Chronos-2, and LSTM have not yet been evaluated.
