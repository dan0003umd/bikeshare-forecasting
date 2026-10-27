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
