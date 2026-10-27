# Split Integrity and Locked Test Data

- Station selection uses only rows before `2025-01-01`; this is stricter than the 2026 boundary.
- The development `model_table.parquet` contains only timestamps before local `2026-01-01`. `327,550` local-2026 rows are physically stored in the ignored locked file `data/processed/locked_test_2026/model_table.parquet`.
- Every training, tuning, scoring, and figure entrypoint loads this table through `load_development_model_table`, which applies a Parquet predicate and raises if a local-2026 row is present.
- Transformer observed-weather standardization computes mean and standard deviation from the 2023--2024 training mask only. Calendar features need no fitted scaler; XGBoost uses unscaled lag/calendar inputs; Chronos-2 is zero-shot and fits no project-level normalization.
- `tests/test_metrics_splits.py` verifies both the predicate/guard and that each development entrypoint routes through the guarded loader.
