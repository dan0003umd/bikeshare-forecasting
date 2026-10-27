"""One-time migration: physically separate local-2026 model rows from development data."""
from __future__ import annotations

import pandas as pd

from . import config as cfg
from .splits import assert_no_locked_test_rows, development_end_utc


def main() -> None:
    source = cfg.MODEL_TABLE_PATH
    table = pd.read_parquet(source)
    timestamps = pd.to_datetime(table.ts_utc, utc=True)
    development = table.loc[timestamps < development_end_utc()].copy()
    locked = table.loc[timestamps >= development_end_utc()].copy()
    if locked.empty:
        raise RuntimeError("no locked local-2026 rows found; refusing to overwrite the model table")
    assert_no_locked_test_rows(development)
    cfg.LOCKED_TEST_DIR.mkdir(parents=True, exist_ok=True)
    temporary = cfg.PROCESSED_DIR / "model_table_development.tmp.parquet"
    development.to_parquet(temporary, index=False)
    locked.to_parquet(cfg.LOCKED_TEST_MODEL_TABLE_PATH, index=False)
    temporary.replace(source)
    print(f"development rows: {len(development):,}; locked local-2026 rows: {len(locked):,}")
    print(f"locked file: {cfg.LOCKED_TEST_MODEL_TABLE_PATH}")


if __name__ == "__main__":
    main()
