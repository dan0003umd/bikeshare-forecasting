import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.metrics import crps_from_quantiles, interval_coverage_90, median_mae, pinball_loss
from src.splits import assert_no_locked_test_rows, development_end_utc, load_development_model_table, split_mask


def test_metrics_are_zero_for_exact_quantiles():
    y = np.array([2.0, 4.0])
    p = np.repeat(y[:, None], 19, axis=1)
    assert pinball_loss(y, p) == 0
    assert crps_from_quantiles(y, p) == 0
    assert interval_coverage_90(y, p) == 1
    assert median_mae(y, p) == 0


def test_split_gates_test_year():
    # Split boundaries are local midnight: EST is 05:00 UTC in January.
    ts = pd.Series(pd.to_datetime(["2024-01-01T05:00Z", "2025-01-01T05:00Z", "2026-01-01T05:00Z"], utc=True))
    assert split_mask(ts, "train").tolist() == [True, False, False]
    assert split_mask(ts, "validation").tolist() == [False, True, False]
    with pytest.raises(PermissionError, match="locked"):
        split_mask(ts, "test")
    assert split_mask(ts, "test", final=True).tolist() == [False, False, True]


def test_development_guard_rejects_local_2026_rows():
    table = pd.DataFrame({"ts_utc": pd.to_datetime(["2025-12-31T04:00Z", "2026-01-01T05:00Z"], utc=True)})
    with pytest.raises(AssertionError, match="locked"):
        assert_no_locked_test_rows(table)


def test_development_loader_pushes_down_test_boundary(monkeypatch):
    seen = {}

    def fake_read_parquet(path, columns=None, filters=None):
        seen["filters"] = filters
        return pd.DataFrame({"ts_utc": pd.to_datetime(["2026-01-01T04:00Z"], utc=True)})

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    loaded = load_development_model_table()
    assert len(loaded) == 1  # 2025-12-31 23:00 local is not a test target.
    assert seen["filters"] == [("ts_utc", "<", development_end_utc())]


def test_every_training_tuning_or_scoring_entrypoint_uses_locked_loader():
    source_root = Path(__file__).parents[1] / "src"
    for filename in ("baselines.py", "train_transformer.py", "evaluate_transformer.py",
                     "xgboost_baseline.py", "chronos_subsample.py", "make_figures.py"):
        assert "load_development_model_table" in (source_root / filename).read_text(encoding="utf-8")
