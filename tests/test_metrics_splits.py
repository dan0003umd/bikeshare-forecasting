import numpy as np
import pandas as pd
import pytest

from src.metrics import crps_from_quantiles, interval_coverage_90, median_mae, pinball_loss
from src.splits import split_mask


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
