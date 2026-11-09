import numpy as np
import pandas as pd

from src.calibration_diagnostics import apply_conformal_adjustment, fit_conformal_adjustment


def test_single_conformal_adjustment_widens_and_preserves_quantile_order():
    quantiles = {f"q{q:02d}": np.array([float(q) / 10, float(q) / 10]) for q in range(5, 100, 5)}
    rows = pd.DataFrame({"actual": [0.0, 20.0], **quantiles})
    adjustment = fit_conformal_adjustment(rows)
    adjusted = apply_conformal_adjustment(rows, adjustment)
    assert adjustment >= 0
    assert np.all(np.diff(adjusted, axis=1) >= 0)
    assert np.allclose(adjusted[:, 9], rows.q50)
