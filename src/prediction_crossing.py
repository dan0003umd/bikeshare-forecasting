"""Validation-only: size of quantile crossing, and effect of sorting and clipping."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .metrics import score
from .splits import assert_no_locked_test_rows
from .prediction_sanity import QCOLS

for model in ("xgboost_shared_quantile", "transformer_xgbfeatures_longcontext_quantile"):
    df = pd.read_parquet(cfg.PREDICTIONS_DIR / f"{model}_validation_2025.parquet")
    assert_no_locked_test_rows(df)
    pred = df[QCOLS].to_numpy(float)
    step = np.diff(pred, axis=1)
    crossing = -step[step < 0]
    print(model)
    print("  share of adjacent steps that cross:", float((step < 0).mean()))
    print("  mean / max crossing size:", float(crossing.mean()) if crossing.size else 0.0,
          float(crossing.max()) if crossing.size else 0.0)
    for label, p in (("raw", pred), ("sorted", np.sort(pred, axis=1)),
                     ("clip+sort", np.sort(np.maximum(pred, 0.0), axis=1))):
        print(f"  {label:10s} pinball:", round(score(df.actual, p)["pinball_loss"], 5))