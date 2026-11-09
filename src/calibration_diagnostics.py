"""Validation-only reliability and held-out post-hoc calibration diagnostics.

Calibration is fitted on local 2025-01-01 through 2025-06-30 and scored only
on local 2025-07-01 through 2025-12-31.  It is not a test-set result and does
not retrain or overwrite any forecasting model.
"""
from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg
from .metrics import score
from .splits import assert_no_locked_test_rows


MODELS = (
    "xgboost_shared_quantile",
    "hf_transformer_daily_embed_noweather",
    "transformer_xgbfeatures_longcontext_nb",
    "transformer_xgbfeatures_longcontext_quantile",
)
Q = np.asarray(cfg.QUANTILES)
QCOLS = [f"q{int(q * 100):02d}" for q in Q]
CALIBRATION_END = pd.Timestamp("2025-07-01")


def local_dates(rows: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(rows.ts_utc, utc=True).dt.tz_convert(cfg.LOCAL_TZ).dt.tz_localize(None).dt.normalize()


def reliability(rows: pd.DataFrame) -> pd.DataFrame:
    predictions = rows[QCOLS].to_numpy(float)
    actual = rows.actual.to_numpy(float)
    return pd.DataFrame({"nominal_quantile": Q, "empirical_coverage": (actual[:, None] <= predictions).mean(axis=0)})


def fit_conformal_adjustment(rows: pd.DataFrame) -> float:
    """One additive central-interval adjustment fitted on the first half only."""
    q05, q95 = rows.q05.to_numpy(float), rows.q95.to_numpy(float)
    actual = rows.actual.to_numpy(float)
    nonconformity = np.maximum.reduce([q05 - actual, actual - q95, np.zeros(len(rows))])
    # ``higher`` gives the finite-sample conservative central-90% choice.
    rank = min(1.0, np.ceil((len(nonconformity) + 1) * 0.90) / len(nonconformity))
    return float(np.quantile(nonconformity, rank, method="higher"))


def apply_conformal_adjustment(rows: pd.DataFrame, adjustment: float) -> np.ndarray:
    """Widen all quantiles linearly around q50 with one non-negative adjustment."""
    prediction = rows[QCOLS].to_numpy(float)
    direction = (Q - 0.50) / 0.45
    # Do not clip or sort: either operation would be a second post-hoc fit and
    # could change forecasts even when the fitted adjustment is exactly zero.
    return prediction + adjustment * direction[None, :]


def _metrics(rows: pd.DataFrame, prediction: np.ndarray) -> dict[str, float | int]:
    return {**score(rows.actual, prediction), "n_observations": int(len(rows)), "days": int(rows.local_date.nunique())}


def run() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    reliability_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    reliability_frames: dict[str, pd.DataFrame] = {}
    for model in MODELS:
        path = cfg.PREDICTIONS_DIR / f"{model}_validation_2025.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing saved predictions: {path}")
        rows = pd.read_parquet(path)
        assert_no_locked_test_rows(rows)
        rows = rows.copy()
        rows["local_date"] = local_dates(rows)
        if len(rows) != 435_600:
            raise AssertionError(f"{model} has {len(rows):,}, expected 435,600 rows")
        first = rows.loc[rows.local_date < CALIBRATION_END]
        second = rows.loc[rows.local_date >= CALIBRATION_END]
        if first.empty or second.empty:
            raise AssertionError("calibration split is empty")
        rel = reliability(rows)
        reliability_frames[model] = rel
        reliability_rows.extend({"record_type": "reliability_full_validation", "model": model, **row}
                                for row in rel.to_dict("records"))
        adjustment = fit_conformal_adjustment(first)
        raw_second = rows[QCOLS].to_numpy(float)[rows.local_date.ge(CALIBRATION_END).to_numpy()]
        adjusted_second = apply_conformal_adjustment(second, adjustment)
        for label, prediction in (("uncalibrated_second_half", raw_second), ("conformal_second_half", adjusted_second)):
            calibration_rows.append({
                "record_type": "held_out_calibration", "model": model, "variant": label,
                "fit_start_local": "2025-01-01", "fit_end_local_exclusive": "2025-07-01",
                "score_start_local": "2025-07-01", "score_end_local_exclusive": "2026-01-01",
                "conformal_adjustment": adjustment, **_metrics(second, prediction),
            })
    return pd.DataFrame(reliability_rows), pd.DataFrame(calibration_rows), reliability_frames


def _table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join(["---"] * len(columns)) + "|"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *body])


def write_plot(frames: dict[str, pd.DataFrame]) -> None:
    plt.figure(figsize=(6.4, 5.0))
    for model, frame in frames.items():
        plt.plot(frame.nominal_quantile, frame.empirical_coverage, marker="o", markersize=3, linewidth=1.4,
                 label=model.replace("hf_transformer_", "").replace("transformer_xgbfeatures_", "long_"))
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="ideal")
    plt.xlabel("Nominal quantile")
    plt.ylabel("Empirical P(actual ≤ predicted quantile)")
    plt.title("2025 validation reliability (diagnostic only)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(cfg.REPORTS_DIR / "calibration_reliability_2025.png", dpi=160)
    plt.close()


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    start = time.perf_counter()
    reliability_rows, calibration_rows, frames = run()
    elapsed = (time.perf_counter() - start) / 60
    write_plot(frames)
    output = pd.concat([reliability_rows, calibration_rows], ignore_index=True, sort=False)
    csv_path = cfg.REPORTS_DIR / "calibration_diagnostics.csv"
    md_path = cfg.REPORTS_DIR / "calibration_diagnostics.md"
    output.to_csv(csv_path, index=False)
    held_out = calibration_rows[["model", "variant", "conformal_adjustment", "pinball_loss", "crps", "coverage_90",
                                 "median_mae", "n_observations", "days"]]
    md_path.write_text(
        "# Calibration diagnostics\n\n"
        "This is a validation-only diagnostic, not a test result. Reliability uses all saved 2025 validation rows. "
        "Post-hoc calibration fits one non-negative additive conformal central-interval adjustment on local "
        "2025-01-01 through 2025-06-30 only, then scores the unchanged and adjusted forecasts on local "
        "2025-07-01 through 2025-12-31 only; no row is used to both fit and score the adjustment. The adjustment "
        "widens quantiles around q50. It does not clip or sort predictions, so the calibrated forecast differs only by "
        "that one fitted adjustment.\n\n"
        "## Held-out second-half scores\n\n" + _table(held_out) + "\n\n"
        "## Full-2025 reliability by quantile\n\n" + _table(reliability_rows) + "\n\n"
        "Figure: `calibration_reliability_2025.png`.\n\n"
        "## Reproduction\n\n`python -m src.calibration_diagnostics`\n\n"
        f"Wall-clock time: {elapsed:.2f} minutes.\n",
        encoding="utf-8",
    )
    print(held_out.to_string(index=False))
    print(f"wrote {csv_path}, {md_path}, and calibration_reliability_2025.png; wall-clock {elapsed:.2f} minutes")


if __name__ == "__main__":
    main()
