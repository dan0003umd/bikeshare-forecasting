"""Build one report-ready Markdown bundle and plainly named figure copies."""
from __future__ import annotations


import shutil

import pandas as pd

from . import config as cfg



DOCUMENTS = [
    "results_table.md",
    "chronos_subsample_results.md",
    "event_day_results.md",
    "calibration_diagnostics.md",
    "split_integrity.md",
    "interim_evidence.md",
    "challenges.md",
    "data_quality.md",
    "weather_probe.md",
    "forecast_history.md",
    "reproduce.md",
]

CSV_TABLES = {
    "chronos_50stations_363days_validation_2025.csv": "Chronos-2 and all full-window models on the full 435,600-row validation window",
    "chronos_full_window_bootstrap.csv": "Paired day-bootstrap comparisons involving full-window Chronos-2",
    "prediction_sanity_2025.csv": "Prediction sanity checks: zero share, negative and crossing quantiles, non-negativity clip sensitivity",
    "seed_check/training_seed_check.csv": "Second training seed (43) check against the seed-42 daily embedded transformer",
}


FIGURES = {
    "eda_hour_of_week.png": "demand_by_hour.png",
    "transformer_loss.png": "training_and_validation_loss.png",
    "seasonal_naive_sample_forecasts.png": "sample_forecasts.png",
    "calibration_coverage.png": "calibration.png",
    "error_by_hour_available_predictions.png": "error_by_hour.png",
    "calibration_reliability_2025.png": "reliability_by_quantile.png",
}

UPDATES = [
    "## Updates that supersede earlier statements in this bundle", "",
    "- Chronos-2 has now also been scored on the full 435,600-row validation window: pinball 0.5987, CRPS 1.1159, "
    "coverage 0.867, median MAE 1.6040. Statements in other sections that Chronos-2 was evaluated only on the "
    "14,400-row slice describe the earlier run; the full-window table below supersedes the slice comparison.",
    "- A second training seed (43) of the daily embedded no-weather transformer, trained with the same procedure, "
    "scored pinball 0.5735 and median MAE 1.5382 (seed 42: 0.5755 and 1.5426). This is a different check from the "
    "sampling-seed audit below, which changed only the generation seed of the existing checkpoint.",
    "- Event-day, calibration and prediction-sanity analyses were added after the earlier sections were written; "
    "see their sections below.", "",
]


def format_cell(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_cell(value) for value in row) + " |")
    return "\n".join(lines)



def main() -> None:
    sections = ["# Consolidated Interim Evidence", "", *UPDATES,
                "## Seed-43 sampling audit", "",
                "The seed-43 audit did **not** retrain the transformer. It loaded the existing daily embedded/no-weather checkpoint and changed only the stochastic generation seed during the 100-sample quantile evaluation.", ""]
    for name in DOCUMENTS:
        path = cfg.REPORTS_DIR / name
        if not path.exists():
            raise FileNotFoundError(path)
        sections.extend([f"## Source: `reports/interim_evidence/{name}`", "", path.read_text(encoding="utf-8").rstrip(), ""])
    for name, title in CSV_TABLES.items():
        path = cfg.REPORTS_DIR / name
        if not path.exists():
            raise FileNotFoundError(path)
        sections.extend([f"## {title}", "", f"Source: `reports/interim_evidence/{name}`", "",
                         markdown_table(pd.read_csv(path)), ""])
    handoff = cfg.ROOT / "reports" / "handoff_one_pager.md"
    sections.extend(["## Source: `reports/handoff_one_pager.md`", "", handoff.read_text(encoding="utf-8").rstrip(), ""])
    (cfg.REPORTS_DIR / "all_evidence.md").write_text("\n".join(sections), encoding="utf-8")


    destination = cfg.REPORTS_DIR / "figures_for_report"
    destination.mkdir(parents=True, exist_ok=True)
    for source_name, plain_name in FIGURES.items():
        source = cfg.REPORTS_DIR / source_name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination / plain_name)
    print(f"wrote {cfg.REPORTS_DIR / 'all_evidence.md'} and {len(FIGURES)} figure copies")



if __name__ == "__main__":
    main()