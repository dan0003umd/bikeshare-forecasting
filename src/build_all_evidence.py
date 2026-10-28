"""Build one report-ready Markdown bundle and plainly named figure copies."""
from __future__ import annotations

import shutil

from . import config as cfg


DOCUMENTS = [
    "results_table.md",
    "chronos_subsample_results.md",
    "split_integrity.md",
    "interim_evidence.md",
    "challenges.md",
    "data_quality.md",
    "weather_probe.md",
    "forecast_history.md",
    "reproduce.md",
]

FIGURES = {
    "eda_hour_of_week.png": "demand_by_hour.png",
    "transformer_loss.png": "training_and_validation_loss.png",
    "seasonal_naive_sample_forecasts.png": "sample_forecasts.png",
    "calibration_coverage.png": "calibration.png",
    "error_by_hour_available_predictions.png": "error_by_hour.png",
}


def main() -> None:
    sections = ["# Consolidated Interim Evidence", "",
                "## Seed-43 sampling audit", "",
                "The seed-43 audit did **not** retrain the transformer. It loaded the existing daily embedded/no-weather checkpoint and changed only the stochastic generation seed during the 100-sample quantile evaluation.", ""]
    for name in DOCUMENTS:
        path = cfg.REPORTS_DIR / name
        if not path.exists():
            raise FileNotFoundError(path)
        sections.extend([f"## Source: `reports/interim_evidence/{name}`", "", path.read_text(encoding="utf-8").rstrip(), ""])
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
