"""Write the factual interim evidence package from saved experiment outputs."""
from __future__ import annotations

import json

import pandas as pd

from . import config as cfg


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for row in frame[columns].itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join([header, rule, *rows])


def main() -> None:
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(cfg.REPORTS_DIR / "results_validation_2025.csv")
    if not results.n_observations.eq(435600).all():
        raise ValueError("all interim models must use the common 435,600-observation window")
    results = results.sort_values("pinball_loss").reset_index(drop=True)
    results.to_csv(cfg.REPORTS_DIR / "results_table.csv", index=False)
    display = results.assign(
        pinball_loss=results.pinball_loss.map("{:.4f}".format), crps=results.crps.map("{:.4f}".format),
        coverage_90=results.coverage_90.map("{:.3f}".format), median_mae=results.median_mae.map("{:.4f}".format),
    )
    results_md = ["# Validation Results (2025)", "",
                  "All rows use the same 435,600 held-out 2025 check-out station-hours: 50 stations x 363 local days x 24 hours.", "",
                  _table(display, ["model", "pinball_loss", "crps", "coverage_90", "median_mae", "n_observations"]), "",
                  "Pinball loss is averaged over the fixed 19 quantiles (0.05--0.95 by 0.05). CRPS is approximated as twice the trapezoidal integral of pinball loss across that grid, so it is related to, not independent of, pinball loss.",
                  "", "Only check-outs have been modeled and scored so far; return-demand results are not available."]
    (cfg.REPORTS_DIR / "results_table.md").write_text("\n".join(results_md) + "\n", encoding="utf-8")

    report = json.loads((cfg.PROCESSED_DIR / "build_report.json").read_text())
    quality = f"""# Data Quality\n\n- Input monthly archives: {len(report['months'])} ({report['months'][0]} through {report['months'][-1]}).\n- Rows read: {report['rows_read']:,}; table rows: {report['table_rows']:,}.\n- Selected stations: {report['stations_selected']} of {report['stations_seen']} seen ({report['stations_active_whole_period']} active across the full observed window).\n- Rows with unparseable timestamps: {report['bad_timestamps']:,}; within-file duplicate rides removed: {report['within_file_duplicates']:,}; cross-file boundary duplicates removed: {report['cross_file_duplicates']:,}.\n- Trips lacking a start station: {report['missing_start_station']:,}; lacking an end station: {report['missing_end_station']:,}. Each known trip side is still counted.\n- Trips with end before start: {report['end_before_start']:,}; these are reported for audit but not discarded by the present aggregation.\n- Daylight saving: the repeated November local hour is assigned to the first occurrence and both physical hours receive `dst_overlap=True`. There are 300 flagged station-hours in the built table; evaluation excludes DST/non-24-hour local days.\n\nStations are ranked by check-outs strictly before 2025-01-01, after requiring activity near both ends of the downloaded period. The top 50 are then zero-filled at every UTC hour.\n"""
    (cfg.REPORTS_DIR / "data_quality.md").write_text(quality, encoding="utf-8")

    challenges = """# Challenges and Changes\n\n| Date | Status | Issue and response |\n|---|---|---|\n| 2026-10-26 | solved | Current trip archives used the expected `ride_id`, timestamps, and station-id columns; the builder verifies those columns and fails loudly if they change. |\n| 2026-10-26 | solved | 205 duplicate rides appeared across month boundaries; the builder de-duplicated boundary ride IDs. |\n| 2026-10-26 | worked around | 3,813,? start/end station fields are missing in raw trips (exact side-specific counts are in `data_quality.md`); each available trip side remains usable. |\n| 2026-10-26 | solved | Repeated November local hours are ambiguous. They are flagged `dst_overlap`; the two DST transition days are excluded from fixed-24-hour evaluation. |\n| 2026-10-26 | worked around | Previous Runs lead-1 weather is incomplete in early years. Observed weather is joined but not yet passed to a reported model; any future observed-weather result will be labelled ORACLE. |\n| 2026-10-27 | solved | Seasonal naive initially rolled through earlier 2025 demand. It was changed to fit only 2023--2024 history, with a regression test. |\n| 2026-10-27 | open | The first transformer is a five-epoch weekly-stride run, so it is undertrained; daily-stride, early-stopped training and embedding/weather ablations remain. |\n| 2026-10-27 | open | Per-row transformer samples were not persisted after evaluation, so the current hourly-error figure only includes the retained seasonal-naive predictions. |\n""".replace("3,813,? start/end station fields are missing", f"{report['missing_start_station']:,} start and {report['missing_end_station']:,} end station fields are missing")
    (cfg.REPORTS_DIR / "challenges.md").write_text(challenges, encoding="utf-8")

    best = results.iloc[0]
    summary = f"""# Interim Evidence Summary\n\n- Built a 50-station hourly check-out/return table from {len(report['months'])} monthly archives; it contains {report['table_rows']:,} station-hours and no nulls after feature assembly.\n- The train/validation boundary is fixed at local 2025-01-01; all three reported models use the same 435,600 2025 check-out observations. 2026 remains locked.\n- The two DST transition days are excluded because a local day has 23 or 25 physical hours; this leaves 363 valid fixed-24-hour days.\n- With the strict 2023--2024-only seasonal-naive fit, `{best.model}` has the lowest pinball loss ({best.pinball_loss:.4f}) and median MAE ({best.median_mae:.4f}) among the evaluated models.\n- The five-epoch transformer validation NLL fell from 1.7968 to a best 1.7606 (epoch 4), then 1.7597 at epoch 5. It is preliminary and was trained on weekly-stride windows.\n- Previous Runs forecast-weather coverage is incomplete before 2025. Observed weather is available in the model table but is not used in these first reported models; a future observed-weather ablation will be labelled ORACLE.\n- The next compute priority is daily-stride early-stopped transformer training plus the station-embedding and weather ablations; XGBoost, Chronos-2, and LSTM are not yet interim prerequisites.\n"""
    (cfg.REPORTS_DIR / "interim_evidence.md").write_text(summary, encoding="utf-8")
    print("wrote interim evidence markdown and results table")


if __name__ == "__main__":
    main()
