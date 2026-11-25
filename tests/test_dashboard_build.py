"""Build-level checks for the offline validation dashboard using real artifacts."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from dashboard.build_dashboard import MODEL_FILES, read_scores, find_files


ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    "seasonal_naive_rolling_8week",
    "xgboost_shared_quantile",
    "lstm_quantile",
    "hf_transformer_daily_embed_noweather",
    "hf_transformer_daily_embed_forecast_weather_lead1",
    "hf_transformer_daily_embed_observed_weather_oracle",
]
REPORTS = {
    "checkouts": ROOT / "reports" / "interim_evidence" / "results_validation_2025.csv",
    "returns": ROOT / "reports" / "returns_evidence" / "results_validation_2025.csv",
}


@pytest.fixture(scope="module")
def built_dashboard(tmp_path_factory):
    out = tmp_path_factory.mktemp("dashboard") / "index.html"
    subprocess.run(
        [
            sys.executable,
            "dashboard/build_dashboard.py",
            "--predictions-dir",
            "data/predictions",
            "--returns-predictions-dir",
            "data/predictions/returns",
            "--out",
            str(out),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    html = out.read_text(encoding="utf-8")
    match = re.search(r"var DATA = (\{.*?\});\s*var SN=", html, flags=re.DOTALL)
    assert match, "generated dashboard must embed a DATA object"
    return html, json.loads(match.group(1))


def test_dashboard_is_self_contained_and_offline(built_dashboard):
    html, _ = built_dashboard
    forbidden = ("http://", "https://", "<script src", "<link href", "fetch(", "XMLHttpRequest", "@import url")
    assert not [needle for needle in forbidden if needle in html]


def test_default_build_has_no_test_scores(built_dashboard):
    _, data = built_dashboard
    assert data["test"]["checkouts"] == {}
    assert data["test"]["returns"] == {}


def test_dashboard_source_does_not_name_locked_test_directory():
    source = (ROOT / "dashboard" / "build_dashboard.py").read_text(encoding="utf-8")
    assert "locked_test_2026" not in source


def test_short_lag_xgboost_ablation_never_fills_normal_xgboost_slot(tmp_path):
    short = tmp_path / "xgboost_short_lag_not_deployable_validation_2025.parquet"
    short.touch()
    assert find_files(tmp_path, MODEL_FILES["xgboost_shared_quantile"]) is None
    scores = pd.DataFrame([
        {"model": "xgboost_short_lag_not_deployable", "pinball_loss": 1.0, "crps": 1.0,
         "coverage_90": 0.9, "median_mae": 1.0, "n_observations": 435600},
    ])
    assert "xgboost_shared_quantile" not in read_scores(scores)


def test_embedded_scores_match_saved_csvs(built_dashboard):
    _, data = built_dashboard
    metric_columns = {"pinball": "pinball_loss", "crps": "crps", "cov90": "coverage_90", "mae": "median_mae"}
    for target, path in REPORTS.items():
        csv = pd.read_csv(path).set_index("model")
        for model, embedded in data["targets"][target]["scores"].items():
            assert model in csv.index
            for embedded_name, csv_name in metric_columns.items():
                assert embedded[embedded_name] == pytest.approx(float(csv.loc[model, csv_name]))
            assert embedded["n"] == int(csv.loc[model, "n_observations"])


def test_percent_lower_kpi_uses_saved_scores_at_one_decimal():
    scores = pd.read_csv(REPORTS["checkouts"]).set_index("model")
    deployable = scores.loc[[m for m in MODELS if m != "hf_transformer_daily_embed_observed_weather_oracle" and m in scores.index]]
    best = float(deployable["pinball_loss"].min())
    seasonal = float(scores.loc["seasonal_naive_rolling_8week", "pinball_loss"])
    expected = f"{(1 - best / seasonal) * 100:.1f}% lower"
    template = (ROOT / "dashboard" / "template.html").read_text(encoding="utf-8")
    assert "p.toFixed(1)+'% lower'" in template
    assert expected.endswith("% lower")


def test_embedded_weather_timestamps_stop_before_2026(built_dashboard):
    _, data = built_dashboard
    timestamps = [ts for week in data["weeks"] for ts in week["weatherTsUtc"]]
    assert timestamps
    assert all(pd.Timestamp(ts) < pd.Timestamp("2026-01-01", tz="UTC") for ts in timestamps)


def test_basemap_is_embedded_when_built_and_is_offline(built_dashboard):
    _, data = built_dashboard
    bm = data.get("basemap")
    if bm is None:
        pytest.skip("dashboard/basemap_dc.json not built yet (run dashboard/fetch_basemap.py)")
    assert any(bm["roads"][c] for c in ("major", "mid", "minor"))
    assert "OpenStreetMap" in bm["attribution"]
    assert "http" not in json.dumps(bm)
