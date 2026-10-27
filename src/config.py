"""Project-wide settings. Change values here, not inside the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_TRIPS_DIR = ROOT / "data" / "raw" / "trips"
PROCESSED_DIR = ROOT / "data" / "processed"
EXTERNAL_DIR = ROOT / "data" / "external"
REPORTS_DIR = ROOT / "reports" / "interim_evidence"
CHECKPOINTS_DIR = ROOT / "checkpoints"
PREDICTIONS_DIR = ROOT / "data" / "predictions"
MODEL_TABLE_PATH = PROCESSED_DIR / "model_table.parquet"
LOCKED_TEST_DIR = PROCESSED_DIR / "locked_test_2026"
LOCKED_TEST_MODEL_TABLE_PATH = LOCKED_TEST_DIR / "model_table.parquet"

# Trip files used: monthly files from this (year, month) up to the latest published month.
START = (2023, 1)
TRIP_URL = "https://s3.amazonaws.com/capitalbikeshare-data/{ym}-capitalbikeshare-tripdata.zip"
GBFS_STATION_INFO_URL = "https://gbfs.lyft.com/gbfs/1.1/dca-cabi/en/station_information.json"

# Trip timestamps in the files are local (Eastern) wall-clock time.
LOCAL_TZ = "America/New_York"

# Stations are ranked by check-outs BEFORE this date (training years only), so the
# 2025 tuning data and the 2026 test data never influence which stations we pick.
SELECTION_END = "2025-01-01"
N_STATIONS = 50

# Chronological experiment split.  Do not change these in individual scripts.
TRAIN_START = "2023-01-01"
TRAIN_END = "2025-01-01"
VALID_START = "2025-01-01"
VALID_END = "2026-01-01"
TEST_START = "2026-01-01"
SEED = 42

# The single quantile grid used by every probabilistic model and report.
QUANTILES = tuple(round(q / 100, 2) for q in range(5, 100, 5))

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
DC_LATITUDE = 38.9072
DC_LONGITUDE = -77.0369
