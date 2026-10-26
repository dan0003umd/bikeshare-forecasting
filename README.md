# Probabilistic Station-Level Demand Forecasting for Capital Bikeshare (MSML 612 PCS1)

Group 2: Dylan Boone, Dhanraj Nandurkar. Goal: forecast the full probability distribution of hourly
check-outs and returns at a station over the next 24 hours with a transformer, and compare it with
seasonal naive, Negative Binomial, XGBoost, LSTM and Chronos-2 baselines.

Data split (from the proposal): train 2023-2024, tune 2025, test ONCE on 2026.
Until the final report, do not evaluate anything on 2026.

## One-time setup (Windows)

1. Install Git for Windows, VS Code (with the Python and Jupyter extensions) and Python 3.11 or 3.12.
2. Open this folder in VS Code, then open a terminal (PowerShell):

```powershell
py -3.12 -m venv .venv                 # use -3.11 if that is what you installed
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
# PyTorch with GPU support: open https://pytorch.org/get-started/locally/, choose
# Windows + Pip + a CUDA version, and run the command it shows. Then:
pip install -r requirements.txt
python -m src.check_env                # shows versions and whether the GPU is visible
python -m pytest -q                    # the table builder's self-checks (no internet needed)
```

3. In VS Code choose the interpreter `.venv` (Ctrl+Shift+P, "Python: Select Interpreter").

## Build the data (about 10-20 minutes of downloading)

```powershell
python -m src.download_trips           # monthly trip zips, 2023-01 to last month -> data/raw/trips
python -m src.download_station_info    # station names, coordinates, dock capacity -> data/processed
python -m src.build_hourly             # hourly check-outs/returns for the busiest stations
```

`build_hourly` writes `data/processed/station_hourly.parquet`, `station_summary.csv` and
`build_report.json`. The JSON lists every row dropped or fixed (no-station trips, duplicates, bad
timestamps); copy those numbers into the "Challenges" part of the reports.

## Folder layout

```
src/        all code (config.py holds the settings)
tests/      small self-checks with made-up trip files
data/       raw downloads and processed tables (NOT committed to git)
reports/    figures and report drafts
```

## Working together

- `git pull --rebase` before you start, commit small and often, push when a step works.
- Do not commit data, model files or `.venv` (the .gitignore already blocks them). Share the
  processed Parquet file through the shared Drive folder.
- Add a line to CONTRIBUTIONS.md whenever you finish something. The interim and final reports
  need an "individual contributions" section.

## Known assumptions to confirm on first run

- The trip file URL pattern and the column names (`ride_id`, `started_at`, `ended_at`,
  `start_station_id`, `end_station_id`) are what current files are reported to use. If a file
  differs, the script stops and prints the columns it found.
- GBFS station ids may differ from trip-file station ids; compare the columns in `station_info.csv`.
