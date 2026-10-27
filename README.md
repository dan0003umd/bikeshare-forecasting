# Probabilistic Station-Level Demand Forecasting for Capital Bikeshare

**Course:** MSML 612 PCS1 (Deep Learning), Fall 2026 | **Group 2:** Dylan Boone, Dhanraj Nandurkar
**Repository:** https://github.com/dan0003umd/bikeshare-forecasting | **Last updated:** Oct 27, 2026

## 1. What this project does

Capital Bikeshare staff move bikes between stations by truck, but those moves are not recorded, so the number
of bikes at a station reflects hidden rebalancing. We therefore forecast **demand** (check-outs and returns per
station per hour), which the trip history records directly.

We build a transformer that forecasts the **full probability distribution** of hourly demand at each station
over the next 24 hours, using trip history, weather and calendar data. The output is a Negative Binomial
distribution (a standard choice for count data), from which quantiles are derived. Like a weather forecast
giving the chance of rain, the model gives the probability that demand exceeds a threshold.

Main questions
1. Does a transformer beat standard baselines on probabilistic accuracy?
2. Does one model trained across many stations (with learned station embeddings) beat independent per-station models?
3. How much do weather forecasts help? (no weather vs. observed vs. 1-day-ahead forecast weather)
4. How do forecasts behave on event days (Nationals home games, Cherry Blossom Festival)?

Known limitation: observed check-outs understate true demand when a station is empty (censoring). We evaluate
high-capacity stations that rarely run empty separately to gauge this effect.

## 2. Project status

Legend: [x] done and checked, [~] code written, not yet run on real data, [ ] not started.

| Area | Status | Notes |
|---|---|---|
| Proposal | [x] | Submitted, 10/10 |
| Repository, environment, GPU check | [x] | Python 3.11 venv, PyTorch CUDA build, `check_env` shows the GPU |
| Trip download script | [x] | `src/download_trips.py`; 45 archives through 2026-09 are locally available |
| Station info download script | [x] | `src/download_station_info.py` |
| Hourly station table builder | [x] | `src/build_hourly.py`; 50-station table and data-quality audit built |
| Weather (observed + 1-day-ahead forecast) | [~] | Observed-weather Oracle completed; Previous Runs forecast-weather coverage is incomplete before 2025 |
| Calendar features (holidays, events) | [x] | Local calendar, US holiday, and event flags are in `model_table.parquet` |
| Metrics (pinball, CRPS, coverage, MAE) | [x] | One shared implementation for all models |
| Baselines: seasonal naive, Negative Binomial, XGBoost, LSTM | [~] | Both seasonal variants, NB, and shared 19-quantile XGBoost complete; LSTM intentionally deferred to Dylan |
| Transformer (adapted Hugging Face Time Series Transformer) | [x] | Daily-stride embedding/no-embedding and observed-weather Oracle ablations complete |
| Chronos-2 zero-shot baseline | [x] | Completed on fixed 10-station x 60-day 2025 validation subsample |
| Ablations and event-day analysis | [~] | Embedding/weather ablations complete; event-day analysis remains |
| Interim report (due Oct 30) | [~] | Evidence package and reproducibility commands are present; narrative submission remains |
| Final report | [ ] | |

Update this table whenever something changes, and log the work in `CONTRIBUTIONS.md`.

## 3. Data sources

| Data | Source | Used for |
|---|---|---|
| Trip history | Monthly zips: `https://s3.amazonaws.com/capitalbikeshare-data/YYYYMM-capitalbikeshare-tripdata.zip` (overview: https://capitalbikeshare.com/system-data) | Check-outs and returns per station per hour |
| Station info | GBFS feed: `https://gbfs.lyft.com/gbfs/1.1/dca-cabi/en/station_information.json` | Names, coordinates, dock capacity |
| Observed weather | Open-Meteo Historical Weather API (https://open-meteo.com/en/docs/historical-weather-api) | Hourly temperature, precipitation, wind for DC |
| Forecast weather | Open-Meteo Previous Runs API (https://open-meteo.com/en/docs/previous-runs-api) | Forecast as issued one day ahead |
| Holidays | Python `holidays` package | Holiday flags |
| Events | Hand-built CSV | Nationals home games, Cherry Blossom Festival |

Use the Previous Runs API, not the Historical Forecast API, for forecast weather. The latter stitches together
the first hours of each model run, so it tracks actual conditions too closely to count as a real forecast.
Check early with one test call that the Previous Runs API really covers January 2023 onward.

The public data page still lists an older column layout. Current trip files are reported to use `ride_id`,
`rideable_type`, `started_at`, `ended_at`, start and end station name and id, coordinates and `member_casual`.
`build_hourly.py` checks the needed columns and stops with a clear message listing the real columns if one is missing.

## 4. How the data is processed

`python -m src.build_hourly` turns the monthly trip files into one table: **one row per selected station per hour**.

1. Each monthly file is read separately (the laptop only has 8 GB of RAM) and reduced to hourly counts.
2. Check-outs are counted from the start station and start time. Returns are counted from the end station and
   end time. A trip with only one known station still counts on that side.
3. Trips with an unreadable timestamp are dropped. Repeated rows are removed, including a trip that appears in
   two neighbouring monthly files (only trips near a month boundary can do this).
4. Trip times in the files are local (Eastern) wall-clock time. They are converted to **UTC**. The one repeated
   hour each November is ambiguous: its trips go to the first occurrence, and both physical hours get
   `dst_overlap = True` so they can be masked in evaluation.
5. Stations are chosen from those active at the start and at the end of the period, ranked by check-outs
   **before 2025-01-01** (training years only), keeping the top `N_STATIONS` (default 50). Zero-filling therefore
   never invents demand for a closed or not-yet-opened station.
6. Every selected station is given a row for every hour (zero where there were no trips).

Output files in `data/processed/` (not committed to git):

| File | Contents |
|---|---|
| `station_hourly.parquet` | `station_id, ts_utc, checkouts, returns, net_outflow, dst_overlap` |
| `station_summary.csv` | Selected stations with totals and the longest idle run (hours) |
| `build_report.json` | Counts of rows read, duplicates, bad timestamps, trips without a station, window dates |

Planned next: attach weather (joined on UTC timestamp only; observed and forecast in separate columns),
calendar flags (hour, weekday and month from local time; holidays and events joined on date), and station
capacity and coordinates (joined on station id; check which GBFS field matches the trip files' station ids).

## 5. Experiment design

- **Split (chronological):** train 2023-2024, tune 2025, test once on 2026 (January 2026 through the latest
  available month). Do not evaluate anything on 2026 before the final report.
- **Task:** forecast the next 24 hours of hourly check-outs and returns, from several forecast start times.
- **Metrics:** pinball (quantile) loss, CRPS, interval coverage (for example, do 90% intervals contain the true
  value about 90% of the time), and MAE of the median forecast. All models are scored by the same code on the
  same windows.
- **Baselines:** seasonal naive, per-station Negative Binomial, XGBoost, LSTM; Chronos-2 (`amazon/chronos-2`,
  zero-shot) as the foundation-model reference.
- **Transformer:** Hugging Face `TimeSeriesTransformerForPrediction` with a Negative Binomial output and
  learned station embeddings, trained as a single model across all selected stations; compared with and
  without the embeddings.

## 6. Reproducibility guide

**Tested setup:** Windows 10/11, Python 3.11.9, numpy 2.4.6, pandas 3.1.0, pyarrow 26.0.0, scikit-learn 1.9.1,
statsmodels 0.15.0, xgboost 3.2.0, transformers 5.20.2, torch 2.14.1+cu130, NVIDIA GTX 1650 (4 GB), 8 GB RAM.
macOS and Linux should work for the data steps; use the matching PyTorch command from pytorch.org.

**Step 1: get the code**

    git clone https://github.com/dan0003umd/bikeshare-forecasting.git
    cd bikeshare-forecasting

**Step 2: create the environment** (PowerShell; on macOS/Linux use `python3 -m venv .venv` and `source .venv/bin/activate`)

    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
    python -m pip install --upgrade pip

**Step 3: install PyTorch first**, using the command from https://pytorch.org/get-started/locally/ for your
system. With an NVIDIA GPU, run `nvidia-smi`, read "CUDA Version", and pick a build that is not newer than it.
On the tested laptop (CUDA 13.1 driver) this was:

    pip install torch --index-url https://download.pytorch.org/whl/cu130

**Step 4: install the other packages and check**

    pip install -r requirements.txt
    python -m src.check_env               # versions; "cuda ready: True" if a GPU is available
    python -m pytest -q                   # expect: 8 passed

**Step 5: download the data**

    python -m src.download_trips          # monthly zips -> data/raw/trips (re-run to resume; 404 = not published yet)
    python -m src.download_station_info   # -> data/processed/station_info.csv

To pin a fixed end month so results can be repeated exactly, use `python -m src.download_trips --end YYYYMM`.

**Step 6: build the hourly table**

    python -m src.build_hourly            # or: python -m src.build_hourly --stations 100

**Step 7: sanity-check the result.** Open `data/processed/build_report.json` and confirm the months listed,
the number of stations selected, and the dropped-row counts. Keep this file with any results you report.

Reproducibility rules for everything we add
- Fixed random seed (42) in every training or sampling script, and record library versions with results.
- Settings live in `src/config.py` (start month, station count, selection date). Do not hard-code them elsewhere.
- Every reported number comes from a script in `src/`, not from a manual notebook edit.
- The set of monthly files used is recorded in `build_report.json`; a later download of newer months changes the window.
- Never touch the 2026 test window until the final run.

## 7. Repository layout

    src/
      config.py                  settings (paths, start month, timezone, station count, selection date)
      download_trips.py          monthly trip zips
      download_station_info.py   GBFS station metadata
      build_hourly.py            hourly station table + build report
      check_env.py               versions and GPU check
    tests/
      test_build_hourly.py       self-checks on small made-up trip files
    data/                        raw and processed data (ignored by git)
    reports/                     figures and report drafts
    CONTRIBUTIONS.md             who did what, and when

Planned: `src/weather.py`, `src/calendar_features.py`, `src/metrics.py`, `src/baselines.py`,
`src/transformer_model.py`, `src/train_transformer.py`, `src/chronos_baseline.py`.

## 8. Working together

- Run `git pull --rebase` before you start, commit small and often, and push when a step works.
- Do not commit data, model files or `.venv` (the `.gitignore` blocks them). Share the processed Parquet file
  through the shared Drive folder.
- Add a line to `CONTRIBUTIONS.md` when you finish something; the reports need an individual-contributions section.
- Keep the status table in section 2 current.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| PowerShell says scripts are disabled | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again |
| `py -3.12` says no suitable runtime | Use the version `py -0` lists, for example `py -3.11` |
| `cuda ready: False` | The CPU build of PyTorch is installed. Uninstall it and reinstall with a CUDA command |
| Download shows 404 for a month | Not published yet, or the file name changed; check the bucket index page |
| `build_hourly` stops: "missing columns" | The message lists the real columns; update `REQUIRED` in `src/build_hourly.py` |
| Out of memory | Close browsers; use fewer stations with `--stations 50` |
| Station ids do not match the GBFS feed | Compare `station_info.csv` columns (`short_name` vs `station_id`) with the trip files |

## 10. Timeline

| Date | Goal |
|---|---|
| Oct 26 | Repo and environment ready; trip data downloading; weather and events started |
| Oct 27 | Hourly table built and checked; metrics module; seasonal naive and Negative Binomial baselines on 2025 |
| Oct 28 | First small transformer run (loss curve and a sample forecast); Chronos-2 baseline started |
| Oct 29 | Results table; challenges, timeline and contributions written |
| Oct 30 | Interim report submitted (due 11:59 pm) |

## 11. References

1. Cheng, J., de Almeida Correia, G.H., Cats, O. and Azadeh, S.S., 2026. T-STAR: A context-aware transformer
   framework for short-term probabilistic demand forecasting in dock-based shared micro-mobility.
   Transportation Research Part C: Emerging Technologies, 192, p.105899.
2. Capital Bikeshare System Data (capitalbikeshare.com/system-data).
3. Open-Meteo Historical Weather and Previous Runs APIs (open-meteo.com).
4. Ansari, A.F., et al., 2025. Chronos-2: From univariate to universal forecasting. arXiv:2510.15821.
5. Hugging Face Transformers documentation, Time Series Transformer.
