"""Build the station-by-hour table from the monthly Capital Bikeshare trip files.

Usage:  python -m src.build_hourly                 # uses settings in src/config.py
        python -m src.build_hourly --stations 100

Outputs (in data/processed/):
  station_hourly.parquet  one row per selected station per UTC hour:
                          station_id, ts_utc, checkouts, returns, net_outflow, dst_overlap
  station_summary.csv     the selected stations with simple quality numbers
  build_report.json       counts of everything dropped or fixed (use it for the report)

Design notes
  * Files are processed one month at a time (small memory use) and reduced to hourly counts.
  * Check-outs are counted from the start station and start time; returns from the end
    station and end time. A trip with only one station still counts on that side.
  * Trip timestamps are local wall-clock time. They are converted to UTC. The one repeated
    hour each November is ambiguous: its trips are assigned to the first occurrence, and
    both physical hours are flagged with dst_overlap so they can be masked in evaluation.
  * Stations are selected only if they were active at the start AND the end of the period
    (so zero-filling never invents demand for a closed or not-yet-opened station), ranked by
    check-outs before config.SELECTION_END.
"""
import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

REQUIRED = ["ride_id", "started_at", "ended_at", "start_station_id", "end_station_id"]
EDGE_DAYS = 2  # only trips near a month boundary can show up in two monthly files
HOUR = pd.Timedelta(hours=1)


def list_month_files(raw_dir, start=cfg.START):
    files = {}
    for p in sorted(Path(raw_dir).glob("*-capitalbikeshare-tripdata.zip")):
        m = re.match(r"(\d{4})(\d{2})-capitalbikeshare-tripdata\.zip$", p.name)
        if m and (int(m[1]), int(m[2])) >= tuple(start):
            files[(int(m[1]), int(m[2]))] = p
    return files


def read_trip_zip(path):
    """Read the columns we need as text. Fails loudly (listing the real columns) if one is missing."""
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX")]
        if not names:
            raise ValueError(f"{Path(path).name}: no CSV inside the zip")
        with z.open(names[0]) as f:
            cols = list(pd.read_csv(f, nrows=0).columns)
        missing = [c for c in REQUIRED if c not in cols]
        if missing:
            raise ValueError(f"{Path(path).name}: missing columns {missing}. Columns found: {cols}")
        with z.open(names[0]) as f:
            df = pd.read_csv(f, usecols=REQUIRED, dtype=str)
    for c in ["start_station_id", "end_station_id"]:
        df[c] = df[c].str.strip().str.replace(r"\.0$", "", regex=True).replace("", pd.NA)
    return df


def encode(series, codes):
    """Map station id strings to small integers (keeps memory low)."""
    for s in series.dropna().unique():
        codes.setdefault(s, len(codes))
    return series.map(codes).astype("int32")


def hourly_counts(df, ts_col, id_col, codes, t0_utc, n_hours):
    """Count trips per (station code, UTC hour index). Returns a Series with a (s, h) index."""
    d = df[[ts_col, id_col]].dropna()
    local_hour = d[ts_col].dt.floor("h")
    first_occurrence = np.ones(len(d), dtype=bool)  # ambiguous November hour -> first occurrence
    utc = local_hour.dt.tz_localize(cfg.LOCAL_TZ, ambiguous=first_occurrence, nonexistent="shift_forward")
    utc = utc.dt.tz_convert("UTC")
    h = ((utc - t0_utc) // HOUR).astype("int64")
    keep = (h >= 0) & (h < n_hours)
    out = pd.DataFrame({"s": encode(d[id_col], codes).values[keep.values], "h": h.values[keep.values]})
    counts = out.groupby(["s", "h"]).size().astype("int32")
    return counts, int((~keep).sum())


def longest_zero_run(is_zero):
    """Length of the longest run of True values."""
    if not is_zero.any():
        return 0
    padded = np.concatenate([[0], is_zero.astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(padded))
    return int((edges[1::2] - edges[0::2]).max())


def build_station_hourly(raw_dir=cfg.RAW_TRIPS_DIR, out_dir=cfg.PROCESSED_DIR, n_stations=cfg.N_STATIONS,
                         start=cfg.START, selection_end=cfg.SELECTION_END, slack_days=31):
    files = list_month_files(raw_dir, start)
    if not files:
        raise FileNotFoundError(f"No *-capitalbikeshare-tripdata.zip files found in {raw_dir}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0_utc = pd.Timestamp(year=start[0], month=start[1], day=1).tz_localize(cfg.LOCAL_TZ).tz_convert("UTC")
    ly, lm = max(files)
    end_local = pd.Timestamp(year=ly, month=lm, day=1) + pd.offsets.MonthBegin(1)
    end_utc = end_local.tz_localize(cfg.LOCAL_TZ).tz_convert("UTC")
    n_hours = int((end_utc - t0_utc) // HOUR)

    stats = dict(months=[f"{y}{m:02d}" for y, m in files], rows_read=0, bad_timestamps=0,
                 within_file_duplicates=0, cross_file_duplicates=0, missing_start_station=0,
                 missing_end_station=0, end_before_start=0, events_outside_window=0)
    codes, seen_edge_ids, co_parts, ret_parts = {}, set(), [], []

    for (y, m), path in files.items():
        df = read_trip_zip(path)
        stats["rows_read"] += len(df)
        df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
        df["ended_at"] = pd.to_datetime(df["ended_at"], errors="coerce")
        for c in ["started_at", "ended_at"]:
            if df[c].dt.tz is not None:
                raise ValueError(f"{path.name}: {c} has a timezone; expected local wall-clock time")
        bad = df["started_at"].isna() | df["ended_at"].isna()
        stats["bad_timestamps"] += int(bad.sum())
        df = df[~bad]

        n = len(df)
        df = df.drop_duplicates("ride_id")
        stats["within_file_duplicates"] += n - len(df)

        ms = pd.Timestamp(year=y, month=m, day=1)
        lo, hi = ms + pd.Timedelta(days=EDGE_DAYS), ms + pd.offsets.MonthBegin(1) - pd.Timedelta(days=EDGE_DAYS)
        edge = lambda d: (d["started_at"] < lo) | (d["started_at"] >= hi)
        dup = edge(df) & df["ride_id"].isin(seen_edge_ids)
        stats["cross_file_duplicates"] += int(dup.sum())
        df = df[~dup]
        seen_edge_ids.update(df.loc[edge(df), "ride_id"])

        stats["missing_start_station"] += int(df["start_station_id"].isna().sum())
        stats["missing_end_station"] += int(df["end_station_id"].isna().sum())
        stats["end_before_start"] += int((df["ended_at"] < df["started_at"]).sum())

        co, dropped1 = hourly_counts(df, "started_at", "start_station_id", codes, t0_utc, n_hours)
        ret, dropped2 = hourly_counts(df, "ended_at", "end_station_id", codes, t0_utc, n_hours)
        stats["events_outside_window"] += dropped1 + dropped2
        co_parts.append(co)
        ret_parts.append(ret)
        print(f"{y}{m:02d}: {len(df):>8,} trips kept")

    co = pd.concat(co_parts).groupby(level=[0, 1]).sum().rename("checkouts")
    ret = pd.concat(ret_parts).groupby(level=[0, 1]).sum().rename("returns")
    ev = pd.concat([co, ret], axis=1).fillna(0).astype("int32")
    ev.index.names = ["s", "h"]
    ev = ev.reset_index()

    # --- station selection -------------------------------------------------------------
    sel_h = int((pd.Timestamp(selection_end).tz_localize(cfg.LOCAL_TZ).tz_convert("UTC") - t0_utc) // HOUR)
    g = ev.groupby("s")["h"]
    first, last = g.min(), g.max()
    active = (first <= slack_days * 24) & (last >= n_hours - 1 - slack_days * 24)
    train_out = ev[ev["h"] < sel_h].groupby("s")["checkouts"].sum().reindex(first.index).fillna(0)
    candidates = train_out[active].sort_values(ascending=False)
    chosen = list(candidates.index[:n_stations])
    stats["stations_seen"] = int(len(first))
    stats["stations_active_whole_period"] = int(active.sum())
    stats["stations_selected"] = len(chosen)
    if len(chosen) < n_stations:
        print(f"WARNING: only {len(chosen)} stations were active for the whole period")

    # --- full grid: every selected station x every hour, zero where no trips ------------
    hours = np.arange(n_hours)
    mi = pd.MultiIndex.from_product([chosen, hours], names=["s", "h"])
    grid = ev.set_index(["s", "h"]).reindex(mi, fill_value=0).reset_index()
    inv = {v: k for k, v in codes.items()}
    ts = t0_utc + pd.to_timedelta(grid["h"].to_numpy(), unit="h")
    local_naive = pd.DatetimeIndex(t0_utc + pd.to_timedelta(hours, unit="h")).tz_convert(cfg.LOCAL_TZ).tz_localize(None)
    overlap_by_hour = local_naive.duplicated(keep=False)
    table = pd.DataFrame({
        "station_id": grid["s"].map(inv).astype("string"),
        "ts_utc": ts,
        "checkouts": grid["checkouts"].astype("int32"),
        "returns": grid["returns"].astype("int32"),
    })
    table["net_outflow"] = table["checkouts"] - table["returns"]
    table["dst_overlap"] = overlap_by_hour[grid["h"].to_numpy()]
    table.to_parquet(out_dir / "station_hourly.parquet", index=False)

    rows = []
    for s in chosen:
        sub = grid[grid["s"] == s]
        idle = ((sub["checkouts"] == 0) & (sub["returns"] == 0)).to_numpy()
        rows.append(dict(station_id=inv[s], checkouts_before_selection_end=int(candidates[s]),
                         total_checkouts=int(sub["checkouts"].sum()), total_returns=int(sub["returns"].sum()),
                         longest_idle_run_hours=longest_zero_run(idle)))
    pd.DataFrame(rows).to_csv(out_dir / "station_summary.csv", index=False)

    stats.update(hours_in_window=n_hours, window_start_utc=str(t0_utc), window_end_utc=str(end_utc),
                 table_rows=len(table))
    with open(out_dir / "build_report.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    return table, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", type=int, default=cfg.N_STATIONS)
    args = ap.parse_args()
    build_station_hourly(n_stations=args.stations)


if __name__ == "__main__":
    main()
