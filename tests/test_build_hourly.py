"""Checks the table builder on small made-up trip files with known answers.
Run:  python -m pytest -q      (or: python -m tests.test_build_hourly)"""
import io
import zipfile

import pandas as pd

from src import build_hourly as bh

COLS = ["ride_id", "rideable_type", "started_at", "ended_at", "start_station_name", "start_station_id",
        "end_station_name", "end_station_id", "start_lat", "start_lng", "end_lat", "end_lng", "member_casual"]


def write_month(path, rows):
    df = pd.DataFrame(rows, columns=["ride_id", "started_at", "ended_at", "start_station_id", "end_station_id"])
    for c in COLS:
        if c not in df:
            df[c] = "x"
    buf = io.StringIO()
    df[COLS].to_csv(buf, index=False)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(path.name.replace(".zip", ".csv"), buf.getvalue())


def daily(prefix, year, month, days, st_from, st_to, hour=12):
    return [(f"{prefix}{year}{month:02d}{d:02d}", f"{year}-{month:02d}-{d:02d} {hour:02d}:10:00",
             f"{year}-{month:02d}-{d:02d} {hour:02d}:25:00", st_from, st_to) for d in days]


def build(tmp_path):
    cfg_start = (2023, 10)
    oct_rows = daily("a", 2023, 10, range(1, 32), "S1", "S2") + daily("b", 2023, 10, range(1, 32), "S2", "S1")
    nov_rows = daily("a", 2023, 11, range(1, 31), "S1", "S2") + daily("b", 2023, 11, range(1, 31), "S2", "S1")
    dec_rows = daily("a", 2023, 12, range(1, 32), "S1", "S2") + daily("b", 2023, 12, range(1, 32), "S2", "S1")
    for rows, y, m in [(oct_rows, 2023, 10), (nov_rows, 2023, 11), (dec_rows, 2023, 12)]:
        rows += daily("c", y, m, range(1, 29, 2), "S3", "S3")          # S3: quieter, every other day
    dec_rows += daily("d", 2023, 12, range(1, 32), "S4", "S4")            # S4 appears only in December
    dec_rows += daily("e", 2023, 12, range(1, 32), "S4", "S4", hour=9)    # ...with lots of trips

    # (1) a trip crossing the Oct/Nov boundary, present in BOTH files -> counted once
    oct_rows.append(("cross", "2023-10-31 23:40:00", "2023-11-01 00:10:00", "S1", "S2"))
    nov_rows.append(("cross", "2023-10-31 23:40:00", "2023-11-01 00:10:00", "S1", "S2"))
    # (2) a repeated row inside one file -> counted once
    nov_rows += [("twice", "2023-11-10 08:05:00", "2023-11-10 08:20:00", "S1", "S2")] * 2
    # (3) missing start station: counts as a return only; (4) missing end station: check-out only
    nov_rows.append(("nostart", "2023-11-11 08:05:00", "2023-11-11 08:20:00", None, "S1"))
    nov_rows.append(("noend", "2023-11-12 08:05:00", "2023-11-12 08:20:00", "S1", None))
    # (5) unparseable timestamp -> dropped
    nov_rows.append(("badts", "not a time", "2023-11-13 08:20:00", "S1", "S2"))
    # (6) the repeated November hour (clocks go back on 2023-11-05): 01:30 local, ambiguous
    nov_rows.append(("dst", "2023-11-05 01:30:00", "2023-11-05 01:50:00", "S1", "S2"))

    raw = tmp_path / "raw"
    raw.mkdir()
    write_month(raw / "202310-capitalbikeshare-tripdata.zip", oct_rows)
    write_month(raw / "202311-capitalbikeshare-tripdata.zip", nov_rows)
    write_month(raw / "202312-capitalbikeshare-tripdata.zip", dec_rows)
    table, stats = bh.build_station_hourly(raw, tmp_path / "out", n_stations=2, start=cfg_start,
                                           selection_end="2023-12-01")
    return table, stats


def test_builder(tmp_path):
    table, stats = build(tmp_path)

    # data-quality counters
    assert stats["within_file_duplicates"] == 1
    assert stats["cross_file_duplicates"] == 1
    assert stats["bad_timestamps"] == 1
    assert stats["missing_start_station"] == 1 and stats["missing_end_station"] == 1

    # stations: S1 and S2 chosen; S3 is too quiet; S4 only exists in December (not active all period)
    assert set(table["station_id"]) == {"S1", "S2"}
    assert stats["stations_selected"] == 2

    # window is Oct 1 - Dec 31 2023 local: 92 days; Nov 5 has 25 hours
    hours = table[table["station_id"] == "S1"]
    assert len(hours) == 92 * 24 + 1
    assert hours["ts_utc"].is_monotonic_increasing and hours["ts_utc"].dt.tz is not None

    s1 = hours.set_index("ts_utc")
    # S1 checkouts: 92 daily trips + cross-boundary trip (once) + 'twice' (once) + 'noend' + 'dst'
    assert s1["checkouts"].sum() == 92 + 1 + 1 + 1 + 1
    # S1 returns: 92 daily trips from S2 + 'nostart' (S1 is its end station)
    assert s1["returns"].sum() == 92 + 1

    # cross-boundary trip: 23:40 EDT on Oct 31 = 03:40 UTC Nov 1 -> hour 03:00 UTC has 2 checkouts?
    # (no other S1 trip at that hour, so exactly 1)
    assert s1.loc[pd.Timestamp("2023-11-01 03:00", tz="UTC"), "checkouts"] == 1
    # that trip ends 00:10 EDT Nov 1 = 04:10 UTC -> a return for S2, not S1
    s2 = table[table["station_id"] == "S2"].set_index("ts_utc")
    assert s2.loc[pd.Timestamp("2023-11-01 04:00", tz="UTC"), "returns"] == 1

    # ambiguous hour: 01:30 on Nov 5 -> first occurrence, 05:30 UTC; both overlapping UTC hours flagged
    assert s1.loc[pd.Timestamp("2023-11-05 05:00", tz="UTC"), "checkouts"] == 1
    assert s1.loc[pd.Timestamp("2023-11-05 06:00", tz="UTC"), "checkouts"] == 0
    flagged = s1.index[s1["dst_overlap"]]
    assert list(flagged) == [pd.Timestamp("2023-11-05 05:00", tz="UTC"), pd.Timestamp("2023-11-05 06:00", tz="UTC")]

    # net outflow is consistent, and nothing is negative
    assert (table["net_outflow"] == table["checkouts"] - table["returns"]).all()
    assert (table[["checkouts", "returns"]] >= 0).all().all()


def test_missing_column_message(tmp_path):
    p = tmp_path / "202301-capitalbikeshare-tripdata.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("x.csv", "Duration,Start date\n1,2\n")
    try:
        bh.read_trip_zip(p)
    except ValueError as e:
        assert "Columns found" in str(e)
    else:
        raise AssertionError("expected ValueError")
