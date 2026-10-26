"""Download the monthly Capital Bikeshare trip files (skips files already downloaded).

Usage:  python -m src.download_trips            # 2023-01 up to last month
        python -m src.download_trips --end 202609
"""
import argparse
import time
from datetime import date

import requests

from . import config as cfg


def month_range(start, end):
    y, m = start
    while (y, m) <= end:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def default_end():
    t = date.today()
    return (t.year - 1, 12) if t.month == 1 else (t.year, t.month - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", help="last month to download, YYYYMM (default: last month)")
    args = ap.parse_args()
    end = (int(args.end[:4]), int(args.end[4:])) if args.end else default_end()

    cfg.RAW_TRIPS_DIR.mkdir(parents=True, exist_ok=True)
    got, missing = 0, []
    for y, m in month_range(cfg.START, end):
        name = f"{y}{m:02d}-capitalbikeshare-tripdata.zip"
        dest = cfg.RAW_TRIPS_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print("have   ", name)
            continue
        r = requests.get(cfg.TRIP_URL.format(ym=f"{y}{m:02d}"), stream=True, timeout=120)
        if r.status_code == 404:
            print("404    ", name, "(not published yet, or the file name changed)")
            missing.append(name)
            continue
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp.rename(dest)
        got += 1
        print("saved  ", name, f"{dest.stat().st_size / 1e6:.1f} MB")
        time.sleep(0.5)
    print(f"done: {got} new files, {len(missing)} missing")
    if missing:
        print("missing:", ", ".join(missing))


if __name__ == "__main__":
    main()
