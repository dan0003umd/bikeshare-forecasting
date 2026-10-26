"""Download station metadata (name, coordinates, dock capacity) from the GBFS feed.

Usage:  python -m src.download_station_info
Writes data/processed/station_info.csv with every field the feed provides. The trip files
identify stations by a short id, so check which column here matches (usually `short_name`).
"""
import pandas as pd
import requests

from . import config as cfg


def main():
    r = requests.get(cfg.GBFS_STATION_INFO_URL, timeout=60)
    r.raise_for_status()
    stations = r.json().get("data", {}).get("stations", [])
    df = pd.DataFrame(stations)
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.PROCESSED_DIR / "station_info.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} stations -> {out}")
    print("columns:", list(df.columns))


if __name__ == "__main__":
    main()
