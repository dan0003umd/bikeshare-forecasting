"""Create the sourced DC event calendar used by calendar_features."""
from __future__ import annotations

import argparse

import pandas as pd
import requests

from . import config as cfg

MLB_URL = "https://statsapi.mlb.com/api/v1/schedule"
# Official festival source pages recorded next to their date ranges.
FESTIVALS = [
    ("2023-03-20", "2023-04-16", "https://nationalcherryblossomfestival.org/wp-content/uploads/2022/12/23NCBF-Sponsor-Prospectus.pdf"),
    ("2024-03-20", "2024-04-14", "https://nationalcherryblossomfestival.org/wp-content/uploads/2023/11/Student-Account-Summer-2023.pdf"),
    ("2025-03-20", "2025-04-13", "https://nationalcherryblossomfestival.org/wp-content/uploads/2025/03/2025-Getting-to-the-Cherry-Blossom-Guide.pdf"),
]


def nationals_home_games(start: str, end: str) -> pd.DataFrame:
    rows = []
    # The schedule endpoint truncates a multi-season range, so query one regular
    # season at a time.  `gameType=R` intentionally excludes spring training.
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        response = requests.get(MLB_URL, params={
            "sportId": 1, "teamId": 120, "startDate": f"{year}-01-01", "endDate": f"{year}-12-31",
            "gameType": "R", "hydrate": "linescore", "limit": 5000,
        }, timeout=120)
        response.raise_for_status()
        games = [g for day in response.json().get("dates", []) for g in day.get("games", [])]
        for game in games:
            if game["teams"]["home"]["team"]["id"] == 120:
                rows.append({"date": game.get("officialDate", game["gameDate"][:10]), "event_type": "nationals_home_game",
                             "event_flag": True, "verified": True, "source": response.url})
    # Event features are daily; one or more games on a date must produce one flag.
    return pd.DataFrame(rows).drop_duplicates(["date", "event_type"])


def festival_days() -> pd.DataFrame:
    rows = []
    for start, end, source in FESTIVALS:
        for day in pd.date_range(start, end, freq="D"):
            rows.append({"date": day.strftime("%Y-%m-%d"), "event_type": "cherry_blossom_festival",
                         "event_flag": True, "verified": True, "source": source})
    return pd.DataFrame(rows)


def build_events(start="2023-01-01", end="2025-12-31") -> pd.DataFrame:
    home = nationals_home_games(start, end)
    festival = festival_days()
    events = pd.concat([home, festival], ignore_index=True).sort_values(["date", "event_type"])
    cfg.EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.EXTERNAL_DIR / "events.csv"
    events.to_csv(path, index=False)
    print(f"{len(home)} Nationals home games, {len(festival)} festival days -> {path}")
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()
    build_events(args.start, args.end)


if __name__ == "__main__":
    main()
