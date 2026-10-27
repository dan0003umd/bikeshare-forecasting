"""Record published 2026 trip-archive availability without downloading archives."""
from __future__ import annotations

from datetime import datetime, timezone

import requests

from . import config as cfg


def check_2026_availability() -> list[tuple[str, int]]:
    results = []
    for month in range(1, 13):
        ym = f"2026{month:02d}"
        response = requests.head(cfg.TRIP_URL.format(ym=ym), timeout=30, allow_redirects=True)
        results.append((ym, response.status_code))
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.REPORTS_DIR / "trip_availability_2026.md"
    available = [ym for ym, status in results if status == 200]
    lines = ["# 2026 Trip Archive Availability", "",
             f"Checked with HTTP HEAD at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
             "No archive was downloaded or used for modeling/evaluation.", "",
             "| Month | HTTP status |", "|---|---:|"]
    lines.extend(f"| {ym} | {status} |" for ym, status in results)
    lines.extend(["", f"Available: {', '.join(available) if available else 'none'}."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)
    print("available:", ", ".join(available) if available else "none")
    return results


if __name__ == "__main__":
    check_2026_availability()
