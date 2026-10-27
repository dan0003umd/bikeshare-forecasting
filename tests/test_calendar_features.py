import pandas as pd

from src.calendar_features import add_calendar_features


def test_calendar_uses_local_not_utc_date(tmp_path):
    events = tmp_path / "events.csv"
    events.write_text("date,event_flag\n2025-01-01,True\n")
    table = pd.DataFrame({"ts_utc": pd.to_datetime(["2025-01-01T04:00Z", "2025-01-01T05:00Z"])})
    result = add_calendar_features(table, events)
    assert result["hour"].tolist() == [23, 0]
    assert result["local_date"].tolist() == ["2024-12-31", "2025-01-01"]
    assert result["event_flag"].tolist() == [False, True]
