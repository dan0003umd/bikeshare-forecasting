# Challenges and Changes

| Date | Status | Issue and response |
|---|---|---|
| 2026-10-26 | solved | Current trip archives used the expected `ride_id`, timestamps, and station-id columns; the builder verifies those columns and fails loudly if they change. |
| 2026-10-26 | solved | 205 duplicate rides appeared across month boundaries; the builder de-duplicated boundary ride IDs. |
| 2026-10-26 | worked around | 3,812,905 start and 3,974,624 end station fields are missing in raw trips (exact side-specific counts are in `data_quality.md`); each available trip side remains usable. |
| 2026-10-26 | solved | Repeated November local hours are ambiguous. They are flagged `dst_overlap`; the two DST transition days are excluded from fixed-24-hour evaluation. |
| 2026-10-26 | worked around | Previous Runs lead-1 weather is incomplete in early years. Observed weather is joined but not yet passed to a reported model; any future observed-weather result will be labelled ORACLE. |
| 2026-10-27 | solved | Seasonal naive initially rolled through earlier 2025 demand. It was changed to fit only 2023--2024 history, with a regression test. |
| 2026-10-27 | open | The first transformer is a five-epoch weekly-stride run, so it is undertrained; daily-stride, early-stopped training and embedding/weather ablations remain. |
| 2026-10-27 | open | Per-row transformer samples were not persisted after evaluation, so the current hourly-error figure only includes the retained seasonal-naive predictions. |
