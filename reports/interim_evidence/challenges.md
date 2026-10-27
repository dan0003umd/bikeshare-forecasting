# Challenges and Changes

| Date | Status | Issue and response |
|---|---|---|
| 2026-10-26 | solved | Current trip archives used the expected `ride_id`, timestamps, and station-id columns; the builder verifies those columns and fails loudly if they change. |
| 2026-10-26 | solved | 205 duplicate rides appeared across month boundaries; the builder de-duplicated boundary ride IDs. |
| 2026-10-26 | worked around | 3,812,905 start and 3,974,624 end station fields are missing in raw trips (exact side-specific counts are in `data_quality.md`); each available trip side remains usable. |
| 2026-10-26 | solved | Repeated November local hours are ambiguous. They are flagged `dst_overlap`; the two DST transition days are excluded from fixed-24-hour evaluation. |
| 2026-10-26 | worked around | Previous Runs lead-1 weather is incomplete in early years. Observed weather is joined but not yet passed to a reported model; any future observed-weather result will be labelled ORACLE. |
| 2026-10-27 | solved | Review showed the original rolling seasonal naive was causal: it used the preceding eight same-weekday/hour observations, all before each forecast origin. It is restored as `seasonal_naive_rolling_8week`; the frozen 2023--2024-only version is retained as a separate stricter comparator. |
| 2026-10-27 | solved | Daily-stride transformer runs used 35,650 train and 18,150 validation windows with seed 42 and patience-3 early stopping. Embedding, no-embedding, and observed-weather ORACLE ablations were evaluated on the common window. |
| 2026-10-27 | solved | Every evaluator now persists actuals and all 19 quantiles by station/hour under ignored `data/predictions/`, so figures can be rebuilt without re-evaluating. |
| 2026-10-27 | worked around | Observed future weather produces an ORACLE comparison only; incomplete 2023--2024 Previous Runs coverage prevents a deployable forecast-weather experiment. |
