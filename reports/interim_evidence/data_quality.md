# Data Quality

- Input monthly archives: 45 (202301 through 202609).
- Rows read: 22,072,596; table rows: 1,642,750.
- Selected stations: 50 of 911 seen (678 active across the full observed window).
- Rows with unparseable timestamps: 0; within-file duplicate rides removed: 0; cross-file boundary duplicates removed: 205.
- Trips lacking a start station: 3,812,905 (17.27% of rows read); lacking an end station: 3,974,624 (18.01%). Demand counts only trip sides with a station: check-outs require a start station and returns require an end station.
- Trips with end before start: 986; these are reported for audit but not discarded by the present aggregation.
- Daylight saving: the repeated November local hour is assigned to the first occurrence and both physical hours receive `dst_overlap=True`. There are 300 flagged station-hours in the built table; evaluation excludes DST/non-24-hour local days.

Stations are ranked by check-outs strictly before 2025-01-01, after requiring activity near both ends of the downloaded period. The top 50 are then zero-filled at every UTC hour.
