"""Chronos-2 zero-shot evaluation on a fixed 10-station x 60-day 2025 slice."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as cfg
from .metrics import score
from .transformer_model import PAST_LENGTH, PREDICTION_LENGTH


def load_pretest_table() -> pd.DataFrame:
    """Parquet predicate prevents this script from reading 2026 test rows."""
    cutoff = pd.Timestamp(cfg.VALID_END).tz_localize(cfg.LOCAL_TZ).tz_convert("UTC")
    table = pd.read_parquet(cfg.PROCESSED_DIR / "model_table.parquet", filters=[("ts_utc", "<", cutoff)])
    if (pd.to_datetime(table.ts_utc, utc=True) >= cutoff).any():
        raise RuntimeError("pre-test table unexpectedly contains 2026 rows")
    return table.sort_values(["station_id", "ts_utc"]).reset_index(drop=True)


def fixed_subsample(table: pd.DataFrame, stations: int = 10, days: int = 60) -> tuple[list[str], list[object]]:
    """First sorted stations and first eligible local dates: deterministic and recorded."""
    local_date = table.ts_utc.dt.tz_convert(cfg.LOCAL_TZ).dt.date
    candidate = (table.ts_utc >= pd.Timestamp(cfg.VALID_START).tz_localize(cfg.LOCAL_TZ).tz_convert("UTC"))
    candidate &= table.ts_utc < pd.Timestamp(cfg.VALID_END).tz_localize(cfg.LOCAL_TZ).tz_convert("UTC")
    counts = table.loc[candidate].assign(local_date=local_date[candidate]).groupby("local_date").size()
    good_days = list(counts.index[counts.eq(24 * table.station_id.nunique())])
    selected_stations = sorted(table.station_id.unique())[:stations]
    selected_days = good_days[:days]
    if len(selected_stations) != stations or len(selected_days) != days:
        raise RuntimeError("requested Chronos subsample is unavailable")
    return selected_stations, selected_days


def subsample_mask(frame: pd.DataFrame, stations: list[str], days: list[object]) -> pd.Series:
    local_date = pd.to_datetime(frame.ts_utc, utc=True).dt.tz_convert(cfg.LOCAL_TZ).dt.date
    return frame.station_id.isin(stations) & local_date.isin(days)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="amazon/chronos-2")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    import torch
    from chronos import Chronos2Pipeline

    table = load_pretest_table()
    stations, days = fixed_subsample(table)
    frames = {station: table[table.station_id.eq(station)].reset_index(drop=True) for station in stations}
    contexts, rows = [], []
    for station, frame in frames.items():
        dates = frame.ts_utc.dt.tz_convert(cfg.LOCAL_TZ).dt.date
        for day in days:
            start = np.flatnonzero((dates == day) & frame.hour.eq(0).to_numpy())
            if len(start) != 1 or start[0] < PAST_LENGTH:
                raise RuntimeError(f"invalid Chronos origin {station=} {day=}")
            start = int(start[0])
            contexts.append(frame.checkouts.iloc[start - PAST_LENGTH:start].to_numpy("float32"))
            rows.append(frame.iloc[start:start + PREDICTION_LENGTH][["station_id", "ts_utc", "checkouts"]])
    pipeline = Chronos2Pipeline.from_pretrained(args.model, device_map="cuda" if torch.cuda.is_available() else "cpu",
                                                torch_dtype=torch.float32)
    quantiles, _ = pipeline.predict_quantiles(
        contexts, prediction_length=PREDICTION_LENGTH, context_length=PAST_LENGTH, batch_size=args.batch_size,
        quantile_levels=list(cfg.QUANTILES), cross_learning=False,
    )
    prediction = pd.concat(rows, ignore_index=True).rename(columns={"checkouts": "actual"})
    values = np.concatenate([item.detach().cpu().numpy().reshape(PREDICTION_LENGTH, len(cfg.QUANTILES)) for item in quantiles])
    if len(prediction) != len(values) or len(prediction) != len(stations) * len(days) * PREDICTION_LENGTH:
        raise RuntimeError("Chronos forecast/metadata alignment failed")
    prediction[[f"q{int(q * 100):02d}" for q in cfg.QUANTILES]] = values
    output = cfg.PREDICTIONS_DIR / "chronos2_zero_shot_subsample_10stations_60days.parquet"
    prediction.to_parquet(output, index=False)
    mask_table = subsample_mask(table, stations, days)
    summary_rows = [{"model": "chronos2_zero_shot", **score(prediction.actual, values),
                     "n_observations": len(prediction)}]
    for path in sorted(cfg.PREDICTIONS_DIR.glob("*_validation_2025.parquet")):
        saved = pd.read_parquet(path)
        mask = subsample_mask(saved, stations, days)
        subset = saved.loc[mask]
        if len(subset) != len(prediction):
            raise RuntimeError(f"{path.name} has {len(subset)} rather than {len(prediction)} selected rows")
        model = path.name.removesuffix("_validation_2025.parquet")
        summary_rows.append({"model": model, **score(subset.actual, subset[[f"q{int(q * 100):02d}" for q in cfg.QUANTILES]]),
                             "n_observations": len(subset)})
    summary = pd.DataFrame(summary_rows).sort_values("pinball_loss")
    summary.to_csv(cfg.REPORTS_DIR / "chronos_subsample_validation_2025.csv", index=False)
    selected = pd.DataFrame({"station_id": np.repeat(stations, len(days)), "local_date": days * len(stations)})
    selected.to_csv(cfg.REPORTS_DIR / "chronos_subsample_definition.csv", index=False)
    print(summary.to_string(index=False))
    print(f"saved {len(prediction):,} Chronos predictions -> {output}")


if __name__ == "__main__":
    main()
