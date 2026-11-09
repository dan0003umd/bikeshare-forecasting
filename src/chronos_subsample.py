"""Chronos-2 zero-shot evaluation on a fixed station x day slice of 2025 validation (default 10 x 60)."""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from . import config as cfg
from .metrics import score
from .splits import load_development_model_table
from .transformer_model import PAST_LENGTH, PREDICTION_LENGTH

CHUNK = 800  # forecast windows per call; series are independent (cross_learning=False)


def load_pretest_table() -> pd.DataFrame:
    """Use the shared guarded loader, then establish deterministic station order."""
    return load_development_model_table().sort_values(["station_id", "ts_utc"]).reset_index(drop=True)


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
    parser.add_argument("--stations", type=int, default=10)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--time-only", type=int, default=0,
                        help="forecast only this many windows, print a projected runtime, write nothing")
    args = parser.parse_args()
    import torch
    from chronos import Chronos2Pipeline

    table = load_pretest_table()
    stations, days = fixed_subsample(table, args.stations, args.days)
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
    kwargs = dict(prediction_length=PREDICTION_LENGTH, context_length=PAST_LENGTH, batch_size=args.batch_size,
                  quantile_levels=list(cfg.QUANTILES), cross_learning=False)
    if args.time_only:
        n = min(args.time_only, len(contexts))
        pipeline.predict_quantiles(contexts[:args.batch_size], **kwargs)  # warm-up
        began = time.perf_counter()
        pipeline.predict_quantiles(contexts[:n], **kwargs)
        seconds = time.perf_counter() - began
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device={device}; {n} windows in {seconds:.1f}s; projected {len(contexts):,} windows "
              f"take about {seconds / n * len(contexts) / 60:.1f} minutes")
        return
    quantiles, began = [], time.perf_counter()
    for i in range(0, len(contexts), CHUNK):
        chunk_quantiles, _ = pipeline.predict_quantiles(contexts[i:i + CHUNK], **kwargs)
        quantiles.extend(chunk_quantiles)
        done = min(i + CHUNK, len(contexts))
        print(f"{done:,}/{len(contexts):,} windows, {(time.perf_counter() - began) / 60:.1f} min", flush=True)
    prediction = pd.concat(rows, ignore_index=True).rename(columns={"checkouts": "actual"})
    values = np.concatenate([item.detach().cpu().numpy().reshape(PREDICTION_LENGTH, len(cfg.QUANTILES)) for item in quantiles])
    if len(prediction) != len(values) or len(prediction) != len(stations) * len(days) * PREDICTION_LENGTH:
        raise RuntimeError("Chronos forecast/metadata alignment failed")
    prediction[[f"q{int(q * 100):02d}" for q in cfg.QUANTILES]] = values
    default_slice = (args.stations, args.days) == (10, 60)
    label = "subsample_10stations_60days" if default_slice else f"{args.stations}stations_{args.days}days"
    output = cfg.PREDICTIONS_DIR / f"chronos2_zero_shot_{label}.parquet"
    prediction.to_parquet(output, index=False)
    summary_rows = [{"model": "chronos2_zero_shot", **score(prediction.actual, values),
                     "n_observations": len(prediction)}]
    for path in sorted(cfg.PREDICTIONS_DIR.glob("*_validation_2025.parquet")):
        if "probe" in path.name:
            continue  # seed/sampling probes are diagnostics, not comparison models
        saved = pd.read_parquet(path)
        mask = subsample_mask(saved, stations, days)
        subset = saved.loc[mask]
        if len(subset) != len(prediction):
            raise RuntimeError(f"{path.name} has {len(subset)} rather than {len(prediction)} selected rows")
        model = path.name.removesuffix("_validation_2025.parquet")
        summary_rows.append({"model": model, **score(subset.actual, subset[[f"q{int(q * 100):02d}" for q in cfg.QUANTILES]]),
                             "n_observations": len(subset)})
    summary = pd.DataFrame(summary_rows).sort_values("pinball_loss")
    summary_name = "chronos_subsample_validation_2025.csv" if default_slice else f"chronos_{label}_validation_2025.csv"
    definition_name = "chronos_subsample_definition.csv" if default_slice else f"chronos_{label}_definition.csv"
    summary.to_csv(cfg.REPORTS_DIR / summary_name, index=False)
    selected = pd.DataFrame({"station_id": np.repeat(stations, len(days)), "local_date": days * len(stations)})
    selected.to_csv(cfg.REPORTS_DIR / definition_name, index=False)
    print(summary.to_string(index=False))
    print(f"saved {len(prediction):,} Chronos predictions -> {output}")


if __name__ == "__main__":
    main()