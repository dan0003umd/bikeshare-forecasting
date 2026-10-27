"""Sample the trained transformer on the common 2025 24-hour windows."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as cfg
from .baselines import save_result
from .forecast_windows import validation_window_summary
from .metrics import score
from .train_transformer import prepare_datasets, set_seed


def validation_metadata(table: pd.DataFrame, pairs: list[tuple[int, int]]) -> pd.DataFrame:
    """Metadata in the exact non-shuffled DataLoader order of generated windows."""
    stations = sorted(table.station_id.unique())
    frames = [table[table.station_id.eq(station)].sort_values("ts_utc").reset_index(drop=True) for station in stations]
    pieces = [frames[station].iloc[start:start + 24][["station_id", "ts_utc", "checkouts"]]
              for station, start in pairs]
    return pd.concat(pieces, ignore_index=True).rename(columns={"checkouts": "actual"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint", default="transformer_last.pt", help="checkpoint filename under checkpoints/")
    parser.add_argument("--model-name", default="hf_transformer_negative_binomial")
    args = parser.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
    set_seed()
    checkpoint = torch.load(cfg.CHECKPOINTS_DIR / args.checkpoint, map_location="cpu", weights_only=False)
    table = pd.read_parquet(cfg.PROCESSED_DIR / "model_table.parquet")
    # Training used weekly validation windows to fit in the first run; reporting
    # uses every eligible 2025 day, exactly matching every other model.
    use_embedding = checkpoint.get("station_embedding", True)
    use_weather = checkpoint.get("observed_weather_oracle", False)
    _, validation, _, _ = prepare_datasets(
        table, validation_every_n_days=1, use_station_embedding=use_embedding, include_observed_weather=use_weather)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformerForPrediction(TimeSeriesTransformerConfig.from_dict(checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"]); model.to(device).eval()
    loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0)
    all_y, all_q = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = {"past_values": batch["past_values"], "past_time_features": batch["past_time_features"],
                      "past_observed_mask": batch["past_observed_mask"], "future_time_features": batch["future_time_features"]}
            if "static_categorical_features" in batch:
                inputs["static_categorical_features"] = batch["static_categorical_features"]
            output = model.generate(**inputs)
            # HF returns (batch, 100 samples, 24 horizon), flatten station-days/hours.
            samples = output.sequences.detach().cpu().numpy()
            all_q.append(np.quantile(samples, cfg.QUANTILES, axis=1).transpose(1, 2, 0).reshape(-1, len(cfg.QUANTILES)))
            all_y.append(batch["future_values"].detach().cpu().numpy().reshape(-1))
    result = {"model": args.model_name, **score(np.concatenate(all_y), np.concatenate(all_q))}
    result.update(validation_window_summary(table)); result["excluded_warmup_rows"] = 0
    prediction = validation_metadata(table, validation.pairs)
    if len(prediction) != result["n_observations"]:
        raise RuntimeError(f"prediction metadata has {len(prediction)} rows, expected {result['n_observations']}")
    prediction[[f"q{int(q * 100):02d}" for q in cfg.QUANTILES]] = np.concatenate(all_q)
    cfg.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.PREDICTIONS_DIR / f"{args.model_name}_validation_2025.parquet"
    prediction.to_parquet(path, index=False)
    print(f"{len(prediction):,} prediction rows -> {path}")
    save_result(result)


if __name__ == "__main__":
    main()
