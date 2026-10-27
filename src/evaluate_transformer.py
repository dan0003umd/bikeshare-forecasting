"""Sample the trained transformer on the common 2025 24-hour windows."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as cfg
from .baselines import save_result
from .forecast_windows import validation_window_summary
from .metrics import score
from .train_transformer import WindowDataset, prepare_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
    checkpoint = torch.load(cfg.CHECKPOINTS_DIR / "transformer_last.pt", map_location="cpu", weights_only=False)
    table = pd.read_parquet(cfg.PROCESSED_DIR / "model_table.parquet")
    # Training used weekly validation windows to fit in the first run; reporting
    # uses every eligible 2025 day, exactly matching every other model.
    _, validation, _, _ = prepare_datasets(table, validation_every_n_days=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformerForPrediction(TimeSeriesTransformerConfig.from_dict(checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"]); model.to(device).eval()
    loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0)
    all_y, all_q = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model.generate(past_values=batch["past_values"], past_time_features=batch["past_time_features"],
                                    past_observed_mask=batch["past_observed_mask"],
                                    future_time_features=batch["future_time_features"],
                                    static_categorical_features=batch["static_categorical_features"])
            # HF returns (batch, 100 samples, 24 horizon), flatten station-days/hours.
            samples = output.sequences.detach().cpu().numpy()
            all_q.append(np.quantile(samples, cfg.QUANTILES, axis=1).transpose(1, 2, 0).reshape(-1, len(cfg.QUANTILES)))
            all_y.append(batch["future_values"].detach().cpu().numpy().reshape(-1))
    result = {"model": "hf_transformer_negative_binomial", **score(np.concatenate(all_y), np.concatenate(all_q))}
    result.update(validation_window_summary(table)); result["excluded_warmup_rows"] = 0
    save_result(result)


if __name__ == "__main__":
    main()
