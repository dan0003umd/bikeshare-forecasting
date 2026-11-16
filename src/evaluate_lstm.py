"""Score the trained quantile LSTM on every eligible 2025 day, matching every other model."""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config as cfg
from .baselines import save_result
from .evaluate_transformer import validation_metadata
from .forecast_windows import validation_window_summary
from .lstm_model import QuantileLSTM
from .metrics import score
from .splits import load_development_model_table
from .train_lstm import MODEL_INPUTS
from .train_transformer import prepare_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="lstm_quantile_best.pt", help="checkpoint filename under checkpoints/")
    parser.add_argument("--model-name", default="lstm_quantile")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--skip-results-table", action="store_true",
                        help="persist predictions without adding an experiment row")
    args = parser.parse_args()

    checkpoint = torch.load(cfg.CHECKPOINTS_DIR / args.checkpoint, map_location="cpu", weights_only=False)
    table = load_development_model_table()
    # Every eligible 2025 day, exactly like evaluate_transformer.
    _, validation, _, _ = prepare_datasets(table, validation_every_n_days=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuantileLSTM(**checkpoint["hyperparameters"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_y, all_q = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # Counts cannot be negative; clamping keeps the quantiles in order.
            quantiles = model(**{k: batch[k] for k in MODEL_INPUTS}).clamp(min=0)
            all_q.append(quantiles.cpu().numpy().reshape(-1, len(cfg.QUANTILES)))
            all_y.append(batch["future_values"].cpu().numpy().reshape(-1))

    result = {"model": args.model_name, **score(np.concatenate(all_y), np.concatenate(all_q))}
    result.update(validation_window_summary(table))
    result["excluded_warmup_rows"] = 0
    prediction = validation_metadata(table, validation.pairs)
    if len(prediction) != result["n_observations"]:
        raise RuntimeError(f"prediction metadata has {len(prediction)} rows, expected {result['n_observations']}")
    prediction[[f"q{int(q * 100):02d}" for q in cfg.QUANTILES]] = np.concatenate(all_q)
    cfg.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.PREDICTIONS_DIR / f"{args.model_name}_validation_2025.parquet"
    prediction.to_parquet(path, index=False)
    print(f"{len(prediction):,} prediction rows -> {path}")
    if not args.skip_results_table:
        save_result(result)


if __name__ == "__main__":
    main()