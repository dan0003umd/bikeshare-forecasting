"""Train the quantile LSTM on 2023-2024 daily windows; early-stop on 2025 validation pinball loss."""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config as cfg
from .lstm_model import QuantileLSTM, pinball_loss
from .splits import load_development_model_table
from .train_transformer import prepare_datasets, set_seed

MODEL_INPUTS = ("past_values", "past_time_features", "future_time_features", "static_categorical_features")


def run_epoch(model, loader, quantiles, device, optimizer=None, max_batches=None):
    """One pass over the loader. With an optimizer it trains; without one it only scores."""
    train = optimizer is not None
    model.train(train)
    losses = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            prediction = model(**{k: batch[k] for k in MODEL_INPUTS})
            loss = pinball_loss(batch["future_values"], prediction, quantiles)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--max-minutes", type=float, default=120)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="smoke test only: stop each epoch after this many batches")
    parser.add_argument("--run-name", default="lstm_quantile")
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    table = load_development_model_table()
    # Same windows as the daily transformers: train on every 2023-2024 day,
    # monitor on weekly 2025 windows (the full daily 2025 scoring happens in evaluate_lstm).
    train_set, val_set, n_stations, n_features = prepare_datasets(
        table, train_every_n_days=1, validation_every_n_days=7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyperparameters = {"n_stations": n_stations, "n_features": n_features,
                       "hidden_size": args.hidden_size, "num_layers": args.num_layers}
    model = QuantileLSTM(**hyperparameters).to(device)
    quantiles = torch.tensor(cfg.QUANTILES, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"device={device}; train windows={len(train_set):,}; validation windows={len(val_set):,}; "
          f"parameters={sum(p.numel() for p in model.parameters()):,}")

    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    history, best_loss, stale_epochs, started = [], float("inf"), 0, time.monotonic()
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, quantiles, device, optimizer, args.max_batches)
        val_loss = run_epoch(model, val_loader, quantiles, device, None, args.max_batches)
        elapsed = (time.monotonic() - started) / 60
        history.append({"epoch": epoch, "train_pinball": train_loss, "validation_pinball": val_loss,
                        "elapsed_minutes": round(elapsed, 2)})
        print(history[-1])
        payload = {"state_dict": model.state_dict(), "hyperparameters": hyperparameters,
                   "epoch": epoch, "run_name": args.run_name, "seed": args.seed}
        torch.save(payload, cfg.CHECKPOINTS_DIR / f"{args.run_name}_last.pt")
        if val_loss < best_loss:
            best_loss, stale_epochs = val_loss, 0
            torch.save(payload, cfg.CHECKPOINTS_DIR / f"{args.run_name}_best.pt")
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience or elapsed >= args.max_minutes:
            print(f"stopping: stale_epochs={stale_epochs}, elapsed_minutes={elapsed:.1f}")
            break
    pd.DataFrame(history).to_csv(cfg.REPORTS_DIR / f"{args.run_name}_loss.csv", index=False)


if __name__ == "__main__":
    main()