"""Train a small shared station-embedding transformer without touching 2026 labels."""
from __future__ import annotations

import argparse
import random

import numpy as np
import pandas as pd

from . import config as cfg
from .transformer_model import PAST_LENGTH, PREDICTION_LENGTH, daily_starts, time_features, transformer_config


def set_seed(seed=cfg.SEED):
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class WindowDataset:
    def __init__(self, values, features, pairs):
        self.values, self.features, self.pairs = values, features, pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        import torch
        station, start = self.pairs[i]
        past = self.values[station, start - PAST_LENGTH:start]
        future = self.values[station, start:start + PREDICTION_LENGTH]
        return {"past_values": torch.tensor(past), "past_time_features": torch.tensor(self.features[start - PAST_LENGTH:start]),
                "past_observed_mask": torch.ones(PAST_LENGTH), "future_values": torch.tensor(future),
                "future_time_features": torch.tensor(self.features[start:start + PREDICTION_LENGTH]),
                "static_categorical_features": torch.tensor([station], dtype=torch.long)}


def prepare_datasets(table: pd.DataFrame, train_every_n_days: int = 7, validation_every_n_days: int = 7):
    stations = sorted(table.station_id.unique())
    station_frames = [table[table.station_id.eq(station)].sort_values("ts_utc").reset_index(drop=True) for station in stations]
    # All selected stations share a time axis by construction.
    values = np.stack([f.checkouts.to_numpy("float32") for f in station_frames])
    features = time_features(station_frames[0])
    train_pairs, val_pairs = [], []
    for station, frame in enumerate(station_frames):
        train_pairs.extend((station, int(start)) for start in daily_starts(frame, "train", every_n_days=train_every_n_days)
                           if start + PREDICTION_LENGTH <= len(frame))
        val_pairs.extend((station, int(start)) for start in daily_starts(frame, "validation", every_n_days=validation_every_n_days)
                         if start + PREDICTION_LENGTH <= len(frame))
    return WindowDataset(values, features, train_pairs), WindowDataset(values, features, val_pairs), len(stations), features.shape[1]


def run_epoch(model, loader, optimizer, device):
    import torch
    train = optimizer is not None
    model.train(train)
    losses = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            loss = model(**batch).loss
            if train:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from transformers import TimeSeriesTransformerForPrediction
    set_seed()
    table = pd.read_parquet(cfg.PROCESSED_DIR / "model_table.parquet")
    train_set, val_set, n_stations, n_features = prepare_datasets(table)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformerForPrediction(transformer_config(n_stations, n_features)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True); cfg.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        val_loss = run_epoch(model, val_loader, None, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        print(history[-1])
        torch.save({"state_dict": model.state_dict(), "config": model.config.to_dict(), "epoch": epoch},
                   cfg.CHECKPOINTS_DIR / "transformer_last.pt")
    pd.DataFrame(history).to_csv(cfg.REPORTS_DIR / "transformer_loss.csv", index=False)


if __name__ == "__main__":
    main()
