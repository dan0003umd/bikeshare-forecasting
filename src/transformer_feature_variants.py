"""Longer-context transformer variants with the XGBoost covariate set only."""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import pandas as pd

from . import config as cfg
from .baselines import save_predictions, save_result
from .forecast_windows import validation_window_summary
from .metrics import score
from .splits import load_development_model_table, split_mask


LAGS = (1, 2, 3, 24, 168)
PREDICTION_LENGTH = 24


def set_seed(seed: int = cfg.SEED) -> None:
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calendar_features(table: pd.DataFrame) -> np.ndarray:
    """Exactly XGBoost's non-demand covariates, without weather."""
    hour, weekday, month = table.hour.to_numpy(float), table.weekday.to_numpy(float), table.month.to_numpy(float)
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7),
        np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
        table.is_us_holiday.to_numpy(float), table.event_flag.to_numpy(float),
    ]).astype("float32")


def daily_starts_with_context(frame: pd.DataFrame, split: str, past_length: int) -> np.ndarray:
    local_date = frame.ts_utc.dt.tz_convert(cfg.LOCAL_TZ).dt.date
    eligible = split_mask(frame.ts_utc, split)
    counts = frame[eligible].assign(local_date=local_date[eligible]).groupby("local_date").size()
    dates = counts.index[counts.eq(24)]
    starts = frame.index[eligible & local_date.isin(dates) & frame.hour.eq(0)].to_numpy()
    # Every context token must itself have all five XGBoost lags available.
    return starts[starts >= past_length + max(LAGS)]


class VariantDataset:
    def __init__(self, values: np.ndarray, features: np.ndarray, pairs: list[tuple[int, int]], past_length: int):
        self.values, self.features, self.pairs, self.past_length = values, features, pairs, past_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        import torch
        station, start = self.pairs[index]
        positions = np.arange(start - self.past_length, start)
        lagged_values = np.column_stack([self.values[station, positions - lag] for lag in LAGS])
        return {
            "past_lags": torch.tensor(lagged_values, dtype=torch.float32),
            "past_features": torch.tensor(self.features[start - self.past_length:start]),
            "future_features": torch.tensor(self.features[start:start + PREDICTION_LENGTH]),
            "future_values": torch.tensor(self.values[station, start:start + PREDICTION_LENGTH]),
            "station": torch.tensor(station, dtype=torch.long),
        }


def prepare_datasets(table: pd.DataFrame, context_length: int):
    past_length = context_length + max(LAGS)
    stations = sorted(table.station_id.unique())
    frames = [table[table.station_id.eq(station)].sort_values("ts_utc").reset_index(drop=True) for station in stations]
    values = np.stack([frame.checkouts.to_numpy("float32") for frame in frames])
    features = calendar_features(frames[0])
    train_pairs, validation_pairs = [], []
    for station, frame in enumerate(frames):
        train_pairs.extend((station, int(start)) for start in daily_starts_with_context(frame, "train", past_length))
        validation_pairs.extend((station, int(start)) for start in daily_starts_with_context(frame, "validation", past_length))
    return (VariantDataset(values, features, train_pairs, past_length), VariantDataset(values, features, validation_pairs, past_length),
            frames, len(stations), features.shape[1], past_length)


def make_model(n_stations: int, n_features: int, head: str, context_length: int):
    import torch
    from torch import nn

    class SharedLagTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_kind = head
            self.station_embedding = nn.Embedding(n_stations, 16)
            self.past_projection = nn.Linear(len(LAGS) + n_features, 48)
            self.future_projection = nn.Linear(n_features, 48)
            self.position = nn.Parameter(torch.zeros(1, context_length + max(LAGS), 48))
            layer = nn.TransformerEncoderLayer(d_model=48, nhead=4, dim_feedforward=96, dropout=.1,
                                               batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Sequential(nn.Linear(48, 64), nn.GELU(), nn.Linear(64, 2 if head == "nb" else len(cfg.QUANTILES)))

        def forward(self, past_lags, past_features, future_features, station):
            x = torch.cat([past_lags, past_features], dim=-1)
            encoded = self.encoder(self.past_projection(x) + self.position[:, :x.shape[1]])[:, -1:]
            station_vector = self.station_embedding(station).unsqueeze(1)
            station_vector = torch.nn.functional.pad(station_vector, (0, 48 - station_vector.shape[-1]))
            decoded = encoded + self.future_projection(future_features) + station_vector
            raw = self.head(decoded)
            if self.head_kind == "nb":
                return torch.nn.functional.softplus(raw[..., 0]) + 1e-4, torch.nn.functional.softplus(raw[..., 1]) + 1e-4
            return torch.sort(raw, dim=-1).values

    return SharedLagTransformer()


def loss_for(head: str, output, target):
    import torch
    if head == "nb":
        mean, alpha = output
        total_count = 1 / alpha
        # PyTorch's ``probs`` is the success probability, whereas SciPy's
        # nbinom CDF below receives the complementary failure probability.
        probability = mean / (total_count + mean)
        return -torch.distributions.NegativeBinomial(total_count=total_count, probs=probability).log_prob(target).mean()
    error = target.unsqueeze(-1) - output
    quantiles = torch.tensor(cfg.QUANTILES, device=output.device, dtype=output.dtype)
    return torch.maximum(quantiles * error, (quantiles - 1) * error).mean()


def train_epoch(model, loader, optimizer, device, head):
    import torch
    training = optimizer is not None
    model.train(training)
    losses = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(training):
            output = model(batch["past_lags"], batch["past_features"], batch["future_features"], batch["station"])
            loss = loss_for(head, output, batch["future_values"])
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train(args) -> None:
    import torch
    from torch.utils.data import DataLoader
    set_seed(args.seed)
    table = load_development_model_table()
    train_set, validation_set, _, n_stations, n_features, past_length = prepare_datasets(table, args.context_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(n_stations, n_features, args.head, args.context_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    cfg.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True); cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    history, best, stale, started = [], float("inf"), 0, time.monotonic()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, args.head)
        validation_loss = train_epoch(model, validation_loader, None, device, args.head)
        row = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss, "head": args.head,
               "context_length": args.context_length, "train_windows": len(train_set), "validation_windows": len(validation_set)}
        history.append(row); print(row)
        payload = {"state_dict": model.state_dict(), "head": args.head, "context_length": args.context_length,
                   "n_stations": n_stations, "n_features": n_features, "past_length": past_length, "run_name": args.run_name}
        torch.save(payload, cfg.CHECKPOINTS_DIR / f"{args.run_name}_last.pt")
        if validation_loss < best:
            best, stale = validation_loss, 0; torch.save(payload, cfg.CHECKPOINTS_DIR / f"{args.run_name}_best.pt")
        else:
            stale += 1
        if stale >= args.patience or (time.monotonic() - started) / 60 >= args.max_minutes:
            print(f"stopping: stale_epochs={stale}, elapsed_minutes={(time.monotonic() - started) / 60:.2f}")
            break
    pd.DataFrame(history).to_csv(cfg.REPORTS_DIR / f"{args.run_name}_loss.csv", index=False)


def evaluate(args) -> None:
    import torch
    from scipy.stats import nbinom
    from torch.utils.data import DataLoader
    set_seed(args.seed)
    checkpoint = torch.load(cfg.CHECKPOINTS_DIR / args.checkpoint, map_location="cpu", weights_only=False)
    table = load_development_model_table()
    _, validation_set, frames, n_stations, n_features, _ = prepare_datasets(table, checkpoint["context_length"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(n_stations, n_features, checkpoint["head"], checkpoint["context_length"]).to(device)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    actuals, predictions = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(batch["past_lags"], batch["past_features"], batch["future_features"], batch["station"])
            if checkpoint["head"] == "nb":
                mean, alpha = [x.detach().cpu().numpy().reshape(-1) for x in output]
                count = 1 / alpha; probability = count / (count + mean)
                predicted = nbinom.ppf(np.asarray(cfg.QUANTILES)[None, :], count[:, None], probability[:, None])
            else:
                predicted = output.detach().cpu().numpy().reshape(-1, len(cfg.QUANTILES))
            actuals.append(batch["future_values"].detach().cpu().numpy().reshape(-1)); predictions.append(predicted)
    prediction = np.concatenate(predictions); actual = np.concatenate(actuals)
    pairs = validation_set.pairs
    pieces = [frames[station].iloc[start:start + PREDICTION_LENGTH][["station_id", "ts_utc"]] for station, start in pairs]
    metadata = pd.concat(pieces, ignore_index=True)
    if len(actual) != 435_600 or len(metadata) != len(actual):
        raise RuntimeError(f"expected 435,600 aligned validation rows, got {len(actual):,} actual and {len(metadata):,} metadata")
    valid = pd.Series(False, index=table.index)
    # save_predictions supports a validation-only array when passed an aligned table slice.
    output = metadata.copy(); output["actual"] = actual
    output[[f"q{int(q * 100):02d}" for q in cfg.QUANTILES]] = prediction
    cfg.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    output.to_parquet(cfg.PREDICTIONS_DIR / f"{args.model_name}_validation_2025.parquet", index=False)
    result = {"model": args.model_name, **score(actual, prediction)}
    result.update(validation_window_summary(table)); result["excluded_warmup_rows"] = 0
    save_result(result)
    print(f"{len(output):,} prediction rows -> {cfg.PREDICTIONS_DIR}")


def discard(args) -> None:
    """Remove an explicitly discarded exploratory result from the tracked table."""
    path = cfg.REPORTS_DIR / "results_validation_2025.csv"
    results = pd.read_csv(path)
    filtered = results.loc[results.model.ne(args.model_name)].copy()
    if len(filtered) == len(results):
        raise ValueError(f"result row not found: {args.model_name}")
    filtered.to_csv(path, index=False)
    print(f"discarded {args.model_name} from {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--head", choices=["nb", "quantile"], required=True)
    train_parser.add_argument("--context-length", type=int, default=336)
    train_parser.add_argument("--run-name", required=True)
    train_parser.add_argument("--epochs", type=int, default=12)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--patience", type=int, default=2)
    train_parser.add_argument("--max-minutes", type=float, default=25)
    train_parser.add_argument("--seed", type=int, default=cfg.SEED)
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--model-name", required=True)
    eval_parser.add_argument("--batch-size", type=int, default=32)
    eval_parser.add_argument("--seed", type=int, default=cfg.SEED)
    discard_parser = sub.add_parser("discard")
    discard_parser.add_argument("--model-name", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        discard(args)


if __name__ == "__main__":
    main()
