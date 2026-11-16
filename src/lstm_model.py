from __future__ import annotations

import torch
from torch import nn

from . import config as cfg

def pinball_loss(y, q_pred, quantiles):
    """Mean pinball loss, y batch 24, q_pred batch 24, 19, quantiles, 19."""
    error = y.unsqueeze(-1) - q_pred
    return torch.maximum(quantiles * error, (quantiles - 1) * error).mean()
class QuantileLSTM(nn.Module):
    def __init__(self, n_stations, n_features, n_quantiles=len(cfg.QUANTILES),
                 embedding_dim = 16, hidden_size = 64, num_layers = 2, dropout = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(n_stations, embedding_dim)
        self.encoder = nn.LSTM(1 + n_features + embedding_dim, hidden_size, num_layers,
                               batch_first = True, dropout = dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size + n_features + embedding_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, n_quantiles))
        
    def forward(self, past_values, past_time_features, future_time_features, static_categorical_features):
        horizon = future_time_features.shape[1]
        station = self.embedding(static_categorical_features.squeeze(-1))
        # Encoder: one input vector per past hour = log demand + calendar + station.
        past_y = torch.log1p(past_values).unsqueeze(-1)
        past_station = station.unsqueeze(1).expand(-1, past_values.shape[1], -1)
        _, (hidden, _) = self.encoder(torch.cat([past_y, past_time_features, past_station], dim=-1))
        # Decoder: the two-week summary + each future hour's calendar + station.
        summary = hidden[-1].unsqueeze(1).expand(-1, horizon, -1)
        future_station = station.unsqueeze(1).expand(-1, horizon, -1)
        raw = self.decoder(torch.cat([summary, future_time_features, future_station], dim=-1))
        # Non-crossing quantiles: lowest quantile, then strictly positive increments.
        lowest = raw[..., :1]
        increments = nn.functional.softplus(raw[..., 1:])
        return torch.cat([lowest, lowest + increments.cumsum(dim=-1)], dim=-1)