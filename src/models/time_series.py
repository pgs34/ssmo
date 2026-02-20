from __future__ import annotations

import math

import torch
import torch.nn as nn


class DLinearForecaster(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int, num_targets: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.num_targets = num_targets
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        b, l, c = x.shape
        if l != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {l}")
        h = x.transpose(1, 2)  # (B, C, L)
        out = self.linear(h)  # (B, C, P)
        out = out.transpose(1, 2)  # (B, P, C)
        if self.num_targets < c:
            out = out[:, :, : self.num_targets]
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_features: int,
        num_targets: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_targets = num_targets
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=seq_len + 8)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, pred_len * num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        b, l, _ = x.shape
        if l != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {l}")
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        z = self.encoder(h)
        pooled = z[:, -1, :]
        out = self.head(pooled).view(b, self.pred_len, self.num_targets)
        return out


def build_time_series_model(
    model_name: str,
    seq_len: int,
    pred_len: int,
    num_features: int,
    num_targets: int,
) -> nn.Module:
    name = model_name.lower()
    if name in {"dlinear", "linear"}:
        return DLinearForecaster(
            seq_len=seq_len,
            pred_len=pred_len,
            num_features=num_features,
            num_targets=num_targets,
        )
    if name in {"transformer", "ts_transformer"}:
        return TransformerForecaster(
            seq_len=seq_len,
            pred_len=pred_len,
            num_features=num_features,
            num_targets=num_targets,
            d_model=128,
            nhead=4,
            num_layers=2,
            dropout=0.1,
        )
    raise ValueError(
        f"Unsupported time-series model '{model_name}'. Use one of: dlinear, transformer"
    )

