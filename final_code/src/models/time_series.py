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
        activation: str = "relu",
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
            activation=activation,
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


class PatchTSTForecaster(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_features: int,
        num_targets: int,
        patch_len: int = 16,
        patch_stride: int = 8,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if seq_len < patch_len:
            raise ValueError(f"Patch length {patch_len} exceeds sequence length {seq_len}")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_targets = num_targets
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.num_patches = 1 + (seq_len - patch_len) // patch_stride
        patch_dim = patch_len * num_features
        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=self.num_patches + 8)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len * num_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, c = x.shape
        if l != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {l}")
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        # (B, N, C, P) -> (B, N, P * C)
        patches = patches.permute(0, 1, 3, 2).reshape(b, self.num_patches, self.patch_len * c)
        h = self.patch_proj(patches)
        h = self.pos_encoding(h)
        z = self.encoder(h)
        pooled = z.mean(dim=1)
        out = self.head(pooled).view(b, self.pred_len, self.num_targets)
        return out


class GRUForecaster(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_features: int,
        num_targets: int,
        hidden_size: int = 160,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_targets = num_targets
        self.encoder = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, pred_len * num_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        if l != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {l}")
        z, _ = self.encoder(x)
        pooled = z[:, -1, :]
        out = self.head(pooled).view(b, self.pred_len, self.num_targets)
        return out


class NeuralODEForecaster(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_features: int,
        num_targets: int,
        hidden_size: int = 192,
        num_layers: int = 2,
        dropout: float = 0.1,
        ode_steps: int = 4,
    ) -> None:
        super().__init__()
        from .neural_ode import ODEBlock

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_targets = num_targets
        self.encoder = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.ode = ODEBlock(
            dim=hidden_size,
            hidden_dim=hidden_size * 2,
            activation="silu",
            solver="rk4",
            steps=ode_steps,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, pred_len * num_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        if l != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {l}")
        z, _ = self.encoder(x)
        state = self.ode(z[:, -1, :])
        out = self.head(state).view(b, self.pred_len, self.num_targets)
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
            activation="relu",
        )
    if name in {"transformer_gelu", "ts_transformer_gelu"}:
        return TransformerForecaster(
            seq_len=seq_len,
            pred_len=pred_len,
            num_features=num_features,
            num_targets=num_targets,
            d_model=128,
            nhead=4,
            num_layers=2,
            dropout=0.1,
            activation="gelu",
        )
    if name in {"transformer_wide", "ts_transformer_wide"}:
        return TransformerForecaster(
            seq_len=seq_len,
            pred_len=pred_len,
            num_features=num_features,
            num_targets=num_targets,
            d_model=192,
            nhead=6,
            num_layers=3,
            dropout=0.1,
            activation="gelu",
        )
    if name in {"patchtst", "patch_tst"}:
        return PatchTSTForecaster(
            seq_len=seq_len,
            pred_len=pred_len,
            num_features=num_features,
            num_targets=num_targets,
            patch_len=16,
            patch_stride=8,
            d_model=128,
            nhead=4,
            num_layers=3,
            dropout=0.1,
        )
    if name in {"gru", "gru_forecaster"}:
        return GRUForecaster(
            seq_len=seq_len,
            pred_len=pred_len,
            num_features=num_features,
            num_targets=num_targets,
            hidden_size=160,
            num_layers=2,
            dropout=0.1,
        )
    if name in {"neural_ode", "ode_forecaster", "node"}:
        return NeuralODEForecaster(
            seq_len=seq_len,
            pred_len=pred_len,
            num_features=num_features,
            num_targets=num_targets,
            hidden_size=192,
            num_layers=2,
            dropout=0.1,
            ode_steps=4,
        )
    raise ValueError(
        f"Unsupported time-series model '{model_name}'. Use one of: dlinear, transformer, transformer_gelu, transformer_wide, patchtst, gru, neural_ode"
    )
