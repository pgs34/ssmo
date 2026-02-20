from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class TimeSeriesDataConfig:
    data_root: str = "data/time_series"
    dataset_name: str = "etth1"  # etth1, ettm1, electricity, weather
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True
    sequence_length: int = 96
    prediction_length: int = 24
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    normalize: bool = True
    feature_mode: str = "multivariate"  # multivariate or univariate
    target_column: Optional[str] = None
    time_column: str = "date"
    shuffle_train: bool = True


class TimeSeriesWindowDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def build_time_series_dataloaders(config: TimeSeriesDataConfig):
    csv_path = _resolve_csv_path(config)
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: '{csv_path}'")

    numeric_df = _select_numeric_frame(df, config.time_column)
    values = numeric_df.to_numpy(dtype=np.float32)
    target_indices = _resolve_target_indices(numeric_df, config)

    n_total = len(values)
    n_train = int(n_total * config.train_ratio)
    n_val = int(n_total * config.val_ratio)
    n_test = n_total - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Invalid split sizes from ratios. train={n_train}, val={n_val}, test={n_test}"
        )

    if config.normalize:
        values = _normalize_by_train_stats(values, n_train)

    seq_len = int(config.sequence_length)
    pred_len = int(config.prediction_length)
    if seq_len <= 0 or pred_len <= 0:
        raise ValueError("sequence_length and prediction_length must be positive.")

    train_x, train_y = _build_windows(values, 0, n_train, seq_len, pred_len, target_indices)
    val_start = max(0, n_train - seq_len)
    val_end = n_train + n_val
    val_x, val_y = _build_windows(values, val_start, val_end, seq_len, pred_len, target_indices)
    test_start = max(0, val_end - seq_len)
    test_x, test_y = _build_windows(values, test_start, n_total, seq_len, pred_len, target_indices)

    train_set = TimeSeriesWindowDataset(train_x, train_y)
    val_set = TimeSeriesWindowDataset(val_x, val_y)
    test_set = TimeSeriesWindowDataset(test_x, test_y)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "meta": {
            "dataset": config.dataset_name.lower(),
            "num_features": int(values.shape[1]),
            "num_targets": int(len(target_indices)),
            "sequence_length": seq_len,
            "prediction_length": pred_len,
            "csv_path": str(csv_path),
        },
    }


def _resolve_csv_path(config: TimeSeriesDataConfig) -> Path:
    name = config.dataset_name.lower()
    root = Path(config.data_root)
    mapping = {
        "etth1": root / "ETT-small" / "ETTh1.csv",
        "etth2": root / "ETT-small" / "ETTh2.csv",
        "ettm1": root / "ETT-small" / "ETTm1.csv",
        "ettm2": root / "ETT-small" / "ETTm2.csv",
        "electricity": root / "electricity" / "electricity.csv",
        "weather": root / "weather" / "weather.csv",
        "traffic": root / "traffic" / "traffic.csv",
        "exchange_rate": root / "exchange_rate" / "exchange_rate.csv",
        "illness": root / "illness" / "national_illness.csv",
    }
    if name not in mapping:
        raise ValueError(f"Unsupported time-series dataset: '{config.dataset_name}'")
    csv_path = mapping[name]
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file: '{csv_path}'")
    return csv_path


def _select_numeric_frame(df: pd.DataFrame, time_column: str) -> pd.DataFrame:
    frame = df.copy()
    if time_column in frame.columns:
        frame = frame.drop(columns=[time_column])
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        raise ValueError("No numeric columns found after dropping time column.")
    return numeric


def _resolve_target_indices(frame: pd.DataFrame, config: TimeSeriesDataConfig):
    if config.feature_mode.lower() == "univariate":
        target_col = config.target_column if config.target_column else frame.columns[-1]
        if target_col not in frame.columns:
            raise ValueError(f"target_column '{target_col}' not found in columns.")
        return [int(frame.columns.get_loc(target_col))]

    if config.target_column:
        if config.target_column not in frame.columns:
            raise ValueError(f"target_column '{config.target_column}' not found in columns.")
        return [int(frame.columns.get_loc(config.target_column))]

    return list(range(frame.shape[1]))


def _normalize_by_train_stats(values: np.ndarray, n_train: int) -> np.ndarray:
    train_values = values[:n_train]
    mean = train_values.mean(axis=0, keepdims=True)
    std = train_values.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (values - mean) / std


def _build_windows(
    values: np.ndarray,
    start: int,
    end: int,
    seq_len: int,
    pred_len: int,
    target_indices: list[int],
):
    if end - start < seq_len + pred_len:
        raise ValueError(
            f"Not enough points to build windows in range [{start}, {end}) "
            f"with seq_len={seq_len}, pred_len={pred_len}"
        )
    x_list = []
    y_list = []
    last_start = end - seq_len - pred_len + 1
    for i in range(start, last_start):
        seq_x = values[i : i + seq_len]
        seq_y = values[i + seq_len : i + seq_len + pred_len, target_indices]
        x_list.append(seq_x)
        y_list.append(seq_y)
    x = torch.tensor(np.stack(x_list, axis=0), dtype=torch.float32)
    y = torch.tensor(np.stack(y_list, axis=0), dtype=torch.float32)
    return x, y

