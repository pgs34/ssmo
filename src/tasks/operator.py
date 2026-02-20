from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset


@dataclass
class OperatorDataConfig:
    data_root: str = "data"
    dataset_name: str = "burgers"  # burgers, darcy, navier_stokes
    batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = True
    seed: int = 0
    download: bool = True

    burgers_file_name: str = "burgers_data_R10.mat"
    burgers_subsample_stride: int = 8
    burgers_n_train: int = 1000
    burgers_n_test: int = 100

    # neuraloperator dataset options
    n_train: int = 1000
    n_test: int = 100

    darcy_train_resolution: int = 64
    darcy_test_resolution: int = 64
    darcy_use_shift_split: bool = True
    darcy_shift_split_ratio: float = 0.8

    navier_train_resolution: int = 128
    navier_test_resolution: int = 128


def build_operator_dataloaders(config: OperatorDataConfig):
    name = config.dataset_name.lower()
    if name == "burgers":
        return _build_burgers(config)
    if name == "darcy":
        return _build_darcy(config)
    if name in {"navier_stokes", "navierstokes", "ns"}:
        return _build_navier_stokes(config)
    raise ValueError(f"Unsupported operator dataset: '{config.dataset_name}'")


def _build_burgers(config: OperatorDataConfig):
    path = Path(config.data_root) / config.burgers_file_name
    if not path.exists():
        raise FileNotFoundError(
            f"Burgers file not found: '{path}'. "
            "Expected burgers_data_R10.mat under data/. "
            "You can place it manually or use gdown."
        )

    try:
        data = sio.loadmat(str(path))
    except Exception as exc:
        raise ValueError(
            f"Failed to read Burgers file '{path}'. "
            "The file is not a valid MATLAB .mat (or is corrupted). "
            "Re-download burgers_data_R10.mat and retry."
        ) from exc
    if "a" not in data or "u" not in data:
        raise KeyError("Burgers .mat must include keys 'a' and 'u'.")

    stride = max(1, int(config.burgers_subsample_stride))
    x = torch.tensor(data["a"], dtype=torch.float32)[:, ::stride].unsqueeze(1)
    y = torch.tensor(data["u"], dtype=torch.float32)[:, ::stride].unsqueeze(1)

    n_train = int(config.burgers_n_train)
    n_test = int(config.burgers_n_test)
    if n_train + n_test > x.shape[0]:
        raise ValueError(
            f"Requested n_train+n_test ({n_train+n_test}) exceeds dataset size ({x.shape[0]})."
        )

    train_set = TensorDataset(x[:n_train], y[:n_train])
    test_set = TensorDataset(x[n_train : n_train + n_test], y[n_train : n_train + n_test])

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
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
        "val_loader": test_loader,
        "meta": {
            "dataset": "burgers",
            "in_channels": int(x.shape[1]),
            "out_channels": int(y.shape[1]),
            "resolution": int(x.shape[-1]),
        },
    }


def _build_darcy(config: OperatorDataConfig):
    DarcyDataset, _ = _import_neuralop_datasets()
    root_dir = Path(config.data_root) / "darcy"

    dataset = DarcyDataset(
        root_dir=str(root_dir),
        n_train=config.n_train,
        n_tests=[config.n_test],
        batch_size=config.batch_size,
        test_batch_sizes=[config.batch_size],
        train_resolution=config.darcy_train_resolution,
        test_resolutions=[config.darcy_test_resolution],
        download=config.download,
    )

    if config.darcy_use_shift_split:
        train_set, shifted_test_set = _split_subset(
            dataset.train_db,
            split_ratio=config.darcy_shift_split_ratio,
            seed=config.seed,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        val_loader = DataLoader(
            shifted_test_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
    else:
        train_loader = DataLoader(
            dataset.train_db,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        val_loader = DataLoader(
            dataset.test_dbs[config.darcy_test_resolution],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

    in_channels, out_channels = _infer_io_channels(train_loader)
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "meta": {
            "dataset": "darcy",
            "in_channels": in_channels,
            "out_channels": out_channels,
            "train_resolution": config.darcy_train_resolution,
            "test_resolution": config.darcy_test_resolution,
        },
    }


def _build_navier_stokes(config: OperatorDataConfig):
    _, NavierStokesDataset = _import_neuralop_datasets()
    root_dir = Path(config.data_root) / "navier_stokes"

    dataset = NavierStokesDataset(
        root_dir=str(root_dir),
        n_train=config.n_train,
        n_tests=[config.n_test],
        batch_size=config.batch_size,
        test_batch_sizes=[config.batch_size],
        train_resolution=config.navier_train_resolution,
        test_resolutions=[config.navier_test_resolution],
        download=config.download,
    )

    train_loader = DataLoader(
        dataset.train_db,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        dataset.test_dbs[config.navier_test_resolution],
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    in_channels, out_channels = _infer_io_channels(train_loader)
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "meta": {
            "dataset": "navier_stokes",
            "in_channels": in_channels,
            "out_channels": out_channels,
            "train_resolution": config.navier_train_resolution,
            "test_resolution": config.navier_test_resolution,
        },
    }


def _split_subset(dataset, split_ratio: float, seed: int):
    n = len(dataset)
    n_train = int(n * split_ratio)
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=g)
    train_indices = indices[:n_train].tolist()
    test_indices = indices[n_train:].tolist()
    return Subset(dataset, train_indices), Subset(dataset, test_indices)


def _infer_io_channels(loader: DataLoader) -> tuple[Optional[int], Optional[int]]:
    batch = next(iter(loader))
    x, y = None, None
    if isinstance(batch, dict):
        x = batch.get("x", batch.get("input", None))
        y = batch.get("y", batch.get("output", None))
    elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
        x, y = batch[0], batch[1]
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        if x.ndim >= 2 and y.ndim >= 2:
            return int(x.shape[1]), int(y.shape[1])
    return None, None


def _import_neuralop_datasets():
    try:
        from neuralop.data.datasets import DarcyDataset, NavierStokesDataset
    except Exception as exc:
        raise ImportError(
            "Failed to import neuralop datasets. "
            "Install and validate neuraloperator/neuralop in your active environment."
        ) from exc
    return DarcyDataset, NavierStokesDataset
