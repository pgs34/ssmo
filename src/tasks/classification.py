from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


@dataclass
class ClassificationDataConfig:
    data_root: str = "data"
    dataset_name: str = "cifar10"  # mnist, cifar10, cifar100
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    download: bool = True
    train_augment: bool = True
    train_subset_size: Optional[int] = None
    val_subset_size: Optional[int] = None
    seed: int = 0

    # Label noise: none, symmetric, asymmetric
    label_noise_type: Optional[str] = None
    label_noise_rate: float = 0.0


class LabelNoiseDataset(Dataset):
    def __init__(self, base_dataset: Dataset, noisy_targets: np.ndarray) -> None:
        self.base_dataset = base_dataset
        self.noisy_targets = noisy_targets.astype(np.int64)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        x, _ = self.base_dataset[idx]
        y = int(self.noisy_targets[idx])
        return x, y


def build_classification_dataloaders(config: ClassificationDataConfig):
    train_set, val_set, num_classes = _build_base_datasets(config)

    if config.label_noise_type and config.label_noise_rate > 0.0:
        targets = _extract_targets(train_set)
        noisy_targets = _inject_label_noise(
            targets=targets,
            num_classes=num_classes,
            dataset_name=config.dataset_name,
            noise_type=config.label_noise_type,
            noise_rate=config.label_noise_rate,
            seed=config.seed,
        )
        train_set = LabelNoiseDataset(train_set, noisy_targets)

    if config.train_subset_size is not None:
        train_set = _subset_dataset(train_set, config.train_subset_size, config.seed)
    if config.val_subset_size is not None:
        val_set = _subset_dataset(val_set, config.val_subset_size, config.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
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

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "meta": {
            "dataset": config.dataset_name.lower(),
            "num_classes": num_classes,
            "label_noise_type": config.label_noise_type or "none",
            "label_noise_rate": float(config.label_noise_rate),
        },
    }


def _build_base_datasets(config: ClassificationDataConfig):
    name = config.dataset_name.lower()
    if name == "mnist":
        mean, std = (0.1307,), (0.3081,)
        train_tf = [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean, std)]
        if config.train_augment:
            train_tf = [
                transforms.Resize((32, 32)),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        test_tf = [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean, std)]
        train_set = datasets.MNIST(
            root=config.data_root,
            train=True,
            download=config.download,
            transform=transforms.Compose(train_tf),
        )
        val_set = datasets.MNIST(
            root=config.data_root,
            train=False,
            download=config.download,
            transform=transforms.Compose(test_tf),
        )
        return train_set, val_set, 10

    if name == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        train_tf = [transforms.ToTensor(), transforms.Normalize(mean, std)]
        if config.train_augment:
            train_tf = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        test_tf = [transforms.ToTensor(), transforms.Normalize(mean, std)]
        train_set = datasets.CIFAR10(
            root=config.data_root,
            train=True,
            download=config.download,
            transform=transforms.Compose(train_tf),
        )
        val_set = datasets.CIFAR10(
            root=config.data_root,
            train=False,
            download=config.download,
            transform=transforms.Compose(test_tf),
        )
        return train_set, val_set, 10

    if name == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        train_tf = [transforms.ToTensor(), transforms.Normalize(mean, std)]
        if config.train_augment:
            train_tf = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        test_tf = [transforms.ToTensor(), transforms.Normalize(mean, std)]
        train_set = datasets.CIFAR100(
            root=config.data_root,
            train=True,
            download=config.download,
            transform=transforms.Compose(train_tf),
        )
        val_set = datasets.CIFAR100(
            root=config.data_root,
            train=False,
            download=config.download,
            transform=transforms.Compose(test_tf),
        )
        return train_set, val_set, 100

    raise ValueError(f"Unsupported classification dataset: '{config.dataset_name}'")


def _extract_targets(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            return targets.cpu().numpy().astype(np.int64)
        return np.asarray(targets, dtype=np.int64)
    raise AttributeError("Dataset does not expose `targets`, cannot inject noise.")


def _subset_dataset(dataset: Dataset, subset_size: int, seed: int):
    n = len(dataset)
    if subset_size <= 0:
        raise ValueError(f"subset_size must be > 0, got {subset_size}")
    if subset_size >= n:
        return dataset
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=g)[:subset_size].tolist()
    return Subset(dataset, indices)


def _inject_label_noise(
    targets: np.ndarray,
    num_classes: int,
    dataset_name: str,
    noise_type: str,
    noise_rate: float,
    seed: int,
) -> np.ndarray:
    n = len(targets)
    if not (0.0 <= noise_rate < 1.0):
        raise ValueError(f"noise_rate must satisfy 0 <= rate < 1, got {noise_rate}")
    if noise_rate == 0.0:
        return targets.copy()

    rng = np.random.default_rng(seed)
    noisy = targets.copy()
    noisy_count = int(round(n * noise_rate))
    noisy_indices = rng.choice(n, size=noisy_count, replace=False)
    noise_name = noise_type.lower()

    if noise_name == "symmetric":
        random_labels = rng.integers(low=0, high=num_classes, size=noisy_count, endpoint=False)
        same_mask = random_labels == noisy[noisy_indices]
        random_labels[same_mask] = (random_labels[same_mask] + 1) % num_classes
        noisy[noisy_indices] = random_labels
        return noisy

    if noise_name == "asymmetric":
        mapping = _asymmetric_mapping(dataset_name.lower(), num_classes)
        for idx in noisy_indices:
            label = int(noisy[idx])
            noisy[idx] = mapping[label]
        return noisy

    raise ValueError(f"Unsupported label_noise_type: '{noise_type}'")


def _asymmetric_mapping(dataset_name: str, num_classes: int) -> np.ndarray:
    mapping = np.arange(num_classes, dtype=np.int64)
    if dataset_name == "cifar10":
        # Standard class-dependent mapping used in noisy CIFAR setups.
        mapping[9] = 1   # truck -> automobile
        mapping[2] = 0   # bird -> airplane
        mapping[3] = 5   # cat -> dog
        mapping[5] = 3   # dog -> cat
        mapping[4] = 7   # deer -> horse
        return mapping
    if dataset_name == "mnist":
        mapping[7] = 1
        mapping[2] = 7
        mapping[5] = 6
        mapping[6] = 5
        mapping[3] = 8
        return mapping
    # Fallback for CIFAR100 and others: deterministic cyclic flip.
    return (mapping + 1) % num_classes
