from __future__ import annotations

import torch


def to_weight_mask(reference: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones_like(reference, dtype=reference.dtype)
    if valid_mask.dtype == torch.bool:
        return valid_mask.to(dtype=reference.dtype)
    return valid_mask.to(dtype=reference.dtype)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w = torch.clamp(weights, min=0.0)
    denom = w.sum()
    if float(denom.item()) <= 0.0:
        return values.new_tensor(0.0)
    return (values * w).sum() / denom


def mask_activation_ratio(weights: torch.Tensor, valid_mask: torch.Tensor | None = None) -> float:
    valid = to_weight_mask(weights, valid_mask)
    denom = float(valid.sum().item())
    if denom <= 0.0:
        return 0.0
    numer = float(torch.clamp(weights, min=0.0).sum().item())
    return numer / denom
