from __future__ import annotations

import torch

from .common import to_weight_mask


def directional_weights(
    supervised_1: torch.Tensor,
    supervised_2: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    del supervised_2, margin
    base = to_weight_mask(supervised_1, valid_mask)
    return base, base
