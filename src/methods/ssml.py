from __future__ import annotations

import torch

from .common import to_weight_mask


def directional_weights(
    supervised_1: torch.Tensor,
    supervised_2: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    # model-1 imitates model-2 only where model-2 is better.
    # model-2 imitates model-1 only where model-1 is better.
    valid = to_weight_mask(supervised_1, valid_mask)

    better_1 = (supervised_1 + margin < supervised_2).to(dtype=supervised_1.dtype) * valid
    better_2 = (supervised_2 + margin < supervised_1).to(dtype=supervised_1.dtype) * valid

    w_imitate_1 = better_2
    w_imitate_2 = better_1
    return w_imitate_1, w_imitate_2
