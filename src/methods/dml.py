from __future__ import annotations

import torch

from .common import to_weight_mask


def directional_weights(
    supervised_1: torch.Tensor,
    supervised_2: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = to_weight_mask(supervised_1, valid_mask)

    # DML: apply a soft gate only where the peer is better than the student.
    # The gate stays in [0, 1), so it sits between naive(=1 everywhere) and
    # studygroup(=hard 0/1 selection).
    def _soft_gate(advantage: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.sigmoid(advantage) * 2.0 - 1.0, min=0.0)

    w_imitate_1 = _soft_gate(supervised_2 - supervised_1 - margin) * valid
    w_imitate_2 = _soft_gate(supervised_1 - supervised_2 - margin) * valid
    return w_imitate_1, w_imitate_2
