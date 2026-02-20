from __future__ import annotations

import torch

from .common import to_weight_mask


def directional_weights(
    supervised_1: torch.Tensor,
    supervised_2: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    # DML: emphasize mutual learning where the peer is currently better.
    peer_advantage = supervised_2 - supervised_1 - margin
    w_imitate_1 = torch.sigmoid(peer_advantage)
    w_imitate_2 = torch.sigmoid(-peer_advantage)
    valid = to_weight_mask(supervised_1, valid_mask)
    return w_imitate_1 * valid, w_imitate_2 * valid
