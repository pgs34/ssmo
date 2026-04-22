from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torchdiffeq import odeint
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torchdiffeq is required for Neural ODE models. Install it in the active environment."
    ) from exc


def _make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported Neural ODE activation: {name}")


class ODEFunc(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, activation: str = "silu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            _make_activation(activation),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return self.net(x)


class ODEBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        activation: str = "silu",
        solver: str = "rk4",
        steps: int = 4,
    ) -> None:
        super().__init__()
        self.func = ODEFunc(dim=dim, hidden_dim=hidden_dim, activation=activation)
        self.solver = solver
        self.steps = max(2, int(steps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        integration_time = torch.linspace(
            0.0,
            1.0,
            self.steps,
            device=x.device,
            dtype=x.dtype,
        )
        out = odeint(self.func, x, integration_time, method=self.solver)
        return out[-1]
