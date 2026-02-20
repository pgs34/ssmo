from __future__ import annotations

import torch
import torch.nn as nn

from models.deeponet import DeepONet1D, DeepONet2D
from models.fno import FNO1D, FNO2D
from models.gnot import GNOT1D, GNOT2D


def _import_neuralop_models():
    try:
        from neuralop.models import FNO as NeuralOpFNO
        from neuralop.models import TFNO as NeuralOpTFNO
        from neuralop.models import UNO as NeuralOpUNO
    except Exception as exc:
        raise ImportError(
            "Failed to import neuralop models (FNO/TFNO/UNO). "
            "Use a compatible environment with neuralop installed (Python >= 3.8). "
            "Example: conda activate ssml"
        ) from exc
    return NeuralOpFNO, NeuralOpTFNO, NeuralOpUNO


def _build_neuralop_fno(
    dataset: str,
    resolution: int,
    in_channels: int,
    out_channels: int,
    tensorized: bool,
) -> nn.Module:
    NeuralOpFNO, NeuralOpTFNO, _ = _import_neuralop_models()
    if dataset == "burgers":
        n_modes = (min(16, max(4, resolution // 8)),)
        hidden = 64
    else:
        n_modes = (12, 12)
        hidden = 96

    common_kwargs = {
        "n_modes": n_modes,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "hidden_channels": hidden,
        "n_layers": 4,
    }
    if tensorized:
        return NeuralOpTFNO(**common_kwargs, factorization="tucker", rank=0.05)
    return NeuralOpFNO(**common_kwargs)


def _build_neuralop_uno(
    dataset: str,
    in_channels: int,
    out_channels: int,
) -> nn.Module:
    if dataset == "burgers":
        raise ValueError(
            "neuralop_uno supports only 2D operator datasets (darcy, navier_stokes)."
        )

    _, _, NeuralOpUNO = _import_neuralop_models()
    return NeuralOpUNO(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=64,
        lifting_channels=128,
        projection_channels=128,
        n_layers=5,
        uno_out_channels=[64, 96, 96, 96, 64],
        uno_n_modes=[[8, 8], [8, 8], [8, 8], [8, 8], [8, 8]],
        uno_scalings=[[1.0, 1.0], [0.5, 0.5], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]],
        domain_padding=0.1,
        channel_mlp_skip="linear",
    )


class DeepONet1DWrapper(nn.Module):
    def __init__(self, resolution: int, hidden: int = 128, depth: int = 3) -> None:
        super().__init__()
        self.resolution = resolution
        self.model = DeepONet1D(branch_dim=resolution, hidden=hidden, depth=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C=1, N)
        b, _, n = x.shape
        u = x.view(b, n)
        coords = torch.linspace(0, 1, n, device=x.device).view(1, n, 1).repeat(b, 1, 1)
        return self.model((u, coords))


class DeepONet2DWrapper(nn.Module):
    def __init__(self, height: int, width: int, hidden: int = 128, depth: int = 3) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.model = DeepONet2D(branch_dim=height * width, trunk_dim=2, hidden=hidden, depth=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C=1, H, W)
        b, _, h, w = x.shape
        u = x.view(b, h * w)

        xs = torch.linspace(0, 1, w, device=x.device)
        ys = torch.linspace(0, 1, h, device=x.device)
        grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1)  # (W,H,2)
        coords = grid.reshape(1, h * w, 2).repeat(b, 1, 1)

        out = self.model((u, coords))  # (B, H*W)
        return out.view(b, 1, h, w)


def build_operator_model(
    model_name: str,
    dataset_name: str,
    dataset_meta: dict,
) -> nn.Module:
    name = model_name.lower()
    dataset = dataset_name.lower()

    if dataset == "burgers":
        resolution = int(dataset_meta.get("resolution", 128))
        in_ch = int(dataset_meta.get("in_channels", 1) or 1)
        out_ch = int(dataset_meta.get("out_channels", 1) or 1)
        if name == "fno":
            return FNO1D(in_ch=in_ch, out_ch=out_ch, width=64, modes=min(16, max(4, resolution // 8)))
        if name == "deeponet":
            return DeepONet1DWrapper(resolution=resolution, hidden=128, depth=3)
        if name == "gnot":
            return GNOT1D(in_channels=in_ch, out_channels=out_ch, width=64, n_latents=64, depth=4, heads=4)
        if name == "neuralop_fno":
            return _build_neuralop_fno(
                dataset=dataset,
                resolution=resolution,
                in_channels=in_ch,
                out_channels=out_ch,
                tensorized=False,
            )
        if name == "neuralop_tfno":
            return _build_neuralop_fno(
                dataset=dataset,
                resolution=resolution,
                in_channels=in_ch,
                out_channels=out_ch,
                tensorized=True,
            )

    if dataset in {"darcy", "navier_stokes"}:
        resolution = int(dataset_meta.get("train_resolution", dataset_meta.get("test_resolution", 64)))
        in_ch = int(dataset_meta.get("in_channels", 1) or 1)
        out_ch = int(dataset_meta.get("out_channels", 1) or 1)
        if name == "fno":
            return FNO2D(in_ch=in_ch, out_ch=out_ch, width=96, modes=12, depth=4)
        if name == "deeponet":
            return DeepONet2DWrapper(height=resolution, width=resolution, hidden=128, depth=3)
        if name == "gnot":
            return GNOT2D(in_channels=in_ch, out_channels=out_ch, width=64, n_latents=64, depth=4, heads=4)
        if name == "neuralop_fno":
            return _build_neuralop_fno(
                dataset=dataset,
                resolution=resolution,
                in_channels=in_ch,
                out_channels=out_ch,
                tensorized=False,
            )
        if name == "neuralop_tfno":
            return _build_neuralop_fno(
                dataset=dataset,
                resolution=resolution,
                in_channels=in_ch,
                out_channels=out_ch,
                tensorized=True,
            )
        if name in {"neuralop_uno", "uno"}:
            return _build_neuralop_uno(
                dataset=dataset,
                in_channels=in_ch,
                out_channels=out_ch,
            )

    raise ValueError(
        f"Unsupported operator combination model='{model_name}', dataset='{dataset_name}'. "
        "Use model in {fno, deeponet, gnot, neuralop_fno, neuralop_tfno, neuralop_uno} "
        "and dataset in {burgers, darcy, navier_stokes}."
    )
