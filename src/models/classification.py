from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tv_models


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = h.flatten(1)
        return self.classifier(h)


class SimpleMLP(nn.Module):
    def __init__(self, in_channels: int, image_size: int, num_classes: int) -> None:
        super().__init__()
        input_dim = in_channels * image_size * image_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_classification_model(
    model_name: str,
    num_classes: int,
    in_channels: int,
    image_size: int,
) -> nn.Module:
    name = model_name.lower()

    if name in {"simple_cnn", "cnn"}:
        return SimpleCNN(in_channels=in_channels, num_classes=num_classes)

    if name in {"simple_mlp", "mlp"}:
        return SimpleMLP(in_channels=in_channels, image_size=image_size, num_classes=num_classes)

    if name == "resnet18":
        model = tv_models.resnet18(weights=None, num_classes=num_classes)
        if in_channels != 3:
            model.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        return model

    if name in {"vit_b16", "vit"}:
        # image_size must be divisible by patch_size(16)
        if image_size % 16 != 0:
            raise ValueError(
                f"vit_b16 requires image_size divisible by 16, got {image_size}"
            )
        model = tv_models.vit_b_16(
            weights=None,
            image_size=image_size,
            num_classes=num_classes,
        )
        if in_channels != 3:
            model.conv_proj = nn.Conv2d(
                in_channels,
                model.hidden_dim,
                kernel_size=16,
                stride=16,
            )
        return model

    raise ValueError(
        f"Unsupported classification model '{model_name}'. "
        "Use one of: simple_cnn, simple_mlp, resnet18, vit_b16"
    )

