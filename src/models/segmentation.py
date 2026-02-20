from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import segmentation as tv_seg


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.bottleneck = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1)
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        xb = self.bottleneck(self.pool(x3))

        d3 = self.up3(xb)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))
        return self.head(d1)


def build_segmentation_model(model_name: str, num_classes: int, in_channels: int = 3) -> nn.Module:
    name = model_name.lower()
    if name in {"unet", "tiny_unet"}:
        return TinyUNet(in_channels=in_channels, num_classes=num_classes, base_channels=32)

    if name in {"deeplabv3_resnet50", "deeplabv3"}:
        model = tv_seg.deeplabv3_resnet50(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
        )
        if in_channels != 3:
            model.backbone.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        return model

    raise ValueError(
        f"Unsupported segmentation model '{model_name}'. Use one of: unet, deeplabv3_resnet50"
    )

