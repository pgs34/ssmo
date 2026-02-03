import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)
        
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class UNO1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, width=64):
        super().__init__()

        self.enc1 = ConvBlock1D(in_channels, width)
        self.enc2 = ConvBlock1D(width, width * 2)
        self.enc3 = ConvBlock1D(width * 2, width * 4)

        self.pool = nn.MaxPool1d(2)

        self.mid = ConvBlock1D(width * 4, width * 8)

        self.up3 = nn.ConvTranspose1d(width * 8, width * 4, 2, stride=2)
        self.dec3 = ConvBlock1D(width * 8, width * 4)

        self.up2 = nn.ConvTranspose1d(width * 4, width * 2, 2, stride=2)
        self.dec2 = ConvBlock1D(width * 4, width * 2)

        self.up1 = nn.ConvTranspose1d(width * 2, width, 2, stride=2)
        self.dec1 = ConvBlock1D(width * 2, width)

        self.out = nn.Conv1d(width, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        m = self.mid(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)
        
class UNO2D(nn.Module):
    """
    U-Net based Neural Operator
    Input:  (B, Cin, H, W)
    Output: (B, Cout, H, W)
    """
    def __init__(self, in_channels=1, out_channels=1, width=64):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, width)
        self.enc2 = ConvBlock(width, width * 2)
        self.enc3 = ConvBlock(width * 2, width * 4)

        self.pool = nn.MaxPool2d(2)

        self.mid = ConvBlock(width * 4, width * 8)

        self.up3 = nn.ConvTranspose2d(width * 8, width * 4, 2, stride=2)
        self.dec3 = ConvBlock(width * 8, width * 4)

        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
        self.dec2 = ConvBlock(width * 4, width * 2)

        self.up1 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.dec1 = ConvBlock(width * 2, width)

        self.out = nn.Conv2d(width, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        m = self.mid(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)