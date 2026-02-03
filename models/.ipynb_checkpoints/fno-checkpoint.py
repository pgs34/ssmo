import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes = modes
        self.scale = 1 / (in_ch * out_ch)
        self.weight = nn.Parameter(
            self.scale * torch.randn(in_ch, out_ch, modes, 2)
        )

    def forward(self, x):
        # x: (B, C, X)
        x_ft = fft.rfft(x)
        out_ft = torch.zeros(
            x.size(0), self.out_ch, x_ft.size(-1),
            device=x.device, dtype=torch.cfloat
        )
        w = torch.view_as_complex(self.weight)
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box",
            x_ft[:, :, :self.modes],
            w
        )
        return fft.irfft(out_ft, n=x.size(-1))
        
class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes = modes
        self.scale = 1 / (in_ch * out_ch)

        self.weight = nn.Parameter(
            self.scale * torch.randn(in_ch, out_ch, modes, modes, 2)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = fft.rfft2(x)

        out_ft = torch.zeros(
            B, self.out_ch, H, W // 2 + 1,
            device=x.device, dtype=torch.cfloat
        )

        w = torch.view_as_complex(self.weight)

        out_ft[:, :, :self.modes, :self.modes] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes, :self.modes],
            w,
        )

        return fft.irfft2(out_ft, s=(H, W))

class FNOBlock2D(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.norm = nn.InstanceNorm2d(width)

    def forward(self, x):
        y = self.spectral(x) + self.pointwise(x)
        y = self.norm(y)
        return F.gelu(y)


class FNO1D(nn.Module):
    def __init__(self, in_ch, out_ch, width=64, modes=16):
        super().__init__()
        self.fc0 = nn.Conv1d(in_ch, width, 1)

        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)

        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)

        self.fc1 = nn.Conv1d(width, 128, 1)
        self.fc2 = nn.Conv1d(128, out_ch, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = self.conv1(x) + self.w1(x)
        x = F.gelu(x)
        x = self.conv2(x) + self.w2(x)
        x = F.gelu(x)
        return self.fc2(F.gelu(self.fc1(x)))
        
class FNO2D(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        width=96,
        modes=12,
        depth=4,
    ):
        super().__init__()

        self.width = width

        # input: (a(x,y), x, y)
        self.fc0 = nn.Conv2d(in_ch + 2, width, 1)

        self.blocks = nn.ModuleList(
            [FNOBlock2D(width, modes) for _ in range(depth)]
        )

        self.fc1 = nn.Conv2d(width, 128, 1)
        self.fc2 = nn.Conv2d(128, out_ch, 1)

    def forward(self, x):
        """
        x: (B, in_ch, H, W)
        """
        B, _, H, W = x.shape

        # coord encoding
        xs = torch.linspace(0, 1, W, device=x.device)
        ys = torch.linspace(0, 1, H, device=x.device)
        grid = torch.stack(
            torch.meshgrid(xs, ys, indexing="xy"), dim=0
        )
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

        x = torch.cat([x, grid], dim=1)
        x = self.fc0(x)

        for block in self.blocks:
            x = block(x)

        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

