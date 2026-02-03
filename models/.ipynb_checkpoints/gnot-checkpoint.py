import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x
        
class GNOT1D(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        width=64,
        latent_dim=64,
        n_latents=64,
        depth=4,
        heads=4,
    ):
        super().__init__()

        self.latents = nn.Parameter(
            torch.randn(1, n_latents, latent_dim)
        )

        self.in_proj = nn.Linear(in_channels + 1, width)
        self.latent_proj = nn.Linear(latent_dim, width)

        self.enc_attn = nn.MultiheadAttention(
            width, heads, batch_first=True
        )

        self.latent_blocks = nn.ModuleList(
            [TransformerBlock(width, heads) for _ in range(depth)]
        )

        self.dec_attn = nn.MultiheadAttention(
            width, heads, batch_first=True
        )

        self.out_proj = nn.Linear(width, out_channels)

    def forward(self, x):
        """
        x: (B, 1, N)
        """
        B, _, N = x.shape
        x = x.permute(0, 2, 1)  # (B, N, 1)

        # dynamic grid (resolution-free)
        grid = torch.linspace(
            0, 1, N, device=x.device
        ).view(1, N, 1).repeat(B, 1, 1)

        x = self.in_proj(torch.cat([x, grid], dim=-1))

        latents = self.latents.repeat(B, 1, 1)
        latents = self.latent_proj(latents)

        # encoder: input → latent
        enc, _ = self.enc_attn(latents, x, x)
        latents = latents + enc

        # latent transformer
        for block in self.latent_blocks:
            latents = block(latents)

        # decoder: latent → output grid
        dec, _ = self.dec_attn(x, latents, latents)
        x = x + dec

        out = self.out_proj(x)
        return out.permute(0, 2, 1)
        
class GNOT2D(nn.Module):
    """
    GNOT 2D (Darcy)
    Input : (B, 1, H, W)
    Output: (B, 1, H, W)
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        width=64,
        latent_dim=64,
        n_latents=64,
        depth=4,
        heads=4,
    ):
        super().__init__()

        self.latents = nn.Parameter(
            torch.randn(1, n_latents, latent_dim)
        )

        self.in_proj = nn.Linear(in_channels + 2, width)
        self.latent_proj = nn.Linear(latent_dim, width)

        self.enc_attn = nn.MultiheadAttention(
            width, heads, batch_first=True
        )

        self.latent_blocks = nn.ModuleList(
            [TransformerBlock(width, heads) for _ in range(depth)]
        )

        self.dec_attn = nn.MultiheadAttention(
            width, heads, batch_first=True
        )

        self.out_proj = nn.Linear(width, out_channels)

    def forward(self, x):
        """
        x: (B, 1, H, W)
        """
        B, _, H, W = x.shape
        N = H * W

        # flatten
        x = x.view(B, 1, N).permute(0, 2, 1)  # (B, N, 1)

        # dynamic grid
        xs = torch.linspace(0, 1, W, device=x.device)
        ys = torch.linspace(0, 1, H, device=x.device)
        grid = torch.stack(
            torch.meshgrid(xs, ys, indexing="xy"), dim=-1
        )
        grid = grid.reshape(1, N, 2).repeat(B, 1, 1)

        x = self.in_proj(torch.cat([x, grid], dim=-1))

        # latent tokens
        latents = self.latents.repeat(B, 1, 1)
        latents = self.latent_proj(latents)

        # encoder: input → latent
        enc, _ = self.enc_attn(latents, x, x)
        latents = latents + enc

        # latent transformer
        for block in self.latent_blocks:
            latents = block(latents)

        # decoder: latent → grid
        dec, _ = self.dec_attn(x, latents, latents)
        x = x + dec

        out = self.out_proj(x)
        return out.permute(0, 2, 1).view(B, 1, H, W)