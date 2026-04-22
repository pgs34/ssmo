import torch
import torch.nn as nn

class DeepONetFeatureExtractor1D(nn.Module):
    def __init__(self, N, local_points=64):
        super().__init__()
        self.N = N
        self.local_points = local_points

        idx = torch.linspace(0, N-1, local_points).long()
        self.register_buffer("idx", idx)

    def forward(self, u):
        """
        u: (B, 1, N)
        """
        u = u.squeeze(1)

        # global
        u_global = u

        # local (subsample)
        u_local = u[:, self.idx]

        # statistics
        mean = u.mean(dim=-1, keepdim=True)
        std = u.std(dim=-1, keepdim=True)
        energy = (u ** 2).mean(dim=-1, keepdim=True)

        u_stats = torch.cat([mean, std, energy], dim=-1)

        return u_global, u_local, u_stats

class DeepONetFeatureExtractor2D(nn.Module):
    """
    Input:
        u : (B, 1, H, W)
    Output:
        u_global : (B, H*W)
        u_local  : (B, L)
        u_stats  : (B, 3)
    """

    def __init__(self, H, W, local_points=128):
        super().__init__()
        self.H = H
        self.W = W
        self.N = H * W
        self.local_points = local_points

        # fixed subsampling indices
        idx = torch.linspace(0, self.N - 1, local_points).long()
        self.register_buffer("idx", idx)

    def forward(self, u):
        """
        u: (B, 1, H, W)
        """
        B = u.size(0)

        # flatten
        u_flat = u.view(B, -1)   # (B, H*W)

        # global branch
        u_global = u_flat

        # local branch (subsampled points)
        u_local = u_flat[:, self.idx]

        # statistics branch
        mean = u_flat.mean(dim=1, keepdim=True)
        std = u_flat.std(dim=1, keepdim=True)
        energy = (u_flat ** 2).mean(dim=1, keepdim=True)

        u_stats = torch.cat([mean, std, energy], dim=1)

        return u_global, u_local, u_stats