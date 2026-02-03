import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        net = []
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                net.append(nn.GELU())
                net.append(nn.LayerNorm(layers[i+1]))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

class DeepONet1D(nn.Module):
    """
    branch: (B, Nb)
    trunk : (B, N, 1)
    output: (B, 1, N)
    """

    def __init__(self, branch_dim, hidden=128, depth=3):
        super().__init__()

        self.branch = MLP(
            [branch_dim] + [hidden]*depth
        )
        self.trunk = MLP(
            [1] + [hidden]*depth
        )

        self.bias = nn.Parameter(torch.zeros(hidden))

    def forward(self, inputs):
        u, x = inputs  # x: (B, N, 1)

        b = self.branch(u)          # (B, h)
        t = self.trunk(x)           # (B, N, h)

        y = torch.einsum("bh,bnh->bn", b, t)
        y = y + self.bias.sum()

        return y.unsqueeze(1)
        
class DeepONet2D(nn.Module):
    """
    branch: (B, Nb)
    trunk : (B, N, 2)
    output: (B, N)
    """

    def __init__(self, branch_dim, trunk_dim=2, hidden=128, depth=3):
        super().__init__()

        self.branch = MLP(
            [branch_dim] + [hidden]*depth
        )
        self.trunk = MLP(
            [trunk_dim] + [hidden]*depth
        )

        self.bias = nn.Parameter(torch.zeros(hidden))

    def forward(self, inputs):
        u, y = inputs

        b = self.branch(u)          # (B, h)
        t = self.trunk(y)           # (B, N, h)

        out = torch.einsum("bh,bnh->bn", b, t)
        out = out + self.bias.sum()

        return out
