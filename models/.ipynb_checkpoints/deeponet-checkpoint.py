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
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

class DeepONet1D(nn.Module):
    """
    branch: (B, Nb)
    trunk:  (B, X, 1)
    output: (B, 1, X)
    """
    def __init__(self, branch_dim, hidden=128):
        super().__init__()
        self.branch = MLP([branch_dim, hidden, hidden])
        self.trunk = MLP([1, hidden, hidden])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        u, x = inputs                  # x: (B, X, 1)
        b = self.branch(u)             # (B, h)
        t = self.trunk(x)              # (B, X, h)
        y = torch.einsum("bh,bxh->bx", b, t) + self.bias
        return y.unsqueeze(1)
        
class DeepONet2D(nn.Module):
    """
    Input:
      branch: (B, Nb)
      trunk:  (B, Nt, d)
    Output:
      (B, Nt)
    """
    def __init__(self, branch_dim, trunk_dim, hidden=128):
        super().__init__()
        self.branch = MLP([branch_dim, hidden, hidden])
        self.trunk = MLP([trunk_dim, hidden, hidden])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        """
        inputs = (branch_input, trunk_input)
        """
        u, y = inputs
        b = self.branch(u)               # (B, h)
        t = self.trunk(y)                # (B, Nt, h)
        return torch.einsum("bh,bnh->bn", b, t) + self.bias
