import torch
import torch.nn as nn

class GateFusion(nn.Module):
    """
    Gate network for fusion
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.fuse_layer = nn.Linear(input_dim * 2, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        features = torch.cat([input1, input2], dim=-1)
        gate = self.gate(self.fuse_layer(features))
        return gate * input1 + (1 - gate) * input2