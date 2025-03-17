import torch
import torch.nn as nn


class Projector(nn.Module):
    """
    Project the hidden states to a.
    """

    def __init__(self, projector_type="linear", input_dim: int=768, output_dim: int=768):
        super().__init__()

        assert projector_type in ["linear", "mlp"], "projector_type should be either 'linear' or 'mlp'"

        modules = [self._build_layer(input_dim, output_dim)]
        if projector_type == "mlp":
            # only 2 layers (Llava 1.5)
            modules.extend([nn.GELU(), self._build_layer(output_dim, output_dim)])

        self.projector = nn.Sequential(*modules)

    def _build_layer(self, input_dim, output_dim):
        layer = nn.Linear(input_dim, output_dim)
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.0)
        return layer

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projector(hidden_states)
