import torch
import torch.nn as nn


class PositionEncoder(nn.Module):
    """
    Position Encoder that adds positional information to the input tensor.

    This one takes positional indices as input and returns the corresponding embeddings.
    """

    def __init__(self, max_len: int = 50, padding_idx: int = 0, dim: int = 128):
        super().__init__()

        self.position_embedding = nn.Embedding(max_len, dim, padding_idx=padding_idx)
        self.padding_idx = padding_idx

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the PositionEncoder.
        Args:
            x: (batch_size, seq_len) tensor of entity indices
            position_ids: (batch_size, seq_len) tensor of positional indices

        Returns:
            (batch_size, seq_len, dim) tensor of positional embeddings
        """
        if not position_ids:
            mask = x.ne(self.padding_idx).int()
            position_ids = torch.cumsum(mask, dim=-1).long().type_as(mask) * mask.long()
        return self.position_embedding(position_ids)
