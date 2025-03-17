import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionModule(nn.Module):
    """
    Self-attention block. equation: 4 in KBRD paper.
    A structured self-attentive sentence embedding.:
    https://arxiv.org/abs/1703.03130
    """

    def __init__(self, dim: int, da: int):
        super(SelfAttentionModule, self).__init__()
        self.dim = dim
        self.da = da
        # nn.Linear equivalents to the matrix multiplication(y=Wx).
        self._init_attention_weights(dim, da)
        self.mask_value = -1e9

    def _init_attention_weights(self, dim: int, da: int):
        """Initialize attention weights."""
        gain = nn.init.calculate_gain('tanh')

        self.weights_a = nn.Linear(dim, da, bias=False)
        self.weights_b = nn.Linear(da, 1, bias=False)

        nn.init.xavier_uniform_(self.weights_a.weight, gain=gain)
        nn.init.xavier_uniform_(self.weights_b.weight, gain=gain)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the self-attention mechanism.

        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, sequence_length, dim].
            mask (torch.Tensor, optional): Binary tensor indicating whether each position is a valid input or not.
                (1 for valid positions, 0 for invalid positions). Defaults to None.

        Returns:
            torch.Tensor: Self-attended tensor of shape [batch_size, dim].
        """
        # use softmax as attention
        attention_scores = self.weights_a(inputs)
        attention_scores = F.tanh(attention_scores)
        attention_scores = self.weights_b(attention_scores)
        if mask is not None:
            attention_scores = attention_scores + self.mask_value * (1 - mask)
        attention_probs = F.softmax(attention_scores, dim=1).transpose(1, 2)
        return torch.matmul(attention_probs, inputs).squeeze(dim=1)


def create_mask(context_entities: torch.Tensor, pad_value: int = 0) -> torch.Tensor:
    """
    Create a mask tensor from the context entities tensor.

    Args:
        context_entities (torch.Tensor): A tensor containing context entities. It may have padding values.
        pad_value (int): The value used for padding in the context_entities tensor.
                         Default is 0.

    Returns:
        torch.Tensor: A binary mask tensor with the same shape as context_entities.
                      It contains 1 where context_entities is valid and 0 where it is padding.
    """
    return (context_entities != pad_value).long()


def create_knowledge_prompt_attention_mask(context_entities, padding_idx=0):
    """
    Create a mask tensor from the context entities tensor.
    if context entities contain entirely of padding index, then the mask is 0, else 1
    Args:
        context_entities: entity ids of the context
        padding_idx: padding index

    Returns:

    """
    mask = (context_entities != padding_idx).any(dim=-1)

    # convert boolean mask to int and reshape to (batch_size, 1)
    mask.int().view(-1, 1)

    return mask
