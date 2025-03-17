import torch


def avg_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Calculates the average pooling of the last hidden states.

    Args:
        last_hidden_states (Tensor): The hidden states to be pooled. Shape: (batch_size, sequence_length, hidden_size).
        attention_mask (Tensor): The attention mask tensor. Shape: (batch_size, sequence_length).

    Returns:
        Tensor: The tensor resulting from the average pooling operation. Shape: (batch_size, hidden_size).
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def max_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Implements max pooling on the given hidden states.

    Args:
        last_hidden_states (torch.Tensor): The hidden states to be pooled.
            Shape: (batch_size, sequence_length, hidden_size).
        attention_mask (torch.Tensor): The attention mask indicating which tokens should be masked.
            Shape: (batch_size, sequence_length).

    Returns:
        torch.Tensor: The pooled hidden states. Shape: (batch_size, hidden_size).
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    last_hidden_states[input_mask_expanded == 0] = -1e9
    return torch.max(last_hidden_states, 1)[0]


def cls_token_pooling(last_hidden_states: torch.Tensor, _) -> torch.Tensor:
    """
    Return the representation of the CLS token from a sequence of hidden states.

    Parameters:
    - hidden_states (torch.Tensor): The hidden states to be pooled. Shape: (batch_size, sequence_length, hidden_size)

    Returns:
        torch.Tensor: The representation of the CLS token. Shape: (batch_size, hidden_size)
    """
    return last_hidden_states[:, 0, :]


def last_token_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Return the representation of the last token from a sequence of hidden states.
    Args:
        last_hidden_states (torch.Tensor): The hidden states to be pooled.
            Shape: (batch_size, sequence_length, hidden_size)
        attention_mask (torch.Tensor): The attention mask indicating which tokens should be masked.

    Returns:
        torch.Tensor: The representation of the last token. Shape: (batch_size, hidden_size)
    """
    if attention_mask is None:
        return last_hidden_states[:, -1, :]
    last_token_positions = attention_mask.sum(dim=1) - 1
    return last_hidden_states[torch.arange(last_hidden_states.size(0)), last_token_positions]


def get_pooling_method(pool_strategy: str):
    """
    Return the corresponding pooling method based on the given pool strategy.

    Parameters:
        pool_strategy (str): The pooling strategy to use. Must be one of "cls", "avg", "max" or "last_token".

    Returns:
        function: The corresponding pooling method based on the given pool strategy.
    """
    pooling_methods = {
        "cls": cls_token_pooling,
        "avg": avg_pooling,
        "max": max_pooling,
        "last_token": last_token_pooling
    }
    return pooling_methods[pool_strategy]
