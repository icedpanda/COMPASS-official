import torch
from torch import Tensor


def cos_similarity(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    Returns:
        Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def dot_score(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the dot product score a[i] * b[j] for all i and j.

    Returns:
        Matrix with res[i][j]  = a[i] * b[j]
    """
    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))
