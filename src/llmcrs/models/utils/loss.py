from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def random_sample(head_idx: Tensor, tail_idx: Tensor, num_nodes) -> Tuple[Tensor, Tensor]:
    # random sample either head or tail
    num_negatives = head_idx.numel() // 2
    random_idx = torch.randint(num_nodes, head_idx.size(), device=head_idx.device)

    head_idx = head_idx.clone()
    tail_idx = tail_idx.clone()
    head_idx[:num_negatives] = random_idx[:num_negatives]
    tail_idx[num_negatives:] = random_idx[num_negatives:]

    return head_idx, tail_idx


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function from https://github.com/google-research/simclr.
    """

    def __init__(self, temperature=0.07, label_smoothing=0.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.loss_func = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, view1, view2):
        # view1, view2: batch_size, hidden_size
        views = torch.cat([view1, view2], dim=0)
        bs = view1.shape[0]

        # create labels
        labels = torch.cat([torch.arange(bs) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(views.device)

        # similarity matrix
        features = F.normalize(views, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        assert similarity_matrix.shape == (2 * bs, 2 * bs)
        assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        return self._compute_loss(positives, negatives)

    def _compute_loss(self, positives, negatives):
        logits = torch.cat([positives, negatives], dim=1)
        logits /= self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return self.loss_func(logits, labels)
