import torch.nn as nn
from torchmetrics import MeanMetric


def get_node_alignment_metrics(stage="train"):
    return nn.ModuleDict({
        f'{stage}/align/avg_node_loss': MeanMetric(),
        f'{stage}/align/itm_node_loss': MeanMetric(),
        f'{stage}/align/lm_node_loss': MeanMetric(),
        f'{stage}/align/itc_node_loss': MeanMetric(),
    })


def get_context_alignment_metrics(stage="train"):
    return nn.ModuleDict({
        f'{stage}/align/avg_context_loss': MeanMetric(),
        f'{stage}/align/lm_context_loss': MeanMetric(),
        f'{stage}/align/itm_context_loss': MeanMetric(),
        f'{stage}/align/itc_context_loss': MeanMetric(),
    })
