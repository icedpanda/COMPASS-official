from typing import Optional, List

import torch
from torchmetrics import Metric
from torchmetrics import MetricCollection


class BaseRecMetric(Metric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True

    def __init__(self, labels_ids, k=1, dist_sync_on_step=False, compute_on_cpu=False):
        """

        Args:
            labels_ids: list of label ids to be used for the metric
            k: the k in "topk"
            dist_sync_on_step: This argument is bool that indicates if the metric should synchronize between different
                devices every time.
            compute_on_cpu: This argument is bool that indicates if the metric should be computed on CPU
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_cpu=compute_on_cpu)
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        # speed up by 750x times compared to list.index()
        self.k = k
        self.labels_ids = labels_ids

    def update(self, preds, labels, sampled_indices=None):
        assert preds.shape[0] == labels.shape[0], "preds and labels must have the same number of samples"
        if sampled_indices is None:
            preds = preds[:, self.labels_ids]
            _, preds_rank = torch.topk(preds, self.k, dim=1)
        else:
            _, local_preds_rank = torch.topk(preds, self.k, dim=1)
            # convert local indices to global indices
            preds_rank = torch.gather(sampled_indices, 1, local_preds_rank)
        self._compute_metric(preds_rank, labels)

    def _compute_metric(self, preds_rank, labels):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError


class HitRate(BaseRecMetric):
    """
    Compute recall@k for a given set of labels and predictions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")

    def _compute_metric(self, preds_rank, labels):
        labels_idx = labels.unsqueeze(-1)
        # Find the matches between predicted ranks and labels.
        # If there is at least one match, then the prediction is correct.
        matches = (preds_rank == labels_idx).sum(dim=-1)
        self.correct += matches.sum()
        self.total += labels.size(0)

    def compute(self):
        return self.correct.float() / self.total


class NDCG(BaseRecMetric):
    """
    Compute NDCG@k for a given set of labels and predictions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("ndcg", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    def _compute_metric(self, preds_rank, labels):
        labels_idx = labels.unsqueeze(-1)

        # Find the matches between predicted ranks and labels
        matches = (preds_rank == labels_idx).nonzero(as_tuple=True)

        # Get the indices of the matches
        label_ranks = matches[1]
        scores = 1 / (torch.log2(label_ranks.float() + 2))
        self.ndcg += scores.sum()
        self.total += labels.size(0)

    def compute(self):
        return self.ndcg.float() / self.total


class MRR(BaseRecMetric):
    """
    Compute MEAN RECIPROCAL RANK @k for a given set of labels and predictions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("mrr", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    def _compute_metric(self, preds_rank, labels):
        labels_idx = labels.unsqueeze(-1)
        matches = (preds_rank == labels_idx).nonzero(as_tuple=True)
        label_ranks = matches[1]  # get the column indices
        scores = 1 / (label_ranks.float() + 1)
        self.mrr += scores.sum()
        self.total += labels.size(0)

    def compute(self):
        return self.mrr.float() / self.total


def get_rec_metrics(item_ids, ks: List[int] = None):
    if ks is None:
        ks = [1, 10, 50]
    metrics = []
    for k in ks:
        metric_collection = MetricCollection(
            [
                HitRate(item_ids, k=k, ),
                NDCG(item_ids, k=k, ),
                MRR(item_ids, k=k, )
            ],
            postfix=f"@{k}"
        )
        metrics.append(metric_collection)
    return metrics
