from torchmetrics import MetricCollection
from torchmetrics.text import BLEUScore
from torchmetrics.text.bert import BERTScore
from torchmetrics.text.rouge import ROUGEScore


def get_gen_metrics(prefix="test/") -> MetricCollection:
    metrics = MetricCollection([
        ROUGEScore(),
    ], prefix=prefix)
    return metrics
