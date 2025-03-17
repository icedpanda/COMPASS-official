"""
Models trained with default MLP item embeddings
"""
from typing import List

import torch
import wandb
from lightning.pytorch.loggers import WandbLogger

import torch.nn as nn
from torchmetrics import MeanMetric

from src.llmcrs.models.lightning_modules import BaseLightningModule
from src.llmcrs.data.datatype import TaskType
from src.llmcrs.metrics import get_rec_metrics


class TransformerDefaultEmbeddingLightning(BaseLightningModule):
    def __init__(
            self,
            net: nn.Module,
            item_entity_ids: List[int],
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            is_compile: bool = False,
            warmup_steps: int = None,
            total_steps: int = None,
    ):
        super().__init__()
        self.phase = TaskType.REC.value
        self.save_hyperparameters(ignore=["item_entity_ids", "net"], logger=False)
        self.rec_model = net
        self.compile = is_compile

        # default metrics
        self.train_avg_loss = MeanMetric()
        self.val_avg_loss = MeanMetric()
        self.test_avg_loss = MeanMetric()
        self.loss_fn = nn.CrossEntropyLoss()

        # other metrics
        self.new_item_entity_ids = item_entity_ids.copy()
        # self.new_item_entity_ids.insert(0, 0)
        metrics_list = get_rec_metrics(self.new_item_entity_ids)
        self.val_metrics = nn.ModuleList([metric.clone(prefix="val/rec/") for metric in metrics_list])
        self.test_metrics = nn.ModuleList([metric.clone(prefix="test/rec/") for metric in metrics_list])
        self.target_metrics = ["val/rec/HitRate@1", "val/rec/HitRate@50"]

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about them.
        This hook is called on every process when using DDP.

        Args:
            stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.

        Returns:
            None
        """
        if self.hparams.is_compile and stage == "fit":
            self.print("Compiling the model...")
            self.rec_model = torch.compile(self.rec_model, dynamic=True)

    def training_step(self, batch, batch_idx):
        inputs = self._prepare_inputs(batch)
        labels = batch['items']

        outputs = self.rec_model(**inputs["context_tokens"])
        loss = self.loss_fn(outputs, labels)

        self.train_avg_loss.update(loss)
        self.log("train/rec/step_loss", loss, prog_bar=False)

        return loss

    def on_train_epoch_end(self):
        train_loss = self.train_avg_loss.compute()
        self.log("train/rec/epoch_loss", train_loss, prog_bar=True)
        self.log("train/rec/num_epoch", float(self.current_epoch), prog_bar=False)
        self.train_avg_loss.reset()

    def on_validation_start(self):
        # if use wandb logger, use wandb define metrics
        if self.trainer.current_epoch == 0 and any(isinstance(logger, WandbLogger) for logger in self.trainer.loggers):
            self.print("Using wandb logger, define metrics")
            wandb.define_metric("val/rec/epoch_loss", summary="min")
            for metrics in self.val_metrics:
                for keys in metrics.keys():
                    wandb.define_metric(keys, summary="max")

    def validation_step(self, batch, batch_idx):
        inputs = self._prepare_inputs(batch)
        labels = batch['items']

        outputs = self.rec_model(**inputs["context_tokens"])
        loss = self.loss_fn(outputs, labels)

        self.val_avg_loss.update(loss)
        for metrics in self.val_metrics:
            metrics.update(outputs, labels)

        return loss

    def on_validation_epoch_end(self):
        val_loss = self.val_avg_loss.compute()
        self.log("val/rec/epoch_loss", val_loss, prog_bar=True)
        self.log("val/rec/num_epoch", float(self.current_epoch), prog_bar=False, )
        self.val_avg_loss.reset()
        target_metrics = 0
        for metrics in self.val_metrics:
            result = metrics.compute()
            self.log_dict(result, prog_bar=True)

            # extract and sum the target metrics
            for key in self.target_metrics:
                if key in result:
                    target_metrics += result[key]
            metrics.reset()
        self.log("val/rec/target_metrics", target_metrics, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        inputs = self._prepare_inputs(batch)
        # context_tokens = batch['preference_tokens']
        # context_tokens = batch['context_tokens']
        labels = batch['items']
        outputs = self.rec_model(**inputs["context_tokens"])

        for metrics in self.test_metrics:
            metrics.update(outputs, labels)

    def on_test_epoch_end(self):
        for metrics in self.test_metrics:
            self.log_dict(metrics.compute(), prog_bar=True)
            metrics.reset()

    def _prepare_inputs(self, batch):
        """
        Prepares the inputs for the model based on the specified user feature type.

        Args:
            batch (dict): A dictionary containing the batch data.

        Returns:
            inputs (dict): A dictionary containing the prepared inputs for the model.
                           The keys in the dictionary depend on the user feature type.

        """
        return {
            "context_tokens": batch['context_tokens'],
        }
