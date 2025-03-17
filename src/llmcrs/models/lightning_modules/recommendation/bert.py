"""
Models trained with default MLP item embeddings
"""
from typing import List

import torch
import torch.nn as nn
import wandb
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MeanMetric

from src.llmcrs.data.datatype import TaskType
from src.llmcrs.metrics import get_rec_metrics
from src.llmcrs.models.lightning_modules import BaseLightningModule
from src.llmcrs.models.recommendation.bert import BERTModel


class BERTLightning(BaseLightningModule):
    def __init__(
            self,
            net: BERTModel,
            item_entity_ids: List[int],
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            base_lr: float = None,
            base_weight_decay: float = None,
            warmup_steps: int = None,
            total_steps: int = None,
    ):
        super().__init__()
        self.phase = TaskType.REC.value
        self.save_hyperparameters(ignore=["item_entity_ids", "net"], logger=False)
        self.rec_model = net

        # default metrics
        self.train_avg_loss = MeanMetric()
        self.val_avg_loss = MeanMetric()
        self.test_avg_loss = MeanMetric()

        # other metrics
        self.new_item_entity_ids = item_entity_ids.copy()
        # self.new_item_entity_ids.insert(0, 0)
        metrics_list = get_rec_metrics(self.new_item_entity_ids)
        self.val_metrics = nn.ModuleList([metric.clone(prefix="val/rec/") for metric in metrics_list])
        self.test_metrics = nn.ModuleList([metric.clone(prefix="test/rec/") for metric in metrics_list])
        self.target_metrics = ["val/rec/HitRate@10", "val/rec/HitRate@50"]

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=[p for p in self.parameters() if p.requires_grad],
                                           lr=self.hparams.base_lr, weight_decay=self.hparams.base_weight_decay)

        if self.hparams.scheduler is not None:
            scheduler, scheduler_config = self._create_scheduler_and_config(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        llm_tokens = batch["context_tokens"]  # this is from llm summarization
        history_tokens = batch["history_tokens"]
        labels = batch['items']
        llm_compressed_tokens = batch['compressed_tokens']
        outputs = self.rec_model(history_tokens, llm_tokens, llm_compressed_tokens, labels)

        self.train_avg_loss.update(outputs.loss)
        self.log("train/rec/step_loss", outputs.loss, prog_bar=False)

        return outputs.loss

    def on_train_epoch_end(self):
        self.log("train/rec/epoch_loss", self.train_avg_loss.compute(), prog_bar=True)
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
        llm_tokens = batch["context_tokens"]  # this is from llm summarization
        history_tokens = batch["history_tokens"]
        labels = batch['items']
        llm_compressed_tokens = batch['compressed_tokens']
        outputs = self.rec_model(history_tokens, llm_tokens, llm_compressed_tokens, labels)
        self.val_avg_loss.update(outputs.loss)
        for metrics in self.val_metrics:
            metrics.update(outputs.logits, labels)

        return outputs.loss

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
        llm_tokens = batch["context_tokens"]  # this is from llm summarization
        history_tokens = batch["history_tokens"]
        labels = batch['items']
        llm_compressed_tokens = batch['compressed_tokens']
        outputs = self.rec_model(history_tokens, llm_tokens, llm_compressed_tokens)

        for metrics in self.test_metrics:
            metrics.update(outputs.logits, labels)

    def on_test_epoch_end(self):
        for metrics in self.test_metrics:
            self.log_dict(metrics.compute(), prog_bar=True)
            metrics.reset()
