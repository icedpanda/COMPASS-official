from typing import List

import torch
import torch.nn as nn
import wandb
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MeanMetric

from src.llmcrs.data.datatype import TaskType
from src.llmcrs.metrics import get_rec_metrics
from src.llmcrs.models.lightning_modules import BaseLightningModule
from src.llmcrs.models.recommendation.kbrd import KBRD


class KBRDLightning(BaseLightningModule):
    def __init__(
            self,
            net: KBRD,
            item_entity_ids: List[int],
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            base_lr: float = None,
            base_weight_decay: float = None,
            kg_lr: float = None,
            kg_weight_decay: float = None,
            is_compile: bool = False,
            warmup_steps: int = None,
            total_steps: int = None,
            use_cl: bool = True,
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
        self.use_cl = use_cl
        if self.use_cl:
            self.train_avg_cl_loss = MeanMetric()
            self.val_avg_cl_loss = MeanMetric()

        # other metrics
        self.new_item_entity_ids = item_entity_ids.copy()
        # self.new_item_entity_ids.insert(0, 0)
        metrics_list = get_rec_metrics(self.new_item_entity_ids)
        self.val_metrics = nn.ModuleList([metric.clone(prefix="val/rec/") for metric in metrics_list])
        self.test_metrics = nn.ModuleList([metric.clone(prefix="test/rec/") for metric in metrics_list])
        self.target_metrics = ["val/rec/HitRate@10", "val/rec/HitRate@50"]

    def configure_optimizers(self):
        # sourcery skip: collection-builtin-to-comprehension
        """Method to configure optimizers and schedulers.

        Returns:
            dict: A dictionary containing the optimizer and scheduler configuration.
            The keys are 'optimizer' and 'lr_scheduler'.
        """
        # Collecting all parameters first
        # Collect all parameters
        all_params = set(self.rec_model.parameters())
        optimizer_params = []

        def add_param_group(params, lr, weight_decay, name=None):
            if params:
                optimizer_params.append({"params": params, "lr": lr, "weight_decay": weight_decay,
                                         "name": name})

        if self.hparams.kg_lr is not None:
            # kg model parameters
            kg_params = [p for p in self.rec_model.kg_attention.parameters() if p.requires_grad] + \
                        [p for p in self.rec_model.graph_encoder.parameters() if p.requires_grad] + \
                        [p for p in self.rec_model.rec_layer.parameters() if p.requires_grad]
            add_param_group(kg_params, self.hparams.kg_lr, self.hparams.kg_weight_decay, "gnn")
            all_params -= set(kg_params)

        # Adding the rest of the parameters
        if all_params:
            self.print("Using default learning rate and weight decay for the rest of the parameters...")
            add_param_group(list(all_params), self.hparams.base_lr, self.hparams.base_weight_decay, "base")

        # Creating the optimizer
        optimizer = self.hparams.optimizer(params=optimizer_params)

        if self.hparams.scheduler is not None:
            scheduler, scheduler_config = self._create_scheduler_and_config(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

        return {"optimizer": optimizer}

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
            self.rec_model = torch.compile(self.rec_model, mode="reduce-overhead")

    def training_step(self, batch, batch_idx):

        context_entities, context_tokens = batch['context_entities'], batch["context_tokens"]
        labels = batch['items']
        llm_compressed_tokens = batch['compressed_tokens']

        outputs = self.rec_model(context_entities, context_tokens, llm_compressed_tokens, labels)

        self.train_avg_loss.update(outputs.loss)
        self.log("train/rec/step_loss", outputs.loss_rec, prog_bar=False)
        if self.use_cl:
            self.train_avg_cl_loss.update(outputs.loss_cl)
            self.log("train/rec/step_cl_loss", outputs.loss_cl, prog_bar=False)

        return outputs.loss

    def on_train_epoch_end(self):
        self.log("train/rec/epoch_loss", self.train_avg_loss.compute(), prog_bar=True)
        self.log("train/rec/num_epoch", float(self.current_epoch), prog_bar=False)
        if self.use_cl:
            self.log("train/rec/epoch_cl_loss", self.train_avg_cl_loss.compute(), prog_bar=True)
            self.train_avg_cl_loss.reset()
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
        context_entities, context_tokens = batch['context_entities'], batch["context_tokens"]
        labels = batch['items']
        llm_compressed_tokens = batch['compressed_tokens']

        outputs = self.rec_model(context_entities, context_tokens, llm_compressed_tokens, labels)

        self.val_avg_loss.update(outputs.loss_rec)
        if self.use_cl:
            self.val_avg_cl_loss.update(outputs.loss_cl)
        for metrics in self.val_metrics:
            metrics.update(outputs.sims, labels)

        return outputs.loss

    def on_validation_epoch_end(self):
        val_loss = self.val_avg_loss.compute()
        self.log("val/rec/epoch_loss", val_loss, prog_bar=True)
        self.log("val/rec/num_epoch", float(self.current_epoch), prog_bar=False, )
        if self.use_cl:
            self.log("val/rec/epoch_cl_loss", self.val_avg_cl_loss.compute(), prog_bar=True)
            self.val_avg_cl_loss.reset()
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

        context_entities, context_tokens = batch['context_entities'], batch["context_tokens"]
        labels = batch['items']
        llm_compressed_tokens = batch['compressed_tokens']

        outputs = self.rec_model(context_entities, context_tokens, llm_compressed_tokens)

        for metrics in self.test_metrics:
            metrics.update(outputs.sims, labels)

    def on_test_epoch_end(self):
        for metrics in self.test_metrics:
            self.log_dict(metrics.compute(), prog_bar=True)
            metrics.reset()
