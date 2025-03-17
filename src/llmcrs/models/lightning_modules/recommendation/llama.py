from typing import List

import torch
import torch.nn as nn
import wandb
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MeanMetric

from src.llmcrs.data.datatype import TaskType
from src.llmcrs.metrics import get_rec_metrics
from src.llmcrs.models.lightning_modules import BaseLightningModule
from src.llmcrs.models.recommendation import LLAMA


class LLMCRSLightning(BaseLightningModule):
    def __init__(
            self,
            net: LLAMA,
            item_entity_ids: List[int],
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            is_compile: bool = False,
            warmup_steps: int = None,
            total_steps: int = None,
            base_lr: float = None,
            base_weight_decay: float = None,
            use_reasoning: bool = False,
    ):
        super().__init__()
        self.phase = TaskType.REC.value
        self.save_hyperparameters(ignore=["item_entity_ids", "net"], logger=False)
        self.rec_model = net
        self.compile = is_compile
        self.use_reasoning = use_reasoning

        # default metrics
        self.train_avg_loss = MeanMetric()
        self.val_avg_loss = MeanMetric()
        self.test_avg_loss = MeanMetric()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        # other metrics
        self.new_item_entity_ids = item_entity_ids.copy()
        # self.new_item_entity_ids.insert(0, 0)
        metrics_list = get_rec_metrics(self.new_item_entity_ids)
        self.val_metrics = nn.ModuleList([metric.clone(prefix="val/rec/") for metric in metrics_list])
        self.test_metrics = nn.ModuleList([metric.clone(prefix="test/rec/") for metric in metrics_list])
        self.target_metrics = ["val/rec/HitRate@10", "val/rec/HitRate@50"]

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

    def configure_optimizers(self):
        # sourcery skip: collection-builtin-to-comprehension
        """Method to configure optimizers and schedulers.

        Returns:
            dict: A dictionary containing the optimizer and scheduler configuration.
            The keys are 'optimizer' and 'lr_scheduler'.
        """
        # Collecting all parameters first
        # Collect all parameters
        all_params = set(self.parameters())
        optimizer_params = []

        def add_param_group(params, lr, weight_decay, name=None):
            if params:
                optimizer_params.append({"params": params, "lr": lr, "weight_decay": weight_decay,
                                         "name": name})

        base_params = [p for p in self.rec_model.model.parameters() if p.requires_grad]
        # Base model parameters
        add_param_group(base_params, self.hparams.base_lr, self.hparams.base_weight_decay, "llm")
        all_params -= set(base_params)  # Remove base_params from all_params

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

    def training_step(self, batch, batch_idx):
        # llm labels are just reasoning labels
        llm_inputs, rec_labels, llm_labels = batch['llm_inputs_rec'], batch['items'], batch["llm_labels"]

        outputs = self.rec_model(llm_inputs, reasoning=llm_labels if self.use_reasoning else None)
        loss = self.loss_fn(outputs, rec_labels)

        self.train_avg_loss.update(loss)
        self.log("train/rec/step_loss", loss, prog_bar=False)

        return loss

    def on_train_epoch_end(self):
        loss = self.train_avg_loss.compute()
        self.log("train/rec/epoch_loss", loss, prog_bar=True)
        self.log("train/rec/num_epoch", float(self.current_epoch), prog_bar=False)
        self.train_avg_loss.reset()

    def on_validation_start(self):
        # if use wandb logger, use wandb define metrics
        if self.trainer.is_global_zero:
            if self.trainer.current_epoch == 0 and any(
                    isinstance(logger, WandbLogger) for logger in self.trainer.loggers):
                self.print("Using wandb logger, define metrics")
                wandb.define_metric("val/rec/epoch_loss", summary="min")
                for metrics in self.val_metrics:
                    for keys in metrics.keys():
                        wandb.define_metric(keys, summary="max")

    def validation_step(self, batch, batch_idx):
        # inputs = self._prepare_inputs(batch)
        llm_inputs, rec_labels, llm_labels = batch['llm_inputs_rec'], batch['items'], batch["llm_labels"]
        outputs = self.rec_model(llm_inputs, reasoning=llm_labels if self.use_reasoning else None)
        loss = self.loss_fn(outputs, rec_labels)

        self.val_avg_loss.update(loss)
        for metrics in self.val_metrics:
            metrics.update(outputs, rec_labels)

        return loss

    def on_validation_epoch_end(self):
        val_loss = self.val_avg_loss.compute()
        self.log("val/rec/epoch_loss", val_loss, prog_bar=True)
        self.log("val/rec/num_epoch", float(self.current_epoch), prog_bar=False)
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
        # inputs = self._prepare_inputs(batch)
        llm_inputs, rec_labels, llm_labels = batch['llm_inputs_rec'], batch['items'], batch["llm_labels"]
        outputs = self.rec_model(llm_inputs, reasoning=llm_labels if self.use_reasoning else None)

        for metrics in self.test_metrics:
            metrics.update(outputs, rec_labels)

    def on_test_epoch_end(self):
        for metrics in self.test_metrics:
            self.log_dict(metrics.compute(), prog_bar=True)
            metrics.reset()
