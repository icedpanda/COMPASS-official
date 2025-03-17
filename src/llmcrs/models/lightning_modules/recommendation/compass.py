import time
from typing import List, Union

import pandas as pd
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MeanMetric
from transformers.utils import ModelOutput

from src.llmcrs.data.datatype import TaskType
from src.llmcrs.metrics import get_gen_metrics
from src.llmcrs.models.lightning_modules import BaseLightningModule
from src.llmcrs.models.recommendation.compass import COMPASS


class COMPASSLightning(BaseLightningModule):
    def __init__(
            self,
            net: Union[COMPASS],
            item_entity_ids: List[int],
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            is_compile: bool = False,
            warmup_steps: int = None,
            total_steps: int = None,
            base_lr: float = None,
            base_weight_decay: float = None,
            kg_lr: float = None,
            kg_weight_decay: float = None,
            phase: TaskType = TaskType.REC,
            id2entity: dict = None,
            is_ddp: bool = False,
            is_kg_freeze: bool = False,
            llm_caption_test: bool = True
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["item_entity_ids", "net"], logger=False)
        self.rec_model = net
        self.compile = is_compile
        self.phase = phase
        # other metrics
        self.item_entity_ids = item_entity_ids.copy()
        # if using ddp mode, need to set flag to metrics
        self.is_ddp = is_ddp
        self.init_metrics()
        self.id2entity = id2entity
        self.is_kg_freeze = is_kg_freeze
        self.llm_caption_test = llm_caption_test

        if self.phase == TaskType.REC and self.is_kg_freeze:
            self.freeze_kg()

    def freeze_kg(self):
        for param in self.rec_model.kg_embeds.parameters():
            param.requires_grad = False
        for para in self.rec_model.graph_encoder.parameters():
            para.requires_grad = False

    def init_metrics(self):
        self.train_lm_avg_loss = MeanMetric()
        self.val_lm_avg_loss = MeanMetric()
        self.test_gen_metrics = get_gen_metrics(prefix=f"test/{self.phase.value}/")

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
            self.rec_model = torch.compile(self.rec_model, fullgraph=True)

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

        # KG model parameters
        if self.hparams.kg_lr is not None:
            self.print("Using KG specific learning rate and weight decay...")
            self.print(f"KG model lr: {self.hparams.kg_lr}, weight decay: {self.hparams.kg_weight_decay}", )
            kg_params = [p for p in self.rec_model.kg_embeds.parameters() if p.requires_grad] + \
                        [p for p in self.rec_model.graph_encoder.parameters() if p.requires_grad]

            add_param_group(kg_params, self.hparams.kg_lr, self.hparams.kg_weight_decay, "gnn")
            all_params -= set(kg_params)  # Remove kg_params from all_params

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

    def _rec_llm_step(self, batch, mode: str = "train"):
        context_entities = batch['context_entities']
        llm_inputs, llm_outs = batch['llm_inputs'], batch["llm_labels"]
        if mode == "train":
            outputs = self.rec_model(context_entities, llm_inputs, llm_outs)
            self.train_lm_avg_loss.update(outputs.loss_lm)
            return outputs.loss

        elif mode == "val":
            outputs = self.rec_model(context_entities, llm_inputs, llm_outs)
            self.val_lm_avg_loss.update(outputs.loss_lm)
            return outputs.loss

        elif mode == "test":
            self.print("LLM Generating text...")
            t0 = time.time()
            outputs = self.rec_model.generate(context_entities=context_entities, llm_inputs=llm_inputs, labels=llm_outs)
            self.print("Generation done!")
            self.print(f"Time taken for LLM generation: {time.time() - t0}")
            self.test_gen_metrics.update(outputs.generated_text, llm_outs)
            # log table
            # make labels to list and save to pandas dataframe
            if self.llm_caption_test:
                temp_df = pd.DataFrame({
                    "prompt": llm_inputs,
                    "generated_text": outputs.generated_text,
                    "label": llm_outs,
                    "context_entities": self.convert_entity_to_text(context_entities)},
                    columns=self.llm_test_table.columns)
                self.llm_test_table = pd.concat([self.llm_test_table, temp_df], ignore_index=True)

    @staticmethod
    def convert_items_to_int(items):
        # convert item (batch_size) to shape of (batch_size, 1)
        items = items.unsqueeze(1)
        items = items.cpu().numpy()
        return items

    def training_step(self, batch, batch_idx):
        batch = self._prepare_model_args(batch, mode="train")
        loss = self._rec_llm_step(batch)
        return loss

    def on_train_epoch_end(self):
        self.log(f"train/{self.phase.value}/lm_epoch_loss", self.train_lm_avg_loss.compute(), prog_bar=True)
        self.train_lm_avg_loss.reset()

    def on_validation_start(self):
        # if use wandb logger, use wandb define metrics
        # only in rank 0
        if self.trainer.is_global_zero:
            if self.trainer.current_epoch == 0 and any(
                    isinstance(logger, WandbLogger) for logger in self.trainer.loggers):
                self.print("Using wandb logger, define rec metrics")
                wandb.define_metric(f"val/{self.phase.value}/lm_epoch_loss", summary="min")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch = self._prepare_model_args(batch, mode="val")
        loss = self._rec_llm_step(batch, "val")
        return loss

    def on_validation_epoch_end(self):
        self.log(f"val/{self.phase.value}/num_epoch", float(self.current_epoch), prog_bar=False)
        lm_val_loss = self.val_lm_avg_loss.compute()
        self.log(f"val/{self.phase.value}/lm_epoch_loss", lm_val_loss, prog_bar=True)
        self.val_lm_avg_loss.reset()
        return lm_val_loss

    def on_test_epoch_start(self) -> None:
        # create empty pandas dataframe for test results
        if self.llm_caption_test:
            self.llm_test_table = pd.DataFrame(columns=["prompt", "generated_text", "label", "rec_items"])

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        batch = self._prepare_model_args(batch, mode="test")
        _ = self._rec_llm_step(batch, "test")

    def on_test_epoch_end(self):
        self.log_dict(self.test_gen_metrics.compute(), prog_bar=True)
        self.test_gen_metrics.reset()
        for logger in self.trainer.loggers:
            if isinstance(logger, WandbLogger):
                logger.log_table(f"{self.phase.value}/test/llm/generated_text", dataframe=self.llm_test_table)

    def _prepare_model_args(self, batch, mode: str = "train"):
        # if self.phase == TaskType.PRE:
        #     # two dataloaders
        #     # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.utilities.combined_loader.html
        #     if mode == "train":
        #         # max size of the batch
        #         node_alignment_batch = batch['kg']
        #         context_alignment_batch = batch['context']
        #         return node_alignment_batch, context_alignment_batch
        return batch

    def convert_entity_to_text(self, entity_ids):
        # convert entity ids to list of list
        entity_ids = entity_ids.detach().cpu().numpy().tolist()
        list_of_entities = []
        for entity_list in entity_ids:
            list_of_entity = [self.id2entity[entity] for entity in entity_list if entity != 0]  # 0 is padding
            list_of_entities.append(list_of_entity)

        return list_of_entities

    def predict_step(self, batch, batch_idx):
        utt_ids = batch['utterance_id']
        context_entities = batch['context_entities']
        llm_inputs, llm_outs = batch['llm_inputs'], batch["llm_labels"]
        self.print("LLM Generating text...")
        t0 = time.time()
        outputs = self.rec_model.generate(context_entities=context_entities, llm_inputs=llm_inputs, labels=llm_outs)
        self.print("Generation done!")
        self.print(f"Time taken for LLM generation: {time.time() - t0}")

        return llm_inputs, outputs.generated_text, llm_outs, self.convert_entity_to_text(
            context_entities), outputs.last_hidden_states.cpu(), utt_ids
