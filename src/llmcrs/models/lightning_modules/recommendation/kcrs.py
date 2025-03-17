import time
from typing import List, Union

import pandas as pd
import torch
import torch.nn as nn
import wandb
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MeanMetric

from src.llmcrs.data.datatype import TaskType
from src.llmcrs.metrics import get_rec_metrics, get_node_alignment_metrics, get_context_alignment_metrics, \
    get_gen_metrics
from src.llmcrs.models.lightning_modules import BaseLightningModule
from src.llmcrs.models.recommendation import GraphBridge
from src.llmcrs.models.recommendation import KCRS
from src.llmcrs.models.recommendation.modules.bridge_outputs import GraphBridgeOutput


class KCRSLightning(BaseLightningModule):
    def __init__(
            self,
            net: Union[KCRS, GraphBridge],
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
            bridge_caption_test: bool = True,
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
        self.bridge_caption_test = bridge_caption_test
        self.llm_caption_test = llm_caption_test

        if self.phase == TaskType.REC and self.is_kg_freeze:
            self.freeze_kg()

    def freeze_kg(self):
        for param in self.rec_model.kg_embeds.parameters():
            param.requires_grad = False
        for para in self.rec_model.graph_encoder.parameters():
            para.requires_grad = False

    def init_metrics(self):
        if self.phase == TaskType.PRE:
            self.setup_alignment_metrics()
        else:
            self.train_lm_avg_loss = MeanMetric()
            self.val_lm_avg_loss = MeanMetric()
            self.train_rec_avg_loss = MeanMetric()
            self.val_rec_avg_loss = MeanMetric()
            self.test_gen_metrics = get_gen_metrics(bert_max_len=300, prefix=f"test/{self.phase.value}/")
            self.setup_rec_metrics()

    def setup_alignment_metrics(self):
        # There are seven types of losses in total: itm_loss, lm_loss, itc loss,
        # each at both the node level and contextual level, plus an additional rec loss for validation and test.
        # need to monitor all of them
        self.train_avg_total_loss = MeanMetric()  # total avg loss of these alignments
        self.train_node_alignment_metrics = get_node_alignment_metrics("train")
        self.train_context_alignment_metrics = get_context_alignment_metrics("train")
        self.val_node_alignment_metrics = get_node_alignment_metrics("val")
        self.val_context_alignment_metrics = get_context_alignment_metrics("val")
        self.test_context_alignment_metrics = get_context_alignment_metrics("test")
        self.test_node_alignment_metrics = get_node_alignment_metrics("test")
        self.test_context_alignment_gen_metrics = get_gen_metrics(bert_max_len=300, prefix=f"test/{self.phase.value}/")
        self.test_node_alignment_gen_metrics = get_gen_metrics(bert_max_len=300, prefix=f"test/{self.phase.value}/")

    def setup_rec_metrics(self):
        self.target_metrics = [f"val/{self.phase.value}/HitRate@10", f"val/{self.phase.value}/HitRate@50"]
        # self.target_metrics = [f"val/align/HitRate@1", f"val/align/HitRate@50"]
        metrics_list = get_rec_metrics(self.item_entity_ids)
        self.val_metrics = nn.ModuleList([metric.clone(prefix=f"val/{self.phase.value}/") for metric in metrics_list])
        self.test_metrics = nn.ModuleList([metric.clone(prefix=f"test/{self.phase.value}/") for metric in metrics_list])

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

        if self.phase == TaskType.PRE:
            # Pre-training model parameters
            bridge_params = [p for p in self.rec_model.bridge.parameters() if p.requires_grad]
            add_param_group(bridge_params, self.hparams.base_lr, self.hparams.base_weight_decay, "bridge")
            all_params -= set(bridge_params)
        elif self.phase == TaskType.REC:
            # Recommendation model parameters
            # base_params = [p for p in self.rec_model.model.parameters() if p.requires_grad]
            # add_param_group(base_params, self.hparams.base_lr, self.hparams.base_weight_decay, "llm")
            # # Base model parameters
            # all_params -= set(base_params)  # Remove base_params from all_params
            bridge_params = [p for p in self.rec_model.bridge.parameters() if p.requires_grad]
            add_param_group(bridge_params, self.hparams.base_lr, self.hparams.base_weight_decay, "bridge")
            all_params -= set(bridge_params)

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

    def _alignment_step(self, batch, mode: str = "train", alignment_type: str = "node"):
        prompt, caption = batch['prompt'], batch['caption']
        if alignment_type == "node":
            # node alignment
            node_idx = batch['node_idx']
            loss = self.rec_model(node_idx, caption, prompt, caption, "node_text")
            self.update_node_alignment_metrics(mode, loss)
            if mode == "test" and self.bridge_caption_test:
                self.print("Generating node text...")
                # measure the time for generation
                t0 = time.time()
                generated_text = self.rec_model.generate(node_idx, prompt, "node_text", max_new_tokens=224)
                self.test_node_alignment_gen_metrics.update(generated_text, caption)
                self.print(f"Time taken for node text generation: {time.time() - t0}")
                node_idx = self.convert_entity_to_text(node_idx)
                # save to wandb
                # prepare the data for pandas dataframe table
                temp_df = pd.DataFrame({
                    "nodes": node_idx,
                    "generated_text": generated_text,
                    "label": caption
                }, columns=self.test_node_table.columns)

                self.test_node_table = pd.concat([self.test_node_table, temp_df], ignore_index=True)
                del temp_df
        else:
            # context alignment
            context_entities, context_tokens = batch['context_entities'], batch['context_tokens']
            loss = self.rec_model(context_entities, context_tokens, prompt, caption, "contextual")
            self.update_context_alignment_metrics(mode, loss)
            if mode == "test" and self.bridge_caption_test:
                self.print("Generating context keywords text...")
                t0 = time.time()
                generated_text = self.rec_model.generate(context_entities, prompt, "contextual", max_new_tokens=192)
                self.test_context_alignment_gen_metrics.update(generated_text, caption)
                self.print(f"Time taken for context keywords generation: {time.time() - t0}")
                context_entities = self.convert_entity_to_text(context_entities)
                temp_df = pd.DataFrame({
                    "nodes": context_entities,
                    "generated_text": generated_text,
                    "label": caption
                }, columns=self.test_context_table.columns)
                self.test_context_table = pd.concat([self.test_context_table, temp_df], ignore_index=True)
                del temp_df

        return loss.loss

    def _rec_step(self, batch, mode: str = "train"):
        context_entities, context_tokens, labels = batch['context_entities'], batch['context_tokens'], batch["items"]
        outputs = self.rec_model.make_rec(context_entities, context_tokens)
        if mode == "train":
            self.train_rec_avg_loss.update(outputs.loss)
            return outputs.loss
        elif mode == "val":
            self.val_rec_avg_loss.update(outputs.loss)
            for metrics in self.val_metrics:
                metrics.update(outputs.rec_output.sims, labels)
            return outputs.loss
        else:
            for metrics in self.test_metrics:
                metrics.update(outputs.rec_output.sims, labels)

    def _rec_llm_step(self, batch, mode: str = "train"):
        context_entities = batch['context_entities']
        llm_inputs, bridge_inputs, llm_outs = batch['llm_inputs'], batch['bridge_inputs'], batch["llm_labels"]
        rec_labels = batch['items']
        if mode == "train":
            outputs = self.rec_model(context_entities, llm_inputs, bridge_inputs, llm_outs, rec_labels)
            self.train_rec_avg_loss.update(outputs.rec_output.loss)
            self.train_lm_avg_loss.update(outputs.loss_lm)
            return outputs.loss

        elif mode == "val":
            outputs = self.rec_model(context_entities, llm_inputs, bridge_inputs, llm_outs, rec_labels)
            self.val_rec_avg_loss.update(outputs.rec_output.loss)
            self.val_lm_avg_loss.update(outputs.loss_lm)
            for metrics in self.val_metrics:
                metrics.update(outputs.rec_output.sims, rec_labels)
            return outputs.loss

        elif mode == "test":
            # num_caption = context_entities.shape[0]
            self.print("LLM Generating text...")
            t0 = time.time()
            outputs = self.rec_model.generate(context_entities, llm_inputs, bridge_inputs, max_new_tokens=400)
            self.print("Generation done!")
            self.print(f"Time taken for LLM generation: {time.time() - t0}")
            for metrics in self.test_metrics:
                metrics.update(outputs.rec_output.sims, rec_labels)
            self.test_gen_metrics.update(outputs.generated_text, llm_outs)
            # log table
            # make labels to list and save to pandas dataframe
            if self.llm_caption_test:
                temp_df = pd.DataFrame({
                    "prompt": llm_inputs,
                    "generated_text": outputs.generated_text,
                    "label": llm_outs,
                    "rec_items": self.convert_items_to_int(rec_labels),
                    "context_entities": self.convert_entity_to_text(context_entities)},
                    columns=self.llm_test_table.columns)
                self.llm_test_table = pd.concat([self.llm_test_table, temp_df], ignore_index=True)

    @staticmethod
    def convert_items_to_int(items):
        # convert item (batch_size) to shape of (batch_size, 1)
        items = items.unsqueeze(1)
        items = items.cpu().numpy()
        return items

    @staticmethod
    def _update_node_metrics(metrics, mode, loss: GraphBridgeOutput):
        metrics[f'{mode}/align/avg_node_loss'].update(loss.loss)
        metrics[f'{mode}/align/lm_node_loss'].update(loss.loss_lm)
        metrics[f'{mode}/align/itm_node_loss'].update(loss.loss_itm)
        metrics[f'{mode}/align/itc_node_loss'].update(loss.loss_itc)

    @staticmethod
    def _update_context_metrics(metrics, mode, loss: GraphBridgeOutput):
        metrics[f'{mode}/align/avg_context_loss'].update(loss.loss)
        metrics[f'{mode}/align/lm_context_loss'].update(loss.loss_lm)
        metrics[f'{mode}/align/itm_context_loss'].update(loss.loss_itm)
        metrics[f'{mode}/align/itc_context_loss'].update(loss.loss_itc)

    def update_node_alignment_metrics(self, mode, loss: GraphBridgeOutput):
        if mode == "train":
            self._update_node_metrics(self.train_node_alignment_metrics, mode, loss)
        elif mode == "val":
            self._update_node_metrics(self.val_node_alignment_metrics, mode, loss)
        else:
            self._update_node_metrics(self.test_node_alignment_metrics, mode, loss)

    def update_context_alignment_metrics(self, mode, loss):
        if mode == "train":
            self._update_context_metrics(self.train_context_alignment_metrics, mode, loss)
        elif mode == "val":
            self._update_context_metrics(self.val_context_alignment_metrics, mode, loss)
        else:
            self._update_context_metrics(self.test_context_alignment_metrics, mode, loss)

    def training_step(self, batch, batch_idx):
        if self.phase == TaskType.PRE:
            # two dataloaders
            node_batch, context_batch = self._prepare_model_args(batch, mode="train")
            node_loss = self._alignment_step(node_batch, alignment_type="node", )
            context_loss = self._alignment_step(context_batch, alignment_type="context", )
            loss = node_loss + context_loss
            self.train_avg_total_loss.update(loss)
        else:
            batch = self._prepare_model_args(batch, mode="train")
            loss = self._rec_llm_step(batch)
        return loss

    def on_train_epoch_end(self):
        if self.phase == TaskType.REC:
            self.log("train/rec/epoch_loss", self.train_rec_avg_loss.compute(), prog_bar=True)
            self.train_rec_avg_loss.reset()
            self.train_lm_avg_loss.reset()
        if self.phase == TaskType.PRE:
            self.log("train/align/avg_total_loss", self.train_avg_total_loss.compute(), prog_bar=True)
            self.train_avg_total_loss.reset()
            for metrics_list in [self.train_node_alignment_metrics, self.train_context_alignment_metrics]:
                for key, metric in metrics_list.items():
                    self.log(key, metric.compute(), prog_bar=False)
                    metric.reset()

    def on_validation_start(self):
        # if use wandb logger, use wandb define metrics
        # only in rank 0
        if self.trainer.is_global_zero:
            if self.trainer.current_epoch == 0 and any(
                    isinstance(logger, WandbLogger) for logger in self.trainer.loggers):
                if self.phase == TaskType.REC:
                    self.print("Using wandb logger, define rec metrics")
                    wandb.define_metric(f"val/{self.phase.value}/rec_epoch_loss", summary="min")
                    wandb.define_metric(f"val/{self.phase.value}/lm_epoch_loss", summary="min")
                    for metrics in self.val_metrics:
                        for keys in metrics.keys():
                            wandb.define_metric(keys, summary="max")

                if self.phase == TaskType.PRE:
                    self.print("defining alignment metrics")
                    wandb.define_metric("val/align/avg_total_loss", summary="min")
                    metrics_list = [self.val_node_alignment_metrics, self.val_context_alignment_metrics]
                    for metrics in metrics_list:
                        for key in metrics.keys():
                            wandb.define_metric(key, summary="min")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch = self._prepare_model_args(batch, mode="val")
        if self.phase == TaskType.PRE:
            if dataloader_idx == 0:
                # node alignment dataloader
                loss = self._alignment_step(batch, "val", "node")
            else:
                # context alignment dataloader
                loss = self._alignment_step(batch, "val", "context")
            return loss
        else:
            loss = self._rec_llm_step(batch, "val")
            return loss

    def on_validation_epoch_end(self):
        self.log(f"val/{self.phase.value}/num_epoch", float(self.current_epoch), prog_bar=False)
        if self.phase == TaskType.REC:
            val_loss = self.val_rec_avg_loss.compute()
            lm_val_loss = self.val_lm_avg_loss.compute()
            self.log(f"val/{self.phase.value}/rec_epoch_loss", val_loss, prog_bar=True)
            self.log(f"val/{self.phase.value}/lm_epoch_loss", lm_val_loss, prog_bar=True)
            self.val_rec_avg_loss.reset()
            self.val_lm_avg_loss.reset()
            target_metrics = 0
            for metric in self.val_metrics:
                result = metric.compute()
                self.log_dict(result, prog_bar=True)
                # extract and sum the target metrics
                for key in self.target_metrics:
                    if key in result:
                        target_metrics += result[key]
                metric.reset()
            self.log(f"val/{self.phase.value}/target_metrics", target_metrics, prog_bar=True)

            return val_loss + lm_val_loss

        if self.phase == TaskType.PRE:
            total_loss = 0
            for metrics in [self.val_node_alignment_metrics, self.val_context_alignment_metrics]:
                for key, metric in metrics.items():
                    out = metric.compute()
                    self.log(key, out, prog_bar=False)
                    if key == "val/align/avg_context_loss" or key == "val/align/avg_node_loss":
                        total_loss += out
                    metric.reset()
            if total_loss == 0:
                raise ValueError("Node or context loss is zero!")
            self.log("val/align/avg_total_loss", total_loss, prog_bar=False)
            return total_loss

    def on_test_epoch_start(self) -> None:
        # create empty pandas dataframe for test results
        # columns = ["prompt", "generated_text", "nodes", "label"]
        if self.phase == TaskType.PRE and self.bridge_caption_test:
            self.test_node_table = pd.DataFrame(columns=["nodes", "generated_text", "label"])
            self.test_context_table = pd.DataFrame(columns=["nodes", "generated_text", "label"])
        if self.phase == TaskType.REC and self.llm_caption_test:
            self.llm_test_table = pd.DataFrame(
                columns=["prompt", "generated_text", "label", "rec_items", "context_entities"])

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        batch = self._prepare_model_args(batch, mode="test")
        if self.phase == TaskType.PRE:
            if dataloader_idx == 0:
                # node alignment dataloader
                _ = self._alignment_step(batch, "test", "node")
            else:
                # context alignment dataloader
                _ = self._alignment_step(batch, "test", "context")
        else:
            _ = self._rec_llm_step(batch, "test")

    def on_test_epoch_end(self):
        if self.phase == TaskType.REC:
            for metrics in self.test_metrics:
                self.log_dict(metrics.compute(), prog_bar=True)
                metrics.reset()
            self.log_dict(self.test_gen_metrics.compute(), prog_bar=True)
            self.test_gen_metrics.reset()
            for logger in self.trainer.loggers:
                if isinstance(logger, WandbLogger):
                    logger.log_table("test/llm/generated_text", dataframe=self.llm_test_table)

        if self.phase == TaskType.PRE:
            for metrics in [self.test_node_alignment_metrics, self.test_context_alignment_metrics]:
                for key, metric in metrics.items():
                    out = metric.compute()
                    self.log(key, out, prog_bar=False)
                    metric.reset()
            self.log_dict(self.test_node_alignment_gen_metrics.compute(), prog_bar=False)
            self.log_dict(self.test_context_alignment_gen_metrics.compute(), prog_bar=False)
            self.test_node_alignment_gen_metrics.reset()
            self.test_context_alignment_gen_metrics.reset()
            # save the table to wandb
            if self.bridge_caption_test:
                for logger in self.trainer.loggers:
                    if isinstance(logger, WandbLogger):
                        logger.log_table("test/align/generated_text", dataframe=self.test_node_table)
                        logger.log_table("test/align/context_generated_text", dataframe=self.test_context_table)

    def _prepare_model_args(self, batch, mode: str = "train"):
        if self.phase == TaskType.PRE:
            # two dataloaders
            # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.utilities.combined_loader.html
            if mode == "train":
                # max size of the batch
                node_alignment_batch = batch['kg']
                context_alignment_batch = batch['context']
                return node_alignment_batch, context_alignment_batch

        return batch

    def convert_entity_to_text(self, entity_ids):
        # convert entity ids to list of list
        entity_ids = entity_ids.detach().cpu().numpy().tolist()
        list_of_entities = []
        for entity_list in entity_ids:
            list_of_entity = [self.id2entity[entity] for entity in entity_list if entity != 0]  # 0 is padding
            list_of_entities.append(list_of_entity)

        return list_of_entities
