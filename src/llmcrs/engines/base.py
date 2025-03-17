import os
import shutil
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import hydra
import torch
import wandb
from lightning import LightningModule
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger as PLLogger
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger
from omegaconf import DictConfig

from src.llmcrs.data import ReDialDataModule
from src.llmcrs.data.datatype import TaskType
from src.llmcrs.utils import instantiate_loggers, instantiate_callbacks


class BaseSystem(ABC):
    """
    Base class for all systems.
    """

    def __init__(self, dataset, config: DictConfig):
        """
        self.config = config
        """
        if not torch.cuda.is_available():
            raise NotImplementedError('No GPU found, please run on GPU')

        torch.set_float32_matmul_precision(config.extras.torch_matmul_precision)  # 'medium'/'high'/'highest'
        self.dataset = dataset
        self.labels_list = list(range(dataset.kg["n_classes"]))
        self.config = config
        self.dataloader_config = config.data.dataloader
        self.model_config = config.model
        self.callbacks = config.callbacks
        self.logger_config = config.logger
        self.paths = config.paths
        self.loggers = self.init_loggers()
        self.pre_train = self.model_config.get('pre_train', False)
        logger.info(f"Pre-training: {self.pre_train}")
        trainer_config = self.model_config.align.trainer if self.pre_train else self.model_config.rec.trainer
        self.strategy = trainer_config.get('strategy', 'auto')
        self.is_ddp = self.strategy in ['ddp', 'ddp_find_unused_parameters_true', 'fsdp', 'deepspeed_stage_3', "deepspeed_stage_2"]
        logger.warning(f"training strategy: {self.strategy}")
        logger.info(f"Output dir: {self.paths.output_dir}")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        # model_name = self.model_config.rec.pl.net.get("model_name", None)
        # logger.info(f"Rec Model name: {model_name}")
        # if model_name is not None:
        #     assert model_name == dataset.tokenizer_name, \
        #             f"Model name {model_name} is not the same as the dataset tokenizer name {dataset.tokenizer_name}"

    def _get_config_for_phase(self, phase: TaskType, attribute: str) -> DictConfig:
        """Helper method to get configuration based on phase."""
        return getattr(self, attribute).get(phase.value)

    def init_dataloader(self, phase: TaskType = TaskType.REC) -> ReDialDataModule:
        """
        Initialize dataloader.
        """
        print(phase.value)
        dataloader_config = self._get_config_for_phase(phase, 'dataloader_config')
        logger.info(f"Initializing dataloader <{dataloader_config._target_}>")
        dataloader: ReDialDataModule = hydra.utils.instantiate(
            dataloader_config,
            dataset=self.dataset,
            phase=phase,
        )
        return dataloader

    def init_trainer(self, phase: TaskType = TaskType.REC, callbacks: Optional[List[Callback]] = None,
                     loggers: Optional[List[PLLogger]] = None, is_ddp_test=False) -> Trainer:
        """
        Initialize trainer.
        Args:
            phase: TaskType.REC or TaskType.CONV
            callbacks: List of callbacksll
            loggers: List of loggers
            is_ddp_test: If True, set the gpu to 1

        Returns: Trainer
        """
        trainer_config = self._get_config_for_phase(phase, 'model_config').trainer
        logger.info(f"Initializing trainer <{trainer_config._target_}>")
        if is_ddp_test:
            trainer_config.devices = 1
            trainer_config.accelerator = "gpu"
            trainer_config.strategy = "auto"
        trainer: Trainer = hydra.utils.instantiate(trainer_config, logger=loggers, callbacks=callbacks)
        return trainer

    def init_callbacks(self, phase: TaskType = TaskType.REC) -> List[Callback]:
        """
        Initialize callbacks.
        """
        callbacks_config = self._get_config_for_phase(phase, 'callbacks')
        logger.info("Initializing callbacks ...")
        callbacks: List[Callback] = instantiate_callbacks(callbacks_config)
        return callbacks

    def init_loggers(self) -> List[PLLogger]:
        """
        Initialize loggers.
        """
        logger.info("Initializing loggers ...")
        loggers: List[PLLogger] = instantiate_loggers(self.logger_config)
        # if use wandb, log the code
        # if wandb.run:
        #     wandb.run.log_code("src/")
        return loggers

    def init_models(self, phase: TaskType = TaskType.REC) -> LightningModule:
        """
        Initialize models.
        """
        models_config = self._get_config_for_phase(phase, 'model_config').pl
        logger.info(f"Initializing models ...{models_config._target_}")
        model: LightningModule = hydra.utils.instantiate(models_config)

        return model

    def log_steps(self, dataloader: ReDialDataModule, phase: TaskType = TaskType.REC) -> Tuple[int, int]:
        """
        Calculate and log the total and warmup steps for the given phase and dataloader.

        Args:
            dataloader: The dataloader to compute total steps.
            phase: The phase of training, e.g., TaskType.REC or TaskType.CONV.

        Returns:
            Tuple containing total_steps and warmup_steps.
        """

        config = self._get_config_for_phase(phase, 'model_config').trainer

        # the first half epoch is dedicated to warmup
        warmup_steps = (dataloader.total_steps // config.accumulate_grad_batches) // 2
        total_steps = warmup_steps * config.max_epochs * 2

        logger.info(f"Total steps: {total_steps}, warmup steps: {warmup_steps}")

        return total_steps, warmup_steps

    @staticmethod
    def is_warmup_scheduler(scheduler):
        """
        Checks if the scheduler is for warmup.
        Args:
            scheduler: The scheduler object.
        Returns:
            True if the scheduler is for warmup, False otherwise.
        """

        return "transformer" in getattr(scheduler, "_target_", "").lower()

    def train_pre(self, best_model_path: str = None):
        return self.train_phase(TaskType.PRE, best_model_path)

    def train_rec(self, best_model_path: str = None):
        return self.train_phase(TaskType.REC, best_model_path)

    @abstractmethod
    def train_phase(self, phase: TaskType, best_model_path: str = None):
        NotImplementedError("This method should be implemented in the subclass.")

    def fit(self):
        self.train_rec()

    @rank_zero_only
    def clean_files(self):
        wandb.finish()
        out_dir = self.paths.output_dir
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
            logger.warning(
                f"Remove {out_dir} to save disk space. You should not delete it if you want to keep the weights.")
        else:
            logger.warning(f"{out_dir} does not exist.")
