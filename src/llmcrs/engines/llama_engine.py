import json

import torch
from hydra.utils import instantiate
from lightning import LightningModule
from loguru import logger
from omegaconf import DictConfig

from src.llmcrs.data import RedialDataset, ReDialDataModule
from src.llmcrs.data.datatype import TaskType
from src.llmcrs.engines import BaseSystem
from src.llmcrs.models.lightning_modules import LLMCRSLightning
from src.llmcrs.utils import log_hyperparameters, log_model_parameters


class LLaMASystem(BaseSystem):
    def __init__(self, dataset: RedialDataset, config: DictConfig):
        super().__init__(dataset, config)

    def init_models(self,
                    phase: TaskType = TaskType.REC,
                    warmup_steps: int = None,
                    total_steps: int = None,
                    best_rec_model_path: str = None) -> LightningModule:
        """
        Initialize callbacks.
        """
        config = self._get_config_for_phase(phase, "model_config").pl
        logger.info(f"Initializing models for {phase} phase... {config._target_}")
        model: LLMCRSLightning = instantiate(
            config,
            item_entity_ids=self.labels_list,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            phase=phase,
            id2entity=self.dataset.kg["id2entity"],
        )
        logger.info("Model initialized successfully!")
        logger.info(f"Phase: {phase} Model path: {best_rec_model_path}")
        if best_rec_model_path:
            logger.info(f"INTI MODEL: Loading best model from {best_rec_model_path}")
            ckpt = torch.load(best_rec_model_path)
            model.load_state_dict(ckpt["state_dict"])

        return model

    def init_dataloader(self, phase: TaskType = TaskType.REC) -> ReDialDataModule:
        dataloader_config = self._get_config_for_phase(phase, 'dataloader_config')
        logger.info(f"Initializing dataloader <{dataloader_config._target_}>")
        dataloader: ReDialDataModule = instantiate(
            dataloader_config,
            dataset=self.dataset,
            phase=phase,
        )
        return dataloader

    def train_phase(self, phase: TaskType, best_model_path: str = None):

        logger.info(f"{phase.value} training started")
        dataloader = self.init_dataloader(phase)
        callbacks = self.init_callbacks(phase)
        use_warmup = self.is_warmup_scheduler(self.model_config[phase.value].pl.scheduler)
        if use_warmup:
            total_steps, warm_up_steps = self.log_steps(dataloader, phase)
        else:
            total_steps, warm_up_steps = None, None

        model = self.init_models(phase, warmup_steps=warm_up_steps, total_steps=total_steps,
                                 best_rec_model_path=best_model_path)
        # Initialize the trainer
        trainer = self.init_trainer(phase, callbacks=callbacks, loggers=self.loggers)

        # log_everything
        log_hyperparameters(self.config, trainer)
        log_model_parameters(model, trainer, phase=phase)
        logger.info(f"Start {phase.value} training")
        # model training
        trainer.fit(model, dataloader)
        logger.info(f"{phase.value} training finished")
        logger.info("start testing !")
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        logger.info(f"best model path: {best_checkpoint}")
        trainer.test(model, dataloader)

        return best_checkpoint

    def fit(self):
        self.train_rec()
        self.clean_files()

