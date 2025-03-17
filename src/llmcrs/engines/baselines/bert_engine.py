from hydra.utils import instantiate
from lightning import LightningModule
from loguru import logger
from omegaconf import DictConfig

from src.llmcrs.data import RedialDataset
from src.llmcrs.data.datatype import TaskType
from src.llmcrs.engines import BaseSystem
from src.llmcrs.models.lightning_modules import BERTLightning
from src.llmcrs.utils import log_hyperparameters, log_model_parameters


class BERTSystem(BaseSystem):
    def __init__(self, dataset: RedialDataset, config: DictConfig):
        super().__init__(dataset, config)

    def init_models(self,
                    phase: TaskType = TaskType.REC,
                    warmup_steps: int = None,
                    total_steps: int = None
                    ) -> LightningModule:
        """
        Initialize callbacks.
        """
        models_config = self._get_config_for_phase(phase, "model_config").pl
        logger.info(f"Initializing models ...{models_config._target_}")
        model: BERTLightning = instantiate(
            models_config,
            net={
                "n_classes": self.dataset.kg["n_classes"],
            },
            item_entity_ids=self.labels_list,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        return model

    def train_phase(self, phase: TaskType, best_model_path: str = None):
        logger.info(f"{phase.value} training started")
        dataloader = self.init_dataloader(phase)
        callbacks = self.init_callbacks(phase)

        model = self.init_models(TaskType.REC)
        trainer = self.init_trainer(TaskType.REC, callbacks=callbacks, loggers=self.loggers)

        # log_everything
        log_hyperparameters(self.config, trainer)
        log_model_parameters(model, trainer, phase=TaskType.REC)

        # model training
        trainer.fit(model, dataloader)
        logger.info("recognition training finished")
        trainer.test(model, dataloader)
        logger.info(f"best model path: {trainer.checkpoint_callback.best_model_path}")

    def fit(self):
        self.train_rec()
        self.clean_files()
