from hydra.utils import instantiate
from lightning import LightningModule
from loguru import logger
from omegaconf import DictConfig

from src.llmcrs.data import RedialDataset
from src.llmcrs.data.datatype import TaskType
from src.llmcrs.engines import BaseSystem
from src.llmcrs.utils import log_hyperparameters, log_model_parameters


class RedialBaseline(BaseSystem):
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
        print("models: ", models_config)
        logger.info(f"Initializing models ...{models_config._target_}")
        model: LightningModule = instantiate(
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

        use_warmup = self.is_warmup_scheduler(self.model_config[phase.value].pl.scheduler)
        if use_warmup:
            total_steps, warm_up_steps = self.log_steps(dataloader, phase)
        else:
            total_steps, warm_up_steps = None, None

        model = self.init_models(phase, warmup_steps=warm_up_steps, total_steps=total_steps)
        # Initialize the trainer
        trainer = self.init_trainer(phase, callbacks=callbacks, loggers=self.loggers)

        # log_everything
        log_hyperparameters(self.config, trainer)
        log_model_parameters(model, trainer, phase=phase)
        # model training
        trainer.fit(model, dataloader)
        logger.info(f"{phase.value} training finished")
        trainer.test(model, dataloader)
        logger.info(f"best model path: {trainer.checkpoint_callback.best_model_path}")

    def fit(self):
        self.train_rec()
