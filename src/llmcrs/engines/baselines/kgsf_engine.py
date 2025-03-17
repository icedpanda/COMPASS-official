from typing import Tuple

from hydra.utils import instantiate
from lightning import LightningModule
from loguru import logger
from omegaconf import DictConfig

from src.llmcrs.data import RedialDataset, ReDialDataModule
from src.llmcrs.data.datatype import TaskType
from src.llmcrs.engines import BaseSystem
from src.llmcrs.models.lightning_modules import KGSFLightning
from src.llmcrs.utils import log_hyperparameters, log_model_parameters


class KGSFSystem(BaseSystem):
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
        edge = self.dataset.kg["edge"]
        n_entity = len(self.dataset.kg["entity2id"]) + 1
        n_relation = len(self.dataset.kg["id2relation"])
        word_edge = self.dataset.kg["word_kg"]["word_edges"]
        n_words = len(self.dataset.kg["word2id"]) + 1
        models_config = self._get_config_for_phase(phase, "model_config").pl
        logger.info(f"Initializing models ...{models_config._target_}")
        model: KGSFLightning = instantiate(
            models_config,
            net={
                "edge": list(edge),
                "n_entity": n_entity,
                "n_relation": n_relation,
                "word_edge": word_edge,
                "n_words": n_words,
            },
            item_entity_ids=self.labels_list,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

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

        # the first epoch is dedicated to warmup
        warmup_steps = (dataloader.total_steps // config.accumulate_grad_batches)
        total_steps = warmup_steps * config.max_epochs

        logger.info(f"Total steps: {total_steps}, warmup steps: {warmup_steps}")

        return total_steps, warmup_steps

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
