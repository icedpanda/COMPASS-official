import logging

from lightning import LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger as log
from omegaconf import DictConfig, OmegaConf

from src.llmcrs.data.datatype import TaskType


# This handler will be used to intercept standard logging messages toward loguru
class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = log.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find the caller from where originated the logged message
        frame, depth = logging.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        log.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


# Ensure all loggings are intercepted (especially from libraries like PyTorch Lightning)
def intercept_standard_logging_messages():
    logging.root.handlers = [InterceptHandler()]
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True


@rank_zero_only
def setup_lightning_logging():
    intercept_standard_logging_messages()
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)


@rank_zero_only
def log_hyperparameters(config: DictConfig,
                        trainer: Trainer,
                        ) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Args:
        config: The config to log.
        trainer: The trainer instance.

    """
    config = OmegaConf.to_container(config)
    keys = ["model", "data", "dataset", "seed", "callbacks", "task_name", "tags"]
    hparams = {k: config.get(k) for k in keys}
    hparams["tags"] = config.get("tags")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


@rank_zero_only
def log_model_parameters(
        model: LightningModule,
        trainer: Trainer,
        phase: TaskType = TaskType.REC,
) -> None:
    """
    Controls which model parameters are saved by Lightning loggers.

    Args:
        model: The model to log.
        trainer:  The trainer instance.
        phase: The phase of the model.

    Returns:
        None
    """
    params_info = {
        f"model/{phase.value}/params_total": sum(p.numel() for p in model.parameters()),
        f"model/{phase.value}/params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        f"model/{phase.value}/params_not_trainable": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }

    for logger in trainer.loggers:
        logger.log_hyperparams(params_info)
