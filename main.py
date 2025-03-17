import logging
import os
import hydra
from lightning import seed_everything
from omegaconf import DictConfig
from loguru import logger

from src.quickstart import run_crs
from src.llmcrs.utils import extras
from src.llmcrs.utils.logging_utils import setup_lightning_logging

# set Environment Variable
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# # Replace Python's standard logging with loguru
# # Set up the InterceptHandler for all lightning loggers
# Call the utility function to set up logging for lightning
setup_lightning_logging()


@hydra.main(config_path="configs", config_name="default.yaml", version_base="1.3")
def main(cfg: DictConfig):
    log_file_path = os.path.join(cfg.paths.output_dir, "logfile.log")
    logger.add(log_file_path, rotation="500 MB", level="TRACE")

    # apply extra utilities
    extras(cfg)

    if cfg.get("seed"):
        logger.info(f"Seed: {cfg.seed}")
        seed_everything(cfg.seed, workers=True)
    run_crs(cfg)

if __name__ == "__main__":
    main()
