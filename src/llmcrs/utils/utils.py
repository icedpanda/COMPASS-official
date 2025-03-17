import warnings
from pathlib import Path
from typing import Sequence

import rich
import rich.syntax
import rich.tree
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.utilities import rank_zero_only
from rich.prompt import Prompt
from torch import set_float32_matmul_precision


@rank_zero_only
def print_config(cfg: DictConfig,
                 print_order: Sequence[str] = ("data", "model", "callbacks", "logger", "paths", "extras", "seed",),
                 resolve: bool = False, save_to_file: bool = True, ) -> None:
    """Prints the contents of a DictConfig as a tree structure using the Rich library.

    Args:
        cfg: A DictConfig composed by Hydra.
        print_order: Determines in what order config components are printed. Default is ``("data", "model",
            "callbacks", "logger", "trainer", "paths", "extras")``.
        resolve: Whether to resolve reference fields of DictConfig. Default is ``False``.
        save_to_file: Whether to export config to the hydra output folder. Default is ``False``.

    Returns:
        None

    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else logger.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing...")

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as f:
            rich.print(tree, file=f)


def valid_args_for_class(cls):
    """Return a list of valid argument names for the class initializer."""
    return list(cls.__init__.__code__.co_varnames)


@rank_zero_only
def enforce_tags(cfg: DictConfig) -> None:
    """Prompts user to input tags from command line if no tags are provided in config.

    Args:
        cfg: A DictConfig composed by Hydra.
    Returns:
        None
    """
    if not cfg.get("tags"):
        logger.warning("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

    logger.info(f"Tags: {cfg.tags}")


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        logger.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        logger.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        logger.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        enforce_tags(cfg)

    if cfg.extras.get("torch_matmul_precision"):
        logger.info(f"Setting torch matmul precision! {cfg.extras.torch_matmul_precision}")
        set_float32_matmul_precision(cfg.extras.torch_matmul_precision)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        logger.info("Printing config tree with Rich!")
        print_config(cfg=cfg, resolve=True, save_to_file=True)
