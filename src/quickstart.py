from omegaconf import DictConfig, OmegaConf

from src.llmcrs.data import get_dataset
from src.llmcrs.engines import get_system
from src.llmcrs.utils import valid_args_for_class


def instantiate_dataset(opts: DictConfig):
    """
    Instantiate the dataset based on the provided options.

    Args:
        opts: The options containing dataset configuration.

    Returns:
        dataset: An instance of the dataset.
    """
    dataset_class = get_dataset(opts.data.dataset.dataset_name)
    valid_args = valid_args_for_class(dataset_class)

    # Filter the arguments based on valid args for the dataset class
    args_dict = OmegaConf.to_container(opts.data.dataset, resolve=True)
    filtered_args = {k: v for k, v in args_dict.items() if k in valid_args}

    return dataset_class(**filtered_args)


def run_crs(opts: DictConfig):
    """
    Run the CRS system.

    Args:
        opts: The options containing the configuration for the system.
    """
    dataset = instantiate_dataset(opts)
    system_class = get_system(opts.model.system_name)
    system = system_class(dataset, opts)
    system.fit()
