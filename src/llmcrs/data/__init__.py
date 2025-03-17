from src.llmcrs.data.dataloader import ReDialDataModule
from src.llmcrs.data.dataset import RedialDataset, CRSDataset, InspiredDataset

dataset_table = {
    "ReDial": RedialDataset,
    "INSPIRED": InspiredDataset,
}


def get_dataset(dataset_name: str) -> CRSDataset:
    """
    Get a dataset based on the provided options.

    Parameters:
        dataset_name (str): The name of the dataset.


    Returns:
        The dataset based on the provided options.

    Raises:
        ValueError: If the dataset name is not supported.
    """
    if dataset_name not in dataset_table:
        raise ValueError(f"dataset {dataset_name} not supported. Supported datasets are {dataset_table.keys()}")
    return dataset_table[dataset_name]
