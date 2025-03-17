from typing import Type

from .bert import BERTModel, ALLOWED_BERT_MODELS
from .kbrd import KBRD
from .kgsf import KGSF
from .llama import LLAMA
from .compass import COMPASS

MODEL_METADATA = [
    {
        "prefixes": ALLOWED_BERT_MODELS,
        "class": BERTModel
    },
    # TODO: Add other models here
    # Add other models' metadata here
]


def get_model_class(model_name: str) -> Type:
    """
    Get the model class based on the model name.

    Args:
        model_name (str): The name of the model (e.g., "bert-base-uncased").

    Returns:
        The model class corresponding to the model name.

    Raises:
        ValueError: If the model name is not supported.
    """
    for model_meta in MODEL_METADATA:
        for prefix in model_meta["prefixes"]:
            if model_name.startswith(prefix):
                return model_meta["class"]
    raise ValueError(f"Model {model_name} not supported")


def get_rec_model(model_name: str) -> Type:
    """
    Get a recommendation model based on the model name.

    Args:
        model_name: (str): The name of the model (e.g., "bert-base-uncased").

    Returns:
        The recommendation model class based on the model name.
    """
    return get_model_class(model_name)
