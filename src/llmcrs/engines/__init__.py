from typing import Type

from .base import BaseSystem
from .baselines import RedialBaseline, KBRDSystem, BERTSystem, KGSFSystem
from .compass_engine import COMPASSSystem
from .llama_engine import LLaMASystem


def get_system(name: str) -> Type[BaseSystem]:
    """
    Return the corresponding system based on the provided name.

    Args:
        name (str): The name of the system.

    Returns:
        BaseSystem: An instance of the corresponding system.
    """

    systems = {
        "kbrd_system": KBRDSystem,
        "compass_system": COMPASSSystem,
        "kgsf_system": KGSFSystem,
        "llama_system": LLaMASystem,
        "bert_system": BERTSystem,
        # Add other systems as needed
    }

    system = systems.get(name.lower(), None)

    if not system:
        raise ValueError(f"No system found with the name: {name}")

    return system
