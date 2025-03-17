from enum import Enum


class TaskType(Enum):
    """
    Enum class representing the types of tasks.

    Attributes:
        REC (str): The task is a recommendation task.
        CONV (str): The task is a conversation task.
    """
    PRE = "align"
    REC = "rec"
    CONV = "conv"


class ItemFeatureType(Enum):
    """
    Enum class representing the types of item features.

    Attributes:
        TITLE (str): Uses the item title as the item feature.
        DESCRIPTION (str): Uses the item description as the item feature.
        INFO (str): Uses the item info as the item feature. For example, using title, actors, director,
        genre, etc. as the item feature for a movie.
        SUMMARY (str): Uses the item summary as the item feature.
        DEFAULT (str): Does not use any textual item features.
    """

    TITLE = "title"
    DESCRIPTION = "description"
    INFO = "info"
    SUMMARY = "summary"
    DEFAULT = "default"
