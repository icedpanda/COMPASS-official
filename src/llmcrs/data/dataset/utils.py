import json
import os
import pickle as pkl
from copy import deepcopy
from random import shuffle
from typing import List, Dict, Tuple

import torch
from loguru import logger
from torch_geometric.utils import add_remaining_self_loops


def save_data(data, path: str, file_name: str = "all_data.pkl"):
    """
    Save the given data to a file.

    Args:
        data: The data to be saved.
        path (str): The path to the directory where the file should be saved.
        file_name (str, optional): The name of the file. The default is "all_data.pkl".

    Returns:
        None
    """
    save_path = os.path.join(path, file_name)
    with open(save_path, "wb") as f:
        pkl.dump(data, f)
    logger.info(f"[Save dataset to {file_name}]")


def restore_data(path: str, file_name: str = "all_data.pkl"):
    """
    Restores data from a saved dataset file.

    Args:
        path (str): The path to the directory containing the dataset file.
        file_name (str, optional): The name of the dataset file. Defaults to "all_data.pkl".

    Returns:
        object: The restored dataset.

    Raises:
        ValueError: If the specified dataset file does not exist.
    """
    file_path = os.path.join(path, file_name)
    if not os.path.exists(file_path):
        raise ValueError(f"Saved dataset [{file_path}] does not exist")
    with open(file_path, "rb") as f:
        dataset = pkl.load(f)
    logger.info(f"Restore dataset from [{file_path}]")
    return dataset


def add_preference_summary(dialogue: List[dict], summary: Dict[str, dict]) -> List[dict]:
    """
    Adds preference summary to each utterance in a dialogue.

    Args:
        dialogue (List[dict]): A list of dictionaries representing the dialogue.
        summary (Dict[str, dict]): A dictionary containing the preference summary for each utterance.

    Returns:
        List[dict]: A modified dialogue with preference added to each utterance.
    """

    for utterance in dialogue:
        utterance_id = utterance["utterance_id"]
        preferences = summary[str(utterance_id)]["gpt_preferences"]
        utterance["preference"] = preferences

    return dialogue


def add_reasoned_interests(dialogue: List[dict], interests: Dict[str, dict], rec_tokens: List = None,
                           use_gpt: bool = False) -> List[dict]:
    """
    Adds reasoned interests to each utterance in a dialogue.
    Args:
        dialogue (List[dict]): A list of dictionaries representing the dialogue.
        interests: (Dict[str, dict]): A dictionary containing reasoned interests in each utterance.
        rec_tokens (List[str, str]): Start and end tokens for recommendation.
        use_gpt (bool): Whether to use ChatGPT generated interest or small LLM generated.

    Returns:
        List[dict]: A modified dialogue with reasoned interests added to each utterance.

    """
    if use_gpt:
        for utterance in dialogue:
            utterance_id = utterance["utterance_id"]
            gpt_output = interests[str(utterance_id)]["gpt_output"]
            gpt_output_with_rec = deepcopy(gpt_output)
            try:
                gpt_output_with_rec["Recommendation"] = utterance["item_name"]
            except TypeError:
                ## llama 8b sometimes does not generate in json format
                ## TODO: skip for now since we are not training COMPASS on llama outputs
                pass
            utterance["gpt_outputs"] = json.dumps(gpt_output)
            utterance["gpt_outputs_rec"] = json.dumps(gpt_output_with_rec)
    else:
        # use our llm generated reasoned interests
        for utterance in dialogue:
            utterance_id = utterance["utterance_id"]
            gpt_output = interests[str(utterance_id)]
            utterance["gpt_outputs"] = gpt_output

    return dialogue

def get_map_entities(raw_txt: str, entity_map: Dict[str, List[str]]) -> List[str]:
    """
    Gets the entities from a given text.

    Args:
        raw_txt (str): The text to extract entities from.
        entity_map (Dict[str, List[str]]): A dictionary mapping raw utterances to their entities.

    Returns:
        List[str]: A list of entities.
    """
    return entity_map.get(raw_txt, [])


def pad_sequences(sequences: List[List[int]], pad_value: int = 0, pad_side: str = "right", max_len=None) -> List[
        List[int]]:
    """Pad sequences to have the same length.

    Args:
        sequences (list of lists): Input sequences to be padded.
        pad_value (int): Value to be used for padding.
        pad_side (str): Side for padding, either 'left' or 'right'.
        max_len (int, optional): Maximum length of the padded sequences. If None, the maximum length of the input sequences

    Returns:
        list of lists: Padded sequences.
    """

    max_len = max(len(seq) for seq in sequences) if max_len is None else max_len
    if pad_side == 'left':
        # Pad each sequence to the maximum sequence length on the left side
        return [[pad_value] * (max_len - len(seq)) + seq for seq in sequences]
    elif pad_side == 'right':
        # Pad each sequence to the maximum sequence length on the right side
        return [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]
    else:
        raise ValueError("pad_side should be either 'left' or 'right'")


def edge_to_pyg_format(edge, types: str = "RGCN"):
    """
    Converts an edge to the format used by PyTorch Geometric (PyG) library.

    Args:
        edge (list or tensor): The edge to be converted. It should be a list of tuples or a tensor with shape (N, 3),
            where N is the number of edges. Each tuple or row in the tensor should contain the source node index,
            target node index, and edge type.
        types (str): The type of edge. It can be either "RGCN" or "GCN". The default is "RGCN".

    Returns:
        tuple: A tuple containing two tensors. The first tensor is the sorted edge index tensor with shape (2, N),
            where N is the number of edges. The second tensor is the edge type tensor with shape (N,).
    """
    if types == "RGCN":
        edge_sets = torch.as_tensor(edge, dtype=torch.long)
        edge_idx = edge_sets[:, :2].t().contiguous()
        edge_type = edge_sets[:, 2]

        # edge_index = torch.stack([edge_idx[1], edge_idx[0]])
        return edge_idx, edge_type

    elif types == "GCN":
        edge_set = [[co[0] for co in edge], [co[1] for co in edge]]
        return torch.as_tensor(edge_set, dtype=torch.long)
    else:
        raise NotImplementedError('type {} has not been implemented', type)


def add_self_loops_edge_type(edge_types: torch.Tensor, num_new_nodes: int, self_loop_type_idx: int) -> torch.Tensor:
    """
    Adds self-loops to the edge types.

    Args:
        edge_types (tensor): The edge types tensor.
        num_new_nodes (int): The number of new nodes that are self-looped.
        self_loop_type_idx (int): The index of the self-loop type.

    Returns:
        tensor: The edge types tensor with self-loops added.
    """
    new_edge_type = torch.cat(
        [edge_types, torch.full([num_new_nodes], self_loop_type_idx, dtype=torch.long)]
    )
    return new_edge_type


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, int]:
    """
    Adds self-loops to the edge index. This is for full graph

    Args:
        edge_index (tensor): The edge index tensor.
        num_nodes (int): The number of nodes in the graph.

    Returns:
        tensor: The edge index tensor with self-loops added.
        int: The number of self-loops added.
    """
    new_edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
    num_of_added_edges = new_edge_index.size()[1] - edge_index.size()[1]
    return new_edge_index, num_of_added_edges


def unique_with_last_position(lst: List[int], last_position: bool = True) -> List[int]:
    """
    Returns a list of unique elements in the given list, preserving the last position of each element.
    Args:
        lst (List[int]): The list to be processed.
        last_position (bool, optional): Whether to preserve the last position of each element. Defaults to True.

    Returns:
        List[int]: A list of unique elements in the given list, preserving the last position of each element if
            the last_position is True, otherwise preserving the first position.

    """
    seen = set()
    result = []

    if last_position:
        for item in reversed(lst):
            if item not in seen:
                seen.add(item)
                result.append(item)

        result.reverse()
    else:
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)
    return result


def shuffle_and_split(data, ratio: float = 0.8, include_val=True):
    shuffle(data)
    length = len(data)
    train_split = int(ratio * length)
    if not include_val:
        return data[:train_split], data[train_split:]
    val_split = int((1 - ratio) / 2 * length) + train_split
    return data[:train_split], data[train_split:val_split], data[val_split:]


def shuffle_and_split_edges(edges: List[Tuple[int, int, int]], ratio: float = 0.8):
    return shuffle_and_split(edges, ratio)


def shuffle_and_split_alignments(entity_2text: Dict[str, str], ratio: float = 0.8):
    data = [[int(idx), text] for idx, text in entity_2text.items()]
    return shuffle_and_split(data, ratio)


def get_onehot(data_list, categories) -> torch.Tensor:
    """Transform lists of label into one-hot.

    Args:
        data_list (list of list of int): source data.
        categories (int): #label class.

    Returns:
        torch.Tensor: one-hot labels.

    """
    onehot_labels = []
    for label_list in data_list:
        onehot_label = torch.zeros(categories)
        for label in label_list:
            onehot_label[label] = 1.0 / len(label_list)
        onehot_labels.append(onehot_label)
    return torch.stack(onehot_labels, dim=0)
