import json
import os
from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np
import torch
from dotenv import load_dotenv
from loguru import logger
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.llmcrs.data.datatype import TaskType, ItemFeatureType

load_dotenv()


class CRSDataset(ABC):
    """
    Base class for Conversational Recommendation dataset.

    Args:
        restore (bool): Flag indicating whether to restore the dataset from a previous state.
        tokenizer_name (str): Name of the tokenizer to use.
        llm_tokenizer_name (str): Name of the LLM tokenizer to use.
        context_max_len (int): Maximum length of the conversation context.
        response_max_len (int): Maximum length of the response.
    """
    AVAILABLE_TOKENIZERS = {
        "bert-base-uncased",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-8B-Instruct",
    }
    ITEM_FEATURE_PATHS = {
        ItemFeatureType.TITLE: "kg/id2title.json",
        ItemFeatureType.DESCRIPTION: "kg/id2description.json",
        ItemFeatureType.INFO: "kg/id2info.json",
        ItemFeatureType.SUMMARY: "kg/id2item_summary.json",
    }
    ENTITY2ID_PATH = "kg/entity2id.json"
    NODE2TYPE_PATH = "kg/node_idx2type.json"
    ID2RELATION_PATH = "kg/id2relation.json"
    ENTITY2TEXT_PATH = "kg/entity_id2title.json"
    NODE_CAPTION_PATH = "kg/node_caption_template.json"
    INSTRUCTION_TEMPLATE_PATH = "kg/instruct_template.json"
    ITEM_INFO_PATH = "kg/id2info.json"
    MOVIE_KG_PATH = "kg/edge_set.csv"
    WORD_KG_PATH = "kg/conceptnet_subkg.txt"
    CONCEPT2ID_PATH = "kg/concept2id.json"
    REASON_INTEREST_TEMPLATE = "preprocessed/{dataset}_reason_interest.json"
    CHATGPT_REASON_INTEREST_HISTORY_TEMPLATE = "preprocessed/{dataset}_reason_interest_history.json"
    LLAMA_REASON_INTEREST_TEMPLATE = "preprocessed/llama3-8b/{dataset}_llama_reason_interest.json"
    GENERATED_INTEREST_PATH = "{dataset}_generated_outputs.json"
    SENTENCE_ENTITY_MAPPING_PATH = "preprocessed/sentence2entity.json"

    TOKEN = os.getenv("HUGGINGFACE_API_KEY")

    def __init__(self,
                 restore: bool = False,
                 tokenizer_name: str = "bert-base-uncased",
                 llm_tokenizer_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 context_max_len: int = 256,
                 response_max_len: int = 128,
                 use_chatgpt: bool = True,
                 addition_enhancer: str = None,
                 llm_generated_path: str = "preprocessed/llm_outputs/pretrain_and_include_rec",
                 ):
        if llm_tokenizer_name not in self.AVAILABLE_TOKENIZERS:
            raise ValueError(f"LLM Tokenizer {llm_tokenizer_name} is not supported yet.")
        self.restore = restore

        # use this for Qformer
        self.tokenizer_name = tokenizer_name
        self.llm_tokenizer_name = llm_tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, token=self.TOKEN)
        self.tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|start|>' + message['role'] + '\n' + message['content'] + '<|end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|start|>assistant\n' }}{% endif %}"
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_name, use_fast=True, token=self.TOKEN)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.llm_direct_rec_prompt = "As a movie recommender, please recommend a movie that aligns with the user's interests discussed in the conversation history provided.\n Conversation history:\n"
        self.llm_rec_prompt = (
            "As a movie recommender, analyze the user's conversation history to make a movie recommendation."
            "Break down the analysis into clear steps and return in json format:\n"
            "1. 'reasoning': analysis the user's conversation history and knowledge graph data highlight their "
            "movie preferences.\n"
            "2. 'overall preferences': A list of keywords summarizing the user's overall preferences.\n"
            "3. 'current interests': A list of keywords reflecting user's current interests.\n"
            "4. 'recommendation': a recommended movie title.\n"
            "History:\n"
        )
        self.bridge_rec_prompt = (
            "Extract the user’s overall movie preferences and current interests")
        self.has_prefix = False

        self.context_max_len = context_max_len
        self.response_max_len = response_max_len
        self.caption_max_len = 1024
        self.llm_generated_path = llm_generated_path
        self.use_chatgpt = use_chatgpt
        assert addition_enhancer in ["chatgpt", "llama", None], "addition_enhancer should be either chatgpt or llama or None"
        self.addition_enhancer = addition_enhancer

    @staticmethod
    def _load_raw_data(file_path: str) -> List[dict]:
        """
        Load the raw CRS dataset from the specified file.
        Args:
             file_path (str): The path to the raw CRS dataset file.

        Returns:
            list: the raw CRS dataset
        """
        logger.debug(f"Loading raw Redial data from {file_path}...")
        data = []
        with open(file_path, "r") as f:
            data.extend(json.loads(line) for line in f.readlines())
        logger.info(f"Successfully loaded {len(data)} entries from {file_path}.")
        return data

    @abstractmethod
    def _preprocess(self):
        """Preprocess the raw CRS dataset"""
        pass

    @abstractmethod
    def tokenize_rec(self, dialogue: List[dict], task_type: TaskType):
        NotImplementedError("This method should be implemented in the subclass.")

    def _load_item2feature(self, item_type: ItemFeatureType = ItemFeatureType.TITLE) -> Dict[str, str]:
        """
        Load item to feature mapping based on the given item type.

        Args:
            item_type (ItemFeatureType): The type of target item feature for which to load the mapping.
                Defaults to ItemFeatureType.TITLE.

        Returns:
            dict: A dictionary mapping item IDs to their corresponding features.
        """

        file_path = os.path.join(self.crs_path, self.ITEM_FEATURE_PATHS[item_type])
        with open(file_path, "r") as f:
            return json.load(f)


class TorchDataset(Dataset):
    def __init__(self, dataset, llm_token_path: str, stage: str):
        self.dataset = dataset
        self.use_generated_token = llm_token_path is not None
        if llm_token_path is not None:
            self.llm_tensor_path = os.path.join(llm_token_path, f"{stage}_last_hidden_states.pth")
            self.llm_tensor = torch.load(self.llm_tensor_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        utt = sample["utterance_id"]
        if self.use_generated_token:
            sample["compressed_token"] = self.llm_tensor[utt]
        return sample


class KGAlignmentDataset(Dataset):
    def __init__(self, dataset, id2entity, node2type, caption_template, enhanced_items, item_prompt, non_item_prompt):
        self.dataset = dataset
        # make key as int
        self.node2type = {int(k): v for k, v in node2type.items()}
        self.id2entity = id2entity
        self.caption_template = caption_template
        self.enhanced_items = {int(k): v for k, v in enhanced_items.items()}
        self.item_prompt = item_prompt
        self.non_item_prompt = non_item_prompt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        node_idx = self.dataset[idx]
        node_type = self.node2type[node_idx]
        if node_type == "movie":
            return {"node_idx": node_idx, "node_prompt": self.get_item_prompt(),
                    "node_text": self.enhanced_items[node_idx]}
        else:
            node_text = self.id2entity[node_idx]
            return {"node_idx": node_idx, "node_prompt": self.get_non_item_prompt(),
                    "node_text": self.generate_caption(node_type, node_text)}

    def generate_caption(self, node_type, node_value):
        if node_type not in self.caption_template:
            raise ValueError(f"node type {node_type} not found in the caption template")

        templates = self.caption_template[node_type]
        template = np.random.choice(templates)
        caption = template.format(**{node_type: node_value})
        return caption

    def get_item_prompt(self):
        return np.random.choice(self.item_prompt)

    def get_non_item_prompt(self):
        return np.random.choice(self.non_item_prompt)