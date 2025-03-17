import os
from typing import Union, Optional

import torch
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from loguru import logger
from torch.utils.data import DataLoader

from src.llmcrs.data.dataset import RedialDataset, TorchDataset, KGAlignmentDataset
from src.llmcrs.data.datatype import TaskType
from ..dataset.utils import pad_sequences, shuffle_and_split, get_onehot


class ReDialDataModule(LightningDataModule):

    def __init__(
            self,
            dataset: RedialDataset,
            phase: TaskType = TaskType.REC,
            train_batch_size: int = 32,
            val_batch_size: int = 128,
            test_batch_size: int = 64,
            node_align_train_batch_size: int = 512,
            node_align_val_batch_size: int = 512,
            node_align_test_batch_size: int = 512,
            is_baseline: bool = False,
            include_rec: bool = True,
            num_workers: int = 4,
    ):
        """
        Initialize the ReDialDataModule.

        Args:
            dataset (RedialDataset): Dataset instance.
            phase (TaskType, optional): Task type. Defaults to TaskType.REC.
            train_batch_size (int, optional): Batch size for training. Defaults to 32.
            val_batch_size (int, optional): Batch size for validation. Defaults to 128.
            num_workers (int, optional): Number of workers for the dataloader. Defaults to 4.
        """
        super().__init__()

        logger.error(f"Phase: {phase}")
        self.phase = phase
        self.is_baseline = is_baseline
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.node_align_train_batch_size = node_align_train_batch_size
        self.node_align_val_batch_size = node_align_val_batch_size
        self.node_align_test_batch_size = node_align_test_batch_size
        self.node_train_ratio = 0.9
        self.include_rec = include_rec

        self.node_alignment = True
        self.contextual_alignment = True
        self.n_relation = len(dataset.kg["id2relation"])
        self.n_items = dataset.kg["n_classes"]
        self.n_entity = len(dataset.kg["entity2id"]) + 1

        self.node_train_data, self.node_val_data, self.node_test_data = None, None, None
        self.train_dataset, self.valid_dataset, self.test_dataset = None, None, None
        self.tokenizer = dataset.tokenizer
        self.llm_tokenizer = dataset.llm_tokenizer
        self.prompt_template = dataset.kg["instruction_template"]
        self.llm_rec_prompts = dataset.llm_rec_prompt
        self.llm_direct_rec_prompt = dataset.llm_direct_rec_prompt
        self.bridge_rec_prompt = dataset.bridge_rec_prompt
        logger.info(f"llm rec prompts: {self.llm_rec_prompts}")
        self.dataset = dataset
        self.total_steps = self._get_total_steps()
        # # Temporarily turn off tokenizers parallelism
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def setup(self, stage: Optional[str] = None):
        if self.phase == TaskType.PRE:
            if not self.node_train_data and not self.train_dataset:
                if self.node_alignment:
                    train_entity, val_entity = shuffle_and_split(
                        list(self.dataset.kg["id2entity"].keys()),
                        ratio=self.node_train_ratio,
                        include_val=False
                    )
                    self.prepare_alignment_data(train_entity, val_entity, self.dataset)

        elif self.phase == TaskType.REC:
            self.prepare_contextual_data(self.dataset, TaskType.REC)

        else:
            raise NotImplementedError(f"Phase {self.phase} is not supported.")

    def prepare_alignment_data(self, train_entity, val_entity, dataset):
        self.node_train_data = self.create_alignment_dataset(train_entity, dataset)
        self.node_val_data = self.create_alignment_dataset(val_entity, dataset)
        # Reuse validation data for testing due to our focus on CRS performance.
        # This approach attempts to cover as many KG entities as possible, though it doesn't encompass all items.
        self.node_test_data = self.create_alignment_dataset(val_entity, dataset)
        logger.info(f"node alignment train data len {len(self.node_train_data)}")

    def prepare_contextual_data(self, dataset, task: TaskType = TaskType.REC):
        self.train_dataset = TorchDataset(dataset.tokenize_rec(dataset.train, task),
                                          dataset.llm_generated_token_path if not dataset.use_chatgpt else None,
                                          "train")
        self.valid_dataset = TorchDataset(dataset.tokenize_rec(dataset.valid, task),
                                          dataset.llm_generated_token_path if not dataset.use_chatgpt else None,
                                          "valid")
        self.test_dataset = TorchDataset(dataset.tokenize_rec(dataset.test, task),
                                         dataset.llm_generated_token_path if not dataset.use_chatgpt else None,
                                         "test")
        logger.debug(f"contextual data len:  {len(self.train_dataset)}")

    def create_alignment_dataset(self, entities, dataset):
        return KGAlignmentDataset(
            entities,
            dataset.kg["id2entity"],
            dataset.kg["node2type"],
            dataset.kg["node_caption"],
            dataset.kg["id2info"],
            self.prompt_template["movie"],  # item prompt
            self.prompt_template["node"],  # non item prompt
        )

    def _create_dataloader(self, dataset, batch_size: int, shuffle: bool = False, phase: str = "train",
                           collate_fn=None) -> DataLoader:
        """Helper function to create a DataLoader."""
        num_workers = self.num_workers if phase == "train" else self.num_workers // 2
        pin_memory = phase != "test"

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            collate_fn=collate_fn
        )

    def _collate_fn_helper(self, task="rec"):
        assert task in ["rec", "node", "context"]
        if task == "rec":
            return self._llm_rec_collate_fn if not self.is_baseline else self.baseline_rec_collate_fn
        elif task == "node":
            return self._node_align_collate_fn

    def train_dataloader(self):
        if self.phase == TaskType.REC:
            return self._create_dataloader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                shuffle=True,
                collate_fn=self._collate_fn_helper(task="rec")
            )
        elif self.phase == TaskType.PRE:
            node_alignment_dataloader = self._create_dataloader(
                self.node_train_data,
                self.node_align_train_batch_size,
                shuffle=True,
                collate_fn=self._collate_fn_helper(task="node")
            )
            return node_alignment_dataloader

    def val_dataloader(self) -> Union[DataLoader, CombinedLoader]:
        """Create DataLoader for validation."""
        if self.phase == TaskType.REC:
            return self._create_dataloader(
                self.valid_dataset,
                batch_size=self.val_batch_size,
                phase="val",
                collate_fn=self._collate_fn_helper(task="rec")
            )
        elif self.phase == TaskType.PRE:
            node_alignment_dataloader = self._create_dataloader(
                self.node_val_data,
                self.node_align_val_batch_size,
                phase="val",
                collate_fn=self._collate_fn_helper(task="node")
            )
            return node_alignment_dataloader

    def test_dataloader(self) -> Union[DataLoader, CombinedLoader]:
        """Create DataLoader for testing."""
        if self.phase == TaskType.REC:
            return self._create_dataloader(
                self.test_dataset,
                batch_size=self.test_batch_size,
                phase="test",
                collate_fn=self._collate_fn_helper(task="rec")
            )
        elif self.phase == TaskType.PRE:
            node_alignment_dataloader = self._create_dataloader(
                self.node_test_data,
                self.node_align_test_batch_size,
                phase="test",
                collate_fn=self._collate_fn_helper(task="node")
            )
            return node_alignment_dataloader

    def _get_total_steps(self) -> int:
        """
        Get the total number of training steps for the current dataset and batch size for one epoch.

        Returns:
            int: Total number of training steps for one epoch.
        """
        return len(self.dataset.train) // self.train_batch_size if self.phase == TaskType.REC else len(
            self.dataset.kg["id2entity"].keys()) * self.node_train_ratio // self.node_align_train_batch_size

    def _llm_rec_collate_fn(self, batch):
        items, bridge_inputs_prompt, context_entities = [], [], []
        llm_labels, items_name_list = [], []
        llm_inputs_prompt, llm_inputs_prompt_rec = [], []
        for rec_dict in batch:
            items.append(rec_dict["items"])
            items_name_list.append(rec_dict["item_name"])
            context_entities.append(rec_dict["context_entities"])
            llm_inputs = [
                {"role": "system", "content": self.llm_rec_prompts + rec_dict["context_text"]}]
            if self.include_rec:
                llm_labels.append(rec_dict["llm_text_rec"])
            else:
                llm_labels.append(rec_dict["llm_text"])
            llm_inputs_prompt.append(
                self.llm_tokenizer.apply_chat_template(
                    llm_inputs, tokenize=False, add_generation_prompt=True)
            )
        batch_dict = {
            "utterance_id": [rec_dict["utterance_id"] for rec_dict in batch],
            "items": torch.tensor(items, dtype=torch.long),
            "item_name": items_name_list,
            "context_entities": torch.tensor(pad_sequences(context_entities), dtype=torch.long),
            "llm_inputs": llm_inputs_prompt,
            "llm_labels": llm_labels,
        }

        return batch_dict

    def baseline_rec_collate_fn(self, batch):
        items, context_entities, context_words = [], [], []
        llm_outputs = []
        compressed_tokens = []
        dialogue_history = []
        # gold llm text is from chatgpt which doesn't have rec item.
        # the llm text from trained model does have rec item if include_rec is True during pretraining
        for rec_dict in batch:
            items.append(rec_dict["items"])
            context_entities.append(rec_dict["context_entities"])
            context_words.append(rec_dict["context_words"])
            llm_outputs.append(rec_dict["llm_text"])
            if not self.dataset.addition_enhancer:
                compressed_tokens.append(rec_dict["compressed_token"])
            dialogue_history.append(rec_dict["context_text"])

        batch_dict = {
            "items": torch.tensor(items, dtype=torch.long),
            "context_entities": torch.tensor(pad_sequences(context_entities), dtype=torch.long),
            "context_words": torch.tensor(pad_sequences(context_words), dtype=torch.long),
            "context_tokens": self.tokenizer(llm_outputs, padding=True, truncation=True, return_tensors="pt"), # this is generated llm text
            "entity_labels": get_onehot(context_entities, self.n_entity),
            "history_tokens": self.tokenizer(dialogue_history, padding=True, return_tensors="pt", truncation=True),
        }

        if compressed_tokens:
            batch_dict["compressed_tokens"] = torch.stack(compressed_tokens, dim=0)
        else:
            # create empty tensor for the compressed tokens for now TODO: fix this
            batch_dict["compressed_tokens"] = torch.tensor([])

        return batch_dict

    def _node_align_collate_fn(self, batch):
        # self.tokenizer.padding_side = "left" if is_test else "right"

        node_idx = torch.tensor([node_dict["node_idx"] for node_dict in batch], dtype=torch.long)
        prompt = [
            self.llm_tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": node_dict["node_prompt"]},
                ],
                tokenize=False, add_generation_prompt=True) for node_dict in batch
        ]
        # label
        node_caption = [node_dict["node_text"] for node_dict in batch]

        batch_dict = {
            "context_entities": node_idx.unsqueeze(1),
            "llm_labels": node_caption,  # gnn caption generation
            "llm_inputs": prompt
        }

        return batch_dict
