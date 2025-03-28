import csv
import itertools
import json
import logging
import os
import re
from copy import deepcopy
from typing import List, Dict, Tuple

import rootutils
from loguru import logger
from nltk.tokenize import word_tokenize

from src.llmcrs.data.dataset import CRSDataset
from src.llmcrs.data.datatype import ItemFeatureType, TaskType
from ..utils import save_data, restore_data, get_map_entities, unique_with_last_position, \
    add_reasoned_interests


class RedialDataset(CRSDataset):
    """
    Redial dataset from the paper: Towards deep conversational recommendations
    """
    # pattern to replace user mentions
    PATTERN = re.compile("@\\d+")

    def __init__(self,
                 restore: bool = False,
                 tokenizer_name: str = "bert-base-uncased",
                 llm_tokenizer_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 context_max_len: int = 256,
                 response_max_len: int = 128,
                 use_chatgpt: bool = True,
                 addition_enhancer: str = None,
                 llm_generated_path: str = "preprocessed/llm_outputs/compass",
                 ):

        super().__init__(restore=restore,
                         tokenizer_name=tokenizer_name,
                         llm_tokenizer_name=llm_tokenizer_name,
                         context_max_len=context_max_len,
                         response_max_len=response_max_len,
                         use_chatgpt=use_chatgpt,
                         llm_generated_path=llm_generated_path,
                         addition_enhancer=addition_enhancer,
                         )

        # print all stuffs from the config
        root_path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
        self.item_mask = "<Movie>"
        self.crs_path = os.path.join(root_path, "data/Redial")
        print(self.crs_path)
        self.max_entity_len = 0
        self.llm_generated_token_path = os.path.join(self.crs_path, self.llm_generated_path)

        if not self.restore:
            self.kg = self.load_external_data()
            self.train, self.valid, self.test = self._preprocess()
            data = (self.train, self.valid, self.test, self.kg)
            save_data(path=self.crs_path, data=data)
        else:
            self.train, self.valid, self.test, self.kg = restore_data(path=self.crs_path)

        logger.info(f"Successfully loaded the Redial dataset from {self.crs_path}.")
        logger.warning(f"Max entity length: {self.max_entity_len}.")


    def _load_crs(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Load the raw CRS dataset"""
        logger.info("Loading raw CRS dataset...")
        train_data = self._load_raw_data(os.path.join(self.crs_path, "raw", "train_data.jsonl"))
        valid_data = self._load_raw_data(os.path.join(self.crs_path, "raw", "valid_data.jsonl"))
        test_data = self._load_raw_data(os.path.join(self.crs_path, "raw", "test_data.jsonl"))
        return train_data, valid_data, test_data

    def _preprocess(self):
        """Preprocess the raw CRS dataset"""
        train, valid, test = self._load_crs()
        preprocessed_train_data = self._build_dialog(train)
        preprocessed_valid_data = self._build_dialog(valid)
        preprocessed_test_data = self._build_dialog(test)
        logger.info("successfully preprocess the Redial dataset")

        # save the preprocessed data for preview
        self._save_preprocessed_data(preprocessed_train_data, "train_data.json")
        self._save_preprocessed_data(preprocessed_valid_data, "valid_data.json")
        self._save_preprocessed_data(preprocessed_test_data, "test_data.json")

        # augment the data for recommendation and generation
        preprocessed_train_data = self._create_rec_dataset(preprocessed_train_data)
        preprocessed_valid_data = self._create_rec_dataset(preprocessed_valid_data)
        preprocessed_test_data = self._create_rec_dataset(preprocessed_test_data)

        # add reasoned interest to each utterance
        preprocessed_train_data = add_reasoned_interests(preprocessed_train_data,
                                                         self.kg["utterance_interest"]["train"],
                                                         use_gpt=self.use_chatgpt,
                                                         )
        preprocessed_valid_data = add_reasoned_interests(preprocessed_valid_data,
                                                         self.kg["utterance_interest"]["valid"],
                                                         use_gpt=self.use_chatgpt,
                                                         )
        preprocessed_test_data = add_reasoned_interests(preprocessed_test_data,
                                                        self.kg["utterance_interest"]["test"],
                                                        use_gpt=self.use_chatgpt,
                                                        )
        # save the preprocessed data for preview
        self._save_preprocessed_data(preprocessed_train_data, "train_rec_data.json")
        self._save_preprocessed_data(preprocessed_valid_data, "valid_rec_data.json")
        self._save_preprocessed_data(preprocessed_test_data, "test_rec_data.json")

        return preprocessed_train_data, preprocessed_valid_data, preprocessed_test_data

    def _build_dialog(self, data):
        """Build dialogs from the raw Redial data"""
        conversations = [self._merge_dialog(dialog) for dialog in data]
        all_utterances = []
        utterance_id = 0
        for conv in conversations:
            dialog_list, utterance_id = self._augment_and_add(conv, utterance_id)
            all_utterances.extend(dialog_list)
        return all_utterances

    def _merge_dialog(self, dialog):
        """
        Merge the utterances into one utterance if the message is sent by same person.
        i.e.[seeker utterance 1, seeker utterance 2, recommender utterance 1, seeker utterance 3] ->
            [seeker utterance 1+2, recommender utterance 1, seeker utterance 3]

        Args:
            dialog: a dialog from the raw Redial data

        Returns:
            augmented_dialog: a dialog with merged utterances
        """
        augmented_dialog = []
        message = dialog["messages"]
        movie2name = dialog["movieMentions"]
        last_role = None
        utter2entity = self.kg["sentence2entity"]
        for utt in message:
            message = utt["text"]
            entity = get_map_entities(utt["text"], utter2entity)
            word_tokens = word_tokenize(message)
            role = self._check_role(dialog["initiatorWorkerId"], utt["senderWorkerId"])
            if role == last_role:
                augmented_dialog[-1]["message"] += f" {message}"
                augmented_dialog[-1]["entity"] += entity
                augmented_dialog[-1]["word"] += word_tokens
            else:
                augmented_dialog.append(
                    {
                        "role": role,
                        "message": f"{role}: {message}",
                        "movies2name": movie2name,
                        "entity": entity,
                        "word": word_tokens,
                    }
                )
            last_role = role
        return augmented_dialog

    def _augment_and_add(self, conversation: dict, utterance_id: int):
        """Augment the conversation and build the context and response for each utterance

        Args:
            conversation: a conversation from the raw Redial data
            utterance_id: the id of the utterance

        Returns:
            augmented_dialog_dicts: a list of augmented dialog dicts
            utterance_id: the id of the utterance
        """
        augmented_dialog_dicts, context_entities, context_items, context_words = [], [], [], []
        unique_entities_list, unique_item_list, unique_words_list = [], [], []
        context = ""
        for i, conv in enumerate(conversation):

            message = conv["message"]
            # for kgsf
            word_ids = [self.kg["word2id"][word] for word in conv["word"] if word in self.kg["word2id"]]
            mask_text, movie_ids = self._convert_ids_to_indices(message, mask=True, movie2name=conv["movies2name"])
            raw_message, _ = self._convert_ids_to_indices(message, mask=False, movie2name=conv["movies2name"])
            try:
                movies = [self.kg["entity2id"][movie] for movie in movie_ids if movie in self.kg["entity2id"]]
                entity_ids = [self.kg["entity2id"][entity] for entity in conv["entity"]]
            except KeyError as e:
                logger.error(f"movie {movie_ids} not found in the KG")
                logger.error(f"message: {message}")
                raise KeyError from e

            if i > 0:
                conv_dict = {
                    "role": deepcopy(conv["role"]),
                    "context": deepcopy(context),
                    "context_entities": deepcopy(unique_entities_list),
                    "context_words": deepcopy(unique_words_list),
                    "raw_response": deepcopy(raw_message),
                    "response": deepcopy(mask_text),
                    "utterance_id": deepcopy(utterance_id),
                    "items": deepcopy(movies),
                }
                augmented_dialog_dicts.append(conv_dict)
                utterance_id += 1

            context += "\n" + raw_message if i > 0 else raw_message
            context_entities.extend(movies)  # add movies
            context_entities.extend(entity_ids)
            context_words.extend(word_ids)
            unique_entities_list = unique_with_last_position(context_entities)
            current_num = len(context_entities)
            unique_words_list = unique_with_last_position(context_words)
            if current_num > self.max_entity_len:
                self.max_entity_len = current_num

        return augmented_dialog_dicts, utterance_id

    def _augment_rec_data(self, utterance):
        """
        Augment the dataset for the recommendation task.

        Parameters:
            utterance (dict): An utterance from the dataset.

        Returns:
            list: A list of augmented utterances.

        Notes:
            - The last utterance should be from the seeker and the response should be from the recommender.
            - The item cannot be empty.
            - If two items are recommended, two utterances will be created, one for each item.
        """

        augment_list = []

        for item in utterance["items"]:
            augment_rec_list = {
                "utterance_id": utterance["utterance_id"],
                "context_entities": utterance["context_entities"],
                "context_words": utterance["context_words"],
                "context": utterance["context"],
                "raw_response": utterance["raw_response"],
                "response": utterance["response"],
                "role": utterance["role"],
                "item": item,
                "item_name": self.kg["id2title"][str(item)],
            }
            augment_list.append(augment_rec_list)

        return augment_list

    def _create_rec_dataset(self, dialog):
        """
        Creates a dataset by extracting and augmenting recommender data from a given dialog.

        Parameters:
            dialog (list): A list of utterances in the dialog.

        Returns:
            list: A list of augmented recommender data extracted from the dialog.
        """
        return list(
            itertools.chain(
                *[
                    self._augment_rec_data(utterance)
                    for utterance in dialog
                    if utterance["role"] == "Recommender" and utterance["items"]
                ]
            )
        )

    @staticmethod
    def _check_role(init_role, role):
        """Check if the role is correct"""
        return "User" if init_role == role else "Recommender"

    def _convert_ids_to_indices(self, text: str, mask: bool = True, movie2name: Dict[str, str] = None):
        """
        Convert @movieID in the given text to movieID or replace them with movie names depending on the mask parameter.

        Args:
            text (str): The text to convert.
            mask (bool, optional): Whether to mask the movie IDs or replace them with movie names. Defaults to True.
            movie2name (Dict[str, str]: A dictionary mapping movie IDs to their names. Defaults to None.

        Returns:
            - text: the converted text
            - movie_id_list: a list of movieIDs
        """
        movie_ids = self.PATTERN.findall(text)
        if not movie_ids:
            return text, []

        movie_id_list = [movie_id[1:] for movie_id in movie_ids]
        for movie_id in movie_id_list:
            if mask:
                replace_with = self.item_mask
            else:
                replace_with = movie2name.get(movie_id, self.item_mask)
            text = text.replace(f"@{movie_id}", replace_with)

        return text, movie_id_list

    def _save_preprocessed_data(self, data: List[dict], file_name: str):
        """
        Save preprocessed data to a file.

        Args:
            data (List[dict]): The preprocessed data to be saved.
            file_name (str): The name of the file to save the data to.

        Returns:
            None
        """
        logger.debug(f"Saving preprocessed data to {file_name}...")
        target_path = os.path.join(self.crs_path, "preprocessed")
        os.makedirs(target_path, exist_ok=True)

        file_path = os.path.join(target_path, file_name)
        with open(file_path, "w") as f:
            json.dump(data, f)
        logger.info(f"Successfully saved the preprocessed data to {file_path}.")

    def load_external_data(self):
        """
        Load an external data file containing entity2id, item2feature, and item_attribute.

        Returns:
            dict: A dictionary containing the loaded external data, with the following keys:
                - entity2id (dict): A dictionary mapping entity names to unique IDs.
                - item2feature (dict): A dictionary mapping item names to their associated features.
                - item_attribute (torch.Tensor): A tensor representing the tokenized item description,
                    with padded and truncated with padding and truncation applied.

        """
        data = {"entity2id": self._load_entity2id(),
                "id2relation": self._load_id2relation(),
                "id2title": self._load_item2feature(ItemFeatureType.TITLE),
                "edge": self._load_structured_kg(),
                "sentence2entity": self._load_sentence_entity_mapping(),
                "id2info": self._load_item2feature(ItemFeatureType.INFO),
                "node_caption": self._load_template(self.NODE_CAPTION_PATH),
                "instruction_template": self._load_template(self.INSTRUCTION_TEMPLATE_PATH),
                }

        word2id = self._load_word2id()
        word_kg = self._load_word_kg(word2id)
        node2type, n_classes = self._load_node2type()
        data["word2id"] = word2id
        data["word_kg"] = word_kg
        data["node2type"] = node2type
        data["n_classes"] = n_classes
        logging.info(f"Number of classes: {n_classes}")
        if self.use_chatgpt:
            utterance_interest = self._get_reasoned_interest()
            data["utterance_interest"] = utterance_interest
        else:
            data["utterance_interest"] = self._get_generated_interest()
        # convert entity2id to id2entity
        data["id2entity"] = {v: k for k, v in data["entity2id"].items()}

        return data

    def _load_entity2text(self) -> Dict[str, str]:
        """
        Load the entity2text dictionary from the preprocessed directory.

        Returns:
            Dict[str, str]: The entity2text dictionary loaded from the JSON file.
        """
        file_path = os.path.join(self.crs_path, self.ENTITY2TEXT_PATH)
        with open(file_path, "r") as f:
            entity2text = json.load(f)

        logger.info("Successfully loaded the entity2text dictionary")
        logger.info(f"Number of entities: {len(entity2text)}")
        return entity2text

    def _load_entity2id(self) -> Dict[str, int]:
        """
        Load the entity2id dictionary from the preprocessed directory.

        Returns:
            Dict[str, int]: The entity2id dictionary loaded from the JSON file.
        """
        file_path = os.path.join(self.crs_path, self.ENTITY2ID_PATH)
        with open(file_path, "r") as f:
            entity2id = json.load(f)

        logger.info("Successfully loaded the entity2id dictionary")
        logger.info(f"Number of entities: {len(entity2id)}")
        return entity2id

    def _load_node2type(self) -> tuple[Dict[str, str], int]:
        """
        Load the node2type dictionary from the preprocessed directory.

        Returns:
            Dict[str, str]: The node2type dictionary loaded from the JSON file.
            int: The number of classes in the dataset.
        """
        file_path = os.path.join(self.crs_path, self.NODE2TYPE_PATH)
        with open(file_path, "r") as f:
            node2type = json.load(f)
        # node idx start from 1, therefore +1
        n_classes = list(node2type.values()).count("movie") + 1

        return node2type, n_classes

    def _load_structured_kg(self) -> List[Tuple[int, int, int]]:
        """
        Load the structured knowledge graph from a CSV file.

        Returns:
            edge_loaded (List[Tuple[int, int, int]]): A list of tuples. Each tuple contains three ints. The
                source, destination, and relation of an edge.
        """

        edge_loaded = set()  # Set to store the loaded edges
        file_path = os.path.join(self.crs_path, self.MOVIE_KG_PATH)  # Get the file path
        with open(file_path, "r") as f:  # Open the file for reading
            reader = csv.reader(f)  # Create a CSV reader
            next(reader)  # Skip the header row
            for row in reader:  # Iterate over each row in the file
                source, destination, relation = map(int, row)  # Convert strings to integers
                edge_loaded.add((source, destination, relation))  # Add the edge to the set

        logger.info("Successfully loaded the structured knowledge graph")
        logger.info(f"Number of edges: {len(edge_loaded)}")
        return list(edge_loaded)  # Return the list of loaded edges

    def _load_id2relation(self) -> Dict[str, str]:
        """
        Load the id2relation dictionary from the preprocessed directory.
        Returns:
            Dict[str, str]: The id2relation dictionary loaded from the JSON file.
        """
        file_path = os.path.join(self.crs_path, self.ID2RELATION_PATH)
        with open(file_path, "r") as f:
            id2relation = json.load(f)

        logger.info("Successfully loaded the id2relation dictionary")
        logger.info(f"Number of relations: {len(id2relation)}")
        return id2relation

    def _load_sentence_entity_mapping(self) -> Dict[str, List[str]]:
        """
        Load the sentence to entity mapping from a JSON file.

        Returns:
            Dict[str, List[str]]: A dictionary mapping sentences to their corresponding entity names.
        """
        file_path = os.path.join(self.crs_path, self.SENTENCE_ENTITY_MAPPING_PATH)
        with open(file_path, "r") as f:
            return json.load(f)

    def _load_template(self, path: str) -> Dict[str, str]:
        """
        Load the node caption template from a JSON file.

        Returns:
            Dict[str, str]: A dictionary mapping node types to their corresponding caption templates.
        """
        file_path = os.path.join(self.crs_path, path)
        with open(file_path, "r") as f:
            return json.load(f)

    def _load_word2id(self) -> Dict[str, int]:
        """
        Load the word2id dictionary from the preprocessed directory.

        Returns:
            Dict[str, int]: The word2id dictionary loaded from the JSON file.
        """
        file_path = os.path.join(self.crs_path, self.CONCEPT2ID_PATH)
        with open(file_path, "r", encoding="utf-8") as f:
            word2id = json.load(f)

        logger.info("Successfully loaded the word2id dictionary")
        logger.info(f"Number of words: {len(word2id)}")

        return word2id

    def _load_word_kg(self, word2id):
        """
        Load the word knowledge graph from a file.

        Returns:
            List[Tuple[int, int, int]]: A list of tuples. Each tuple contains three ints. The
                source, destination, and relation of an edge.
        """
        file_path = os.path.join(self.crs_path, self.WORD_KG_PATH)
        with open(file_path, "r") as f:
            word_kg = f.readlines()

        edges = set()
        entities = set()
        for line in word_kg:
            kg = line.strip().split('\t')
            entity1 = kg[1].split('/')[0]
            entity2 = kg[2].split('/')[0]
            entities.add(entity1)
            entities.add(entity2)
            e0 = word2id[entity1]
            e1 = word2id[entity2]
            edges.add((e0, e1))
            edges.add((e1, e0))
        return {
            "word_edges": list(edges),
        }

    def _get_reasoned_interest(self) -> Dict[str, Dict]:

        datasets = ["train", "valid", "test"]
        if self.addition_enhancer is None:
            path = self.REASON_INTEREST_TEMPLATE
        else:
            path_templates = {
                "chatgpt": self.CHATGPT_REASON_INTEREST_HISTORY_TEMPLATE,
                "llama": self.LLAMA_REASON_INTEREST_TEMPLATE
            }
            path  = path_templates[self.addition_enhancer]
        logger.info(f"Loading reasoned interest from {path}")
        reason_interest = {}
        for dataset in datasets:
            file_path = os.path.join(self.crs_path, path.format(dataset=dataset))
            with open(file_path, "r") as f:
                reason_interest[dataset] = json.load(f)

        return reason_interest

    def _get_generated_interest(self) -> Dict[str, Dict]:
        """
        Load the generated interest from the preprocessed directory.

        Returns:
            Dict[str, Dict]: The generated interest loaded from the JSON file.
        """
        datasets = ["train", "valid", "test"]
        generated_interest = {}
        for datasets in datasets:
            file_path = os.path.join(self.crs_path, self.llm_generated_path, self.GENERATED_INTEREST_PATH.format(dataset=datasets))
            with open(file_path, "r") as f:
                generated_text = json.load(f)
                # only keep generated_text key
                generated_interest[datasets] = {k: v["generated_text"] for k, v in generated_text.items()}

        return generated_interest

    def tokenize_rec(self, dialogue: List[dict], task_type: TaskType.REC) -> List[dict]:
        """
        Tokenizes a dialogue and returns a new tokenized dialogue.

        Args:
            dialogue (List[dict]): A dictionary representing a dialogue, containing context and preference information.
            task_type (TaskType): The task type for which to tokenize the dialogue.

        Returns:
            List[dict]: A list of dictionaries representing the tokenized dialogue, with token IDs for context and
                preference, and other information.

        """
        self.llm_tokenizer.truncation_side = "left"
        new_dialogue = []
        for utterance in dialogue:
            # TODO: Clean up this part
            if task_type == TaskType.REC:
                for text_type in ["context"]:
                    original_text = utterance[text_type]
                    # TODO: optimize this part and not hardcode the max length

                    llm_tokenized_original_text = self.llm_tokenizer(
                        original_text,
                        return_tensors="pt",
                        return_attention_mask=False,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=self.context_max_len,
                    )["input_ids"]

                    # Convert tokenized text back to string, might have issues due to subword tokenization
                    truncated_text = self.llm_tokenizer.decode(llm_tokenized_original_text[0], skip_special_tokens=True)
                    utterance["context_text"] = truncated_text

                conv_dict = {
                    "utterance_id": utterance["utterance_id"],
                    "context_text": utterance["context_text"],
                    "llm_text": utterance["gpt_outputs"],
                    "context_words": utterance["context_words"],
                    "context_entities": utterance["context_entities"],
                    "items": utterance["item"],
                    "item_name": utterance["item_name"],
                }
                if self.use_chatgpt:
                    conv_dict["llm_text_rec"] = utterance["gpt_outputs_rec"]  # this is only training LLM not downstream models
                new_dialogue.append(conv_dict)

        return new_dialogue
