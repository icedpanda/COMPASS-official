import os
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
from dotenv import load_dotenv
from loguru import logger
from peft import LoraConfig, get_peft_model
from torch_geometric.nn import GATConv
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers.utils import ModelOutput

from src.llmcrs.data.dataset.utils import edge_to_pyg_format
from src.llmcrs.models.utils.modules import create_mask
from src.llmcrs.models.utils import RGCN

load_dotenv()
ALLOWED_BERT_MODELS = {"meta-llama/Llama-2-7b-chat-hf",
                       "meta-llama/Llama-2-13b-chat-hf",
                       "meta-llama/Meta-Llama-3-8B",
                       "meta-llama/Meta-Llama-3-8B-Instruct",
                       "meta-llama/Llama-3.1-8B-Instruct",
                       }
TOKEN = os.getenv("HUGGINGFACE_API_KEY")


def create_attention_masks(entities, mode: str) -> torch.Tensor:
    """
    Create attention masks for the entities based on the mode.
    Args:
        entities (torch.Tensor): A tensor containing the entity indices.
        mode (str): The mode for creating the attention mask.
            - "node_text": Create an attention mask for the node-text alignment.
            - "contextual": Create an attention mask for the contextual alignment.
    Returns:
        torch.Tensor: A binary tensor containing the attention mask.
    """
    if mode == "node_text":
        # only one entity and there is no padding
        return torch.ones(entities.size(), dtype=torch.long).to(entities.device)
    return create_mask(entities)


def concat_text_input_output(input_ids, input_atts, output_ids, output_atts):
    input_part_targets_len = []
    llm_tokens = {"input_ids": [], "attention_mask": []}
    for i in range(input_ids.size(0)):
        this_input_ones = input_atts[i].sum()
        input_part_targets_len.append(this_input_ones)
        llm_tokens['input_ids'].append(
            torch.cat([
                input_ids[i][:this_input_ones],
                output_ids[i][1:],
                input_ids[i][this_input_ones:]
            ])
        )
        llm_tokens['attention_mask'].append(
            torch.cat([
                input_atts[i][:this_input_ones],
                output_atts[i][1:],
                input_atts[i][this_input_ones:]
            ])
        )
    llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
    llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
    return llm_tokens, input_part_targets_len


class GAT(nn.Module):
    def __init__(self,
                 in_dim: int = 256,
                 out_dim: int = 256,
                 heads=8,
                 negative_slope=0.2,
                 add_self_loops=False
                 ):
        super().__init__()

        self.conv1 = GATConv(in_dim, out_dim, heads=heads, negative_slope=negative_slope, add_self_loops=add_self_loops, concat=False).to(
            torch.bfloat16)
        # self.conv2 = RGCNConv(hid_dim, out_dim, num_relations, num_bases)
        self.out_dim = out_dim

    def forward(self, x, edge_index, edge_type=None):
        x = self.conv1(x, edge_index, edge_type)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training, p=self.dropout)
        # x = self.conv2(x, edge_index, edge_type)

        return x


class KnowledgeEmbedding(nn.Module):
    """
    Knowledge embedding module that takes in the entity and relation embeddings and returns the knowledge embeddings.
    """

    def __init__(
            self,
            edges: torch.Tensor,
            n_ent: int,
            n_rel: int,
            ent_dim: int,
            rel_dim: int,
            ent_path: str = None,
            rel_path: str = None,
    ):
        super().__init__()
        edge_idx, edge_type = edge_to_pyg_format(edges)
        self.register_buffer("edge_idx", edge_idx)
        self.register_buffer("edge_type", edge_type)
        self.ent_embeds = self._initialize_embeddings("entity", ent_path, n_ent, ent_dim, padding_idx=0)
        # self.rel_embeds = self._initialize_embeddings("relation", rel_path, n_rel, rel_dim, padding_idx=0)

    @staticmethod
    def _initialize_embeddings(name, path, n_embeds, dim, padding_idx=None):
        if path:
            logger.info(f"{name} embeddings path provided, loading from {path}")
            embeddings = nn.Embedding.from_pretrained(torch.load(path), freeze=False, padding_idx=padding_idx).to(
                torch.bfloat16)
        else:
            logger.info(f"{name} embeddings path not provided, initializing {name} embeddings")
            embeddings = nn.Embedding(n_embeds, dim, padding_idx=padding_idx).to(torch.bfloat16)

        return embeddings

    def get_ent_embeddings(self, node_idx=None):
        return self.ent_embeds(node_idx) if node_idx is not None else self.ent_embeds.weight

    def get_rel_embeddings(self, rel_idx=None):
        return self.rel_embeds(rel_idx) if rel_idx is not None else self.rel_embeds.weight

    def get_edge_idx(self):
        return self.edge_idx

    def get_edge_type(self):
        return self.edge_type


class COMPASS(nn.Module):
    """
    LLM enhanced recommender model that uses mlp to connect the language model and the knowledge graph.
    """

    def __init__(self,
                 lora_config: LoraConfig,
                 edges: torch.Tensor,
                 n_ent: int,
                 n_rel: int,
                 ent_dim: int,
                 rel_dim: int,
                 ent_path: str = None,
                 rel_path: str = None,
                 in_dim: int = 768,
                 hid_dim: int = 768,
                 out_dim: int = 768,
                 num_bases: int = 8,
                 num_relations: int = 11,
                 dropout: float = 0.1,
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 is_lora: bool = False,
                 ):
        super().__init__()

        if model_name not in ALLOWED_BERT_MODELS:
            raise ValueError(f"Model name {model_name} is not supported yet.")

        self.model_name = model_name
        self.lora_config = lora_config
        self.kg_embeds = KnowledgeEmbedding(
            edges, n_ent, n_rel, ent_dim, rel_dim, ent_path, rel_path, )
        self.graph_encoder = RGCN(
            num_relations=num_relations, in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim, num_bases=num_bases,
            dropout=dropout)
        # self.graph_encoder = GAT(
        #     in_dim, out_dim,
        # )
        self.is_lora = is_lora
        self.init_llm()
        self.max_llm_prompt_len = 1024
        self.max_output_txt_len = 1024

    def init_llm(self):
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_name,
            token=TOKEN,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, token=TOKEN)
        self.llm_tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.llm_tokenizer.pad_token = "<pad>"
        logger.info(f"LLM padding token id : {self.llm_tokenizer.pad_token_id}")

        self.terminators = [
            self.llm_tokenizer.eos_token_id,
            self.llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.model.resize_token_embeddings(len(self.llm_tokenizer))

        self.bridge_proj = nn.Linear(self.graph_encoder.out_dim, self.model.config.hidden_size).to(torch.bfloat16)

        if self.is_lora:
            logger.info(f"Lora config: %s {self.lora_config}")
            self.model = get_peft_model(self.model, self.lora_config)

        logger.info(f"Initialized encoders with {self.model_name}")

    def get_kg_feature(self):
        x = self.kg_embeds.get_ent_embeddings()
        edge_idx, edge_type = self.kg_embeds.get_edge_idx(), self.kg_embeds.get_edge_type()
        kg_feat = self.graph_encoder(x, edge_idx, edge_type)
        # edge_embed = self.kg_embeds.get_rel_embeddings(edge_type)
        # kg_feat = self.graph_encoder(x, edge_idx)
        return kg_feat

    def forward(self, context_entities, llm_inputs, labels=None):

        kg_embeds = self.get_kg_feature()
        entities_embeds = kg_embeds[context_entities]

        inputs_llm = self.bridge_proj(entities_embeds)
        atts_llm = create_mask(context_entities)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = "left"

        text_input_tokens = self.llm_tokenizer(
            llm_inputs,
            padding="longest",
            max_length=self.max_llm_prompt_len,
            truncation=True,
            return_tensors="pt",
        ).to(entities_embeds.device)

        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            labels,
            padding="longest",
            max_length=self.max_output_txt_len,
            truncation=True,
            return_tensors="pt",
        ).to(entities_embeds.device)

        llm_tokens, input_part_target_len = concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # do not apply loss to the padding
        targets = llm_tokens["input_ids"].masked_fill(llm_tokens["input_ids"] == self.llm_tokenizer.pad_token_id, -100)

        # do not apply loss to the text input (i.e., instruction and history)
        for i, l in enumerate(input_part_target_len):
            targets[i, :l] = -100

        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(atts_llm.size(), dtype=torch.long).to(inputs_llm.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.model.get_input_embeddings()(llm_tokens["input_ids"])
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens["attention_mask"]], dim=1)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=targets,
        )

        lm_loss = outputs.loss
        return COMPASSOutput(
            loss_lm=lm_loss,
            loss=lm_loss,
        )

    @torch.no_grad()
    def generate(self,
                 context_entities,
                 llm_inputs,
                 use_nucleus_sampling=False,
                 num_beams=3,
                 max_length=None,
                 min_length=10,
                 top_p=None,
                 top_k=None,
                 repetition_penalty=1.5,
                 length_penalty=1,
                 num_captions=1,
                 temperature=1,
                 max_new_tokens=512,
                 labels=None,
                 ):
        self.llm_tokenizer.padding_side = "left"
        # TODO save kg_embeds so that we don't have to compute it again
        kg_embeds = self.get_kg_feature()
        entities_embeds = kg_embeds[context_entities]

        inputs_llm = self.bridge_proj(entities_embeds)
        atts_llm = create_mask(context_entities)

        llm_tokens = self.llm_tokenizer(
            llm_inputs,
            padding="longest",
            return_tensors="pt",
        ).to(entities_embeds.device)

        # use labels to found max new token so we can save some time as we need to set a hard max length
        if labels is not None:
            out_tokens = self.llm_tokenizer(
                labels,
                padding=True,
                truncation=True,
                max_length=self.max_output_txt_len,
                return_tensors="pt",
            )
            max_new_tokens = out_tokens.input_ids.shape[1]

        input_embeds = self.model.get_input_embeddings()(llm_tokens["input_ids"])
        input_embeds = torch.cat([inputs_llm, input_embeds], dim=1)
        attention_masks = torch.cat([atts_llm, llm_tokens["attention_mask"]], dim=1)

        generation_outputs = self.model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_masks,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            eos_token_id=self.terminators,
            pad_token_id=self.llm_tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
            max_new_tokens=max_new_tokens,
        )

        token_ids = generation_outputs.sequences

        # only get the generated text
        token_ids[token_ids == 0] = 2  # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return COMPASSOutput(
            generated_text=output_text,
            last_hidden_states=self.get_last_embed(generation_outputs),
        )

    @torch.no_grad()
    def get_last_embed(self, generated):
        # get last token embeddings
        # last step, last layer, all batches, last token, dim
        token_embeds = generated.hidden_states[-1][-1][:, -1, :]
        return token_embeds  # bs, dim


@dataclass
class COMPASSOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None

    loss_lm: Optional[torch.FloatTensor] = None

    generated_text: Optional[List[str]] = None

    last_hidden_states: Optional[torch.FloatTensor] = None
