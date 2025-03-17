from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, GCNConv
from transformers import BertModel
from transformers.modeling_outputs import ModelOutput

from src.llmcrs.data.dataset.utils import edge_to_pyg_format
from src.llmcrs.models.utils import SelfAttentionModule, get_pooling_method
from src.llmcrs.models.utils.loss import ContrastiveLoss
from src.llmcrs.models.utils.modules import create_mask
from src.llmcrs.models.utils.modules.fusion import GateFusion
from src.llmcrs.models.utils.modules.graph import Feedforward, GatedResidual

@dataclass
class KGSFOutputs(ModelOutput):
    loss: torch.FloatTensor = None

    loss_rec: torch.FloatTensor = None

    loss_info: torch.FloatTensor = None

    loss_cl: torch.FloatTensor = None

    sims: torch.Tensor = None


class KGSF(nn.Module):
    def __init__(self,
                 edge,
                 word_edge,
                 n_words: int,
                 n_entity: int,
                 n_relation: int,
                 n_bases: int,
                 kg_dim: int = 128,
                 item_dim: int = 128,
                 use_bert: bool = False,
                 use_llm_embedding: bool = False,
                 pooling_method: str = "cls",
                 use_cl: bool = False,
                 ):
        super().__init__()
        edge_idx, edge_type = edge_to_pyg_format(edge)
        word_edge = edge_to_pyg_format(word_edge, "GCN")
        assert edge_idx.size()[1] == edge_type.size()[0], "Mismatch in number of edges and edge types"

        self.edge_idx = edge_idx
        self.edge_type = edge_type
        self.word_edge = word_edge
        self.n_word = n_words
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.n_bases = n_bases
        self.item_dim = item_dim
        self.kg_dim = kg_dim
        self.use_bert = use_bert
        self.use_cl = use_cl
        self.use_llm_embedding = use_llm_embedding
        self.rec_layer = nn.Linear(self.item_dim, self.n_entity)
        self.default_gate_layer = GateFusion(self.kg_dim, self.kg_dim)
        self.init_kg_encoder()
        self.init_loss()

        if self.use_bert:
            self.pooling_method = pooling_method
            self.init_bert()
        if self.use_llm_embedding:
            self.llm_dim = 4096
            self.ffn = Feedforward(
                self.llm_dim,  # llama embedding size
                self.llm_dim // 2,
                self.llm_dim
            )
            self.ffn_gate = GatedResidual(self.llm_dim)
            self.llm_align = nn.Linear(self.llm_dim, self.kg_dim)
        if self.use_bert or self.use_llm_embedding:
            self.gateway = GateFusion(self.kg_dim, self.kg_dim)

    def init_kg_encoder(self):
        self.entity_encoder = RGCNConv(
            self.n_entity,
            self.kg_dim,
            num_relations=self.n_relation,
            num_bases=self.n_bases,
        )
        self.entity_attention = SelfAttentionModule(self.kg_dim, self.item_dim)
        self.word_kg_embedding = nn.Embedding(self.n_word, self.kg_dim, 0)
        nn.init.normal_(self.word_kg_embedding.weight, mean=0, std=self.kg_dim ** -0.5)
        nn.init.constant_(self.word_kg_embedding.weight[0], 0)
        self.word_encoder = GCNConv(self.kg_dim, self.kg_dim)
        self.word_attention = SelfAttentionModule(self.kg_dim, self.item_dim)

    def init_loss(self):
        self.rec_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        # kgsf pretraining loss
        self.info_max_norm = nn.Linear(self.kg_dim, self.kg_dim)
        self.info_max_bias = nn.Linear(self.kg_dim, self.n_entity)
        self.info_max_loss = nn.MSELoss(reduction='sum')
        if self.use_cl and (self.use_bert or self.use_llm_embedding):
            self.cl_loss = ContrastiveLoss(0.07, 0.1)
            self.base_proj = nn.Linear(self.kg_dim, 128)
            self.context_proj = nn.Linear(self.kg_dim, 128)

    def init_bert(self):
        self.bert = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False)
        self.bert_align = nn.Linear(self.bert.config.hidden_size, self.kg_dim)
        self.pooling_layer = get_pooling_method(self.pooling_method)

    def pretrain_info_max(self, context_words, entity_labels, context_entities, context_tokens, llm_compressed_tokens):
        loss_mask = torch.sum(entity_labels)
        if loss_mask.item() == 0:
            return None
        entity_kg_rep = self.compute_entity_representations()
        word_kg_rep = self.compute_word_representations()

        word_rep = word_kg_rep[context_words]
        word_padding_mask = create_mask(context_words)

        word_rep = self.word_attention(word_rep, word_padding_mask.unsqueeze(-1))
        word_info_rep = self.info_max_norm(word_rep)
        info_predict = F.linear(word_info_rep, entity_kg_rep, self.info_max_bias.bias)
        loss = self.info_max_loss(info_predict, entity_labels) / loss_mask
        cl_loss = 0
        if self.use_cl and (self.use_bert or self.use_llm_embedding):
            user_kg_embeds = entity_kg_rep[context_entities]
            entity_masks = create_mask(context_entities)
            kg_attn_embeds = self.entity_attention(user_kg_embeds, entity_masks.unsqueeze(-1))
            _, user_context_embeds, base_rep = self.encoder_reasoning(
                kg_attn_embeds,
                word_rep,
                context_tokens,
                llm_compressed_tokens
            )
            cl_loss = self.compute_cl(base_rep, user_context_embeds)

        return KGSFOutputs(
            loss_info=loss,
            loss=loss + cl_loss,
            loss_cl=cl_loss
        )

    def forward(self, context_entities, context_words, context_tokens, llm_compressed_tokens, labels=None,
                entity_labels=None):
        entity_kg_rep = self.compute_entity_representations()
        word_kg_rep = self.compute_word_representations()

        entity_masks = create_mask(context_entities)
        user_kg_embeds = entity_kg_rep[context_entities]
        kg_attn_embeds = self.entity_attention(user_kg_embeds, entity_masks.unsqueeze(-1))

        word_rep = word_kg_rep[context_words]
        word_padding_mask = create_mask(context_words)
        word_attn_rep = self.word_attention(word_rep, word_padding_mask.unsqueeze(-1))

        if self.use_bert or self.use_llm_embedding:
            user_embedding, user_context_embeds, base_rep = self.encoder_reasoning(
                kg_attn_embeds,
                word_attn_rep,
                context_tokens,
                llm_compressed_tokens
            )
        else:
            user_embedding = self.default_gate_layer(kg_attn_embeds, word_attn_rep)

        sims = F.linear(user_embedding, entity_kg_rep, self.rec_layer.bias)

        if labels is not None:
            info_loss_mask = torch.sum(entity_labels)
            if info_loss_mask.item() == 0:
                loss_info = None
            else:
                word_info_rep = self.info_max_norm(word_attn_rep)
                info_predict = F.linear(word_info_rep, entity_kg_rep, self.info_max_bias.bias)
                loss_info = self.info_max_loss(info_predict, entity_labels) / info_loss_mask
            rec_loss = self.rec_loss(sims, labels)
            cl_loss = 0
            if self.use_cl and (self.use_bert or self.use_llm_embedding):
                # cl_loss = self.compute_cl(kg_attn_embeds, user_context_embeds, word_attn_rep)
                cl_loss = self.compute_cl(base_rep, user_context_embeds)
            loss_info = 0 if loss_info is None else loss_info * 0.025  # same as the original paper
            total_loss = rec_loss + loss_info + cl_loss * 0.1
            return KGSFOutputs(
                sims=sims,
                loss=total_loss,
                loss_rec=rec_loss,
                loss_info=loss_info,
                loss_cl=cl_loss
            )

        # testing only needs similarity scores
        return KGSFOutputs(
            sims=sims,
        )

    def encoder_reasoning(self, db_rep, word_rep, context_tokens, llm_embeds):
        base_embedding = self.default_gate_layer(db_rep, word_rep)
        if self.use_bert:
            context_embeddings = self.bert(**context_tokens).last_hidden_state
            context_embeddings = self.bert_align(
                self.pooling_layer(context_embeddings, context_tokens["attention_mask"]))
        elif self.use_llm_embedding:
            context_embeddings = self.ffn_gate(self.ffn(llm_embeds), llm_embeds)  # x, residual
            context_embeddings = self.llm_align(context_embeddings)
        else:
            raise NotImplementedError(
                "Either use_bert or use_llm_embedding should be True. Both are False.")
        user_embeddings = self.gateway(base_embedding, context_embeddings)
        return user_embeddings, context_embeddings, base_embedding

    def compute_entity_representations(self):
        return self.entity_encoder(None, self.edge_idx, self.edge_type)

    def compute_word_representations(self):
        return self.word_encoder(self.word_kg_embedding.weight, self.word_edge)

    # def compute_cl(self, kg_rep, context_rep, word_rep):
    #     kg_rep = self.kg_proj(kg_rep)
    #     context_proj = self.context_proj(context_rep)
    #     word_proj = self.words_proj(word_rep)
    #
    #     # 3 pairs
    #     uc_loss = self.cl_loss(kg_rep, context_proj)
    #     uw_loss = self.cl_loss(kg_rep, word_proj)
    #     cw_loss = self.cl_loss(word_proj, context_proj)
    #     return (uc_loss + uw_loss + cw_loss) / 3.0

    def compute_cl(self, base_rep, context_rep):
        base_rep = self.base_proj(base_rep)
        context_proj = self.context_proj(context_rep)
        return self.cl_loss(base_rep, context_proj)
