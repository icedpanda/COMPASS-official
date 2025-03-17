from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from transformers import BertModel
from transformers.modeling_outputs import ModelOutput

from src.llmcrs.data.dataset.utils import edge_to_pyg_format
from src.llmcrs.models.utils import SelfAttentionModule, get_pooling_method
from src.llmcrs.models.utils.loss import ContrastiveLoss
from src.llmcrs.models.utils.modules import create_mask
from src.llmcrs.models.utils.modules.fusion import GateFusion
from src.llmcrs.models.utils.modules.graph import Feedforward, GatedResidual


class KBRD(nn.Module):
    def __init__(self,
                 edge,
                 n_entity: int,
                 n_relation: int,
                 n_bases: int,
                 kg_dim: int = 128,
                 item_dim: int = 128,
                 use_bert: bool = True,
                 use_llm_embedding: bool = False,
                 pooling_method: str = "cls",
                 use_cl: bool = True,
                 ):
        super().__init__()
        edge_idx, edge_type = edge_to_pyg_format(edge)
        assert edge_idx.size()[1] == edge_type.size()[0], "Mismatch in number of edges and edge types"
        assert not (use_bert and use_llm_embedding), "use_bert and use_llm_embedding cannot be True at the same time"

        self.register_buffer("edge_idx", edge_idx)
        self.register_buffer("edge_type", edge_type)
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.n_bases = n_bases
        self.item_dim = item_dim
        self.kg_dim = kg_dim
        self.use_cl = use_cl
        # same as KBRD
        self.graph_encoder = RGCNConv(
            self.n_entity,
            self.kg_dim,
            num_relations=self.n_relation,
            num_bases=self.n_bases,
        )
        self.kg_attention = SelfAttentionModule(self.kg_dim, self.item_dim)
        self.use_bert = use_bert
        self.use_llm_embedding = use_llm_embedding
        if self.use_bert:
            self.bert = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False)
            self.bert_align = nn.Linear(self.bert.config.hidden_size, self.kg_dim)
            self.pooling_layer = get_pooling_method(pooling_method)
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
        if use_cl:
            assert use_cl == use_bert or use_cl == use_llm_embedding, "contrastive learning need two views"
            self.kg_proj = nn.Linear(self.kg_dim, 128)
            self.context_proj = nn.Linear(self.kg_dim, 128)
            self.cl_loss = ContrastiveLoss(0.07, 0.1)
        self.rec_layer = nn.Linear(self.item_dim, self.n_entity)
        self.rec_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, context_entities, context_tokens, llm_tokens, labels=None):
        user_embedding, kg_embedding = self.encoder_user_kg(context_entities)
        if self.use_bert or self.use_llm_embedding:
            user_embedding, user_context_embeds, user_kg_embeds = self.encoder_reasoning(
                user_embedding,
                context_tokens,
                llm_tokens)

        sims = F.linear(user_embedding, kg_embedding, self.rec_layer.bias)
        if labels is not None:
            rec_loss = self.rec_loss(sims, labels)
            cl_loss = 0
            if self.use_cl and (self.use_bert or self.use_llm_embedding):
                cl_loss = self.cl_loss(self.kg_proj(user_kg_embeds), self.context_proj(user_context_embeds))
            total_loss = rec_loss + cl_loss * 0.5
            return KBRDOutputs(
                sims=sims,
                loss=total_loss,
                loss_rec=rec_loss,
                loss_cl=cl_loss
            )
        return KBRDOutputs(
            sims=sims
        )

    def encoder_reasoning(self, user_kg_embedding, context_tokens, llm_embeds):
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
        user_embeddings = self.gateway(user_kg_embedding, context_embeddings)
        return user_embeddings, context_embeddings, user_kg_embedding

    def compute_entity_representations(self):
        return self.graph_encoder(None, self.edge_idx, self.edge_type)

    def encoder_user_kg(self, context_entities):
        """
        Encode the user preference using the context entities and entity turns.
        """
        kg_embedding = self.compute_entity_representations()
        user_embedding = kg_embedding[context_entities]
        entity_masks = create_mask(context_entities)
        user_embedding = self.kg_attention(user_embedding, entity_masks.unsqueeze(-1))
        return user_embedding, kg_embedding


@dataclass
class KBRDOutputs(ModelOutput):
    sims: torch.Tensor = None

    loss: torch.FloatTensor = None

    loss_cl: torch.FloatTensor = None

    loss_rec: torch.FloatTensor = None
