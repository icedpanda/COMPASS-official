
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import BertModel
from transformers.modeling_outputs import ModelOutput
from src.llmcrs.models.utils import get_pooling_method
from src.llmcrs.models.utils.modules.graph import GatedResidual, Feedforward
from src.llmcrs.models.utils.modules.fusion import GateFusion
from src.llmcrs.models.utils.loss import ContrastiveLoss

ALLOWED_BERT_MODELS = ["bert-base-uncased", "bert-large-uncased"]


@dataclass
class BERTOutputs(ModelOutput):
    logits: torch.Tensor = None

    loss: torch.FloatTensor = None

    loss_cl: torch.FloatTensor = None

    loss_rec: torch.FloatTensor = None


class BERTModel(nn.Module):
    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 pooling_strategy: str = "avg",
                 item_dim: int = 768,
                 n_classes: int = None,
                 use_bert: bool = True,
                 use_llm_embedding: bool = False,
                 pooling_method: str = "cls",
                 use_cl: bool = True,
                 ):
        super().__init__()

        if model_name not in ALLOWED_BERT_MODELS:
            raise ValueError(f"Model name {model_name} is not supported yet.")

        self.model = BertModel.from_pretrained(model_name, add_pooling_layer=False)
        self.hidden_size = self.model.config.hidden_size
        self.pooler_output_size = item_dim
        self.pooling_layer = get_pooling_method(pooling_strategy)
        self.rec_gate = GatedResidual(item_dim)
        self.rec_head = Feedforward(item_dim, item_dim)

        self.n_classes = n_classes
        self.classifier = nn.Linear(item_dim, self.n_classes)
        self.rec_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.use_bert = use_bert
        self.use_cl = use_cl
        self.use_llm_embedding = use_llm_embedding

        if self.use_bert:
            self.bert = BertModel.from_pretrained(model_name, add_pooling_layer=False)
            self.encoder_pooling_layer = get_pooling_method(pooling_method)
        if self.use_llm_embedding:
            self.llm_dim = 4096
            self.ffn = Feedforward(
                self.llm_dim,  # llama embedding size
                self.llm_dim // 2,
                self.llm_dim
            )
            self.ffn_gate = GatedResidual(self.llm_dim)
            self.llm_align = nn.Linear(self.llm_dim, item_dim)
        if self.use_bert or self.use_llm_embedding:
            self.gateway = GateFusion(item_dim, item_dim)
        if use_cl:
            assert use_cl == use_bert or use_cl == use_llm_embedding, "contrastive learning need two views"
            self.summary_proj = nn.Linear(self.model.config.hidden_size, 128)
            self.context_proj = nn.Linear(self.model.config.hidden_size, 128)
            self.cl_loss = ContrastiveLoss(0.07, 0.1)

    def forward(self, history_tokens, llm_tokens, llm_compressed_tokens, labels=None):
        history_embeds = self.model(**history_tokens)
        history_embeds = self.pooling_layer(history_embeds.last_hidden_state, history_tokens.attention_mask)
        if self.use_llm_embedding or self.use_bert:
            llm_embeddings = self.encode_user_preference(llm_tokens, llm_compressed_tokens)
            outputs = self.gateway(llm_embeddings, history_embeds)
            outputs = self.rec_gate(self.rec_head(outputs), outputs)
        else:
            outputs = self.rec_gate(self.rec_head(history_embeds), history_embeds)

        logits = self.classifier(outputs)
        if labels is not None:
            rec_loss = self.rec_loss(logits, labels)
            cl_loss = 0
            if self.use_cl and (self.use_bert or self.use_llm_embedding):
                cl_loss = self.cl_loss(self.summary_proj(llm_embeddings), self.context_proj(history_embeds))
            total_loss = rec_loss + cl_loss * 0.3
            return BERTOutputs(
                logits=logits,
                loss=total_loss,
                loss_rec=rec_loss,
                loss_cl=cl_loss
                )
        return BERTOutputs(
                logits=logits,
                )

    def encode_user_preference(self, llm_tokens, llm_compressed_tokens):
        if self.use_bert:
            llm_embeddings = self.bert(**llm_tokens).last_hidden_state
            llm_embeddings = self.encoder_pooling_layer(llm_embeddings, llm_tokens.attention_mask)
        elif self.use_llm_embedding:
            llm_embeddings = self.ffn_gate(self.ffn(llm_compressed_tokens), llm_compressed_tokens)  # x, residual
            llm_embeddings = self.llm_align(llm_embeddings)
        else:
            raise NotImplementedError(
                "Either use_bert or use_llm_embedding should be True. Both are False.")
        return llm_embeddings


