from dataclasses import dataclass
from typing import Optional, List
import torch
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)


@dataclass
class LLAMARecOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None

    generated_text: Optional[List[str]] = None

    last_hidden_states: Optional[torch.FloatTensor] = None

    sims: Optional[torch.FloatTensor] = None

