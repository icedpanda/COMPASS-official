import os

import torch
import torch.nn as nn
from dotenv import load_dotenv
from loguru import logger
from peft import LoraConfig, get_peft_model
from transformers import LlamaModel, AutoTokenizer, LlamaForCausalLM

from src.llmcrs.models.recommendation.llama.llama_outputs import LLAMARecOutput
from src.llmcrs.models.utils import get_pooling_method
from src.llmcrs.models.utils.modules.graph import GatedResidual, Feedforward

load_dotenv()
ALLOWED_BERT_MODELS = {"meta-llama/Llama-2-7b-chat-hf",
                       "meta-llama/Llama-2-13b-chat-hf",
                       "meta-llama/Meta-Llama-3-8B",
                       "meta-llama/Meta-Llama-3-8B-Instruct",
                       "meta-llama/Llama-3.1-8B-Instruct",
                       "meta-llama/Llama-3.1-8B",
                       "lmsys/vicuna-7b-v1.5-16k",
                       "lmsys/vicuna-7b-v1.5"
                       }
TOKEN = os.getenv("HUGGINGFACE_API_KEY")


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


class LLAMA(nn.Module):
    def __init__(self,
                 lora_config: LoraConfig,
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 pooling_strategy: str = "last_token",
                 item_dim: int = 768,
                 n_classes: int = None,
                 max_length_prompt: int = 8000,
                 max_output_txt_len: int = 1024,
                 reasoner: bool = False,
                 ):
        super().__init__()

        if model_name not in ALLOWED_BERT_MODELS:
            raise ValueError(f"Model name {model_name} is not supported yet.")
        if n_classes and reasoner:
            raise ValueError("Cannot have both a reasoner and a classifier.")

        if reasoner:
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                token=TOKEN,
                torch_dtype=torch.bfloat16,
                use_flash_attention_2=True
            )
        else:
            model = LlamaModel.from_pretrained(
                model_name,
                token=TOKEN,
                torch_dtype=torch.bfloat16,
                use_flash_attention_2=True
            )
        #
        # model.config.pad_token_id = 0
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=TOKEN)
        self.llm_tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.llm_tokenizer.pad_token = "<pad>"

        if reasoner:
            self.terminators = [
                self.llm_tokenizer.eos_token_id,
                self.llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        model.resize_token_embeddings(len(self.llm_tokenizer))
        self.hidden_size = model.config.hidden_size
        self.max_llm_prompt_len = max_length_prompt
        self.max_output_txt_len = max_output_txt_len
        logger.info(f"Lora config: %s {lora_config}")
        self.model = get_peft_model(model, lora_config)
        # self.model = self.model.merge_and_unload()

        self.n_classes = n_classes
        if self.n_classes is not None:
            self.pooling_layer = get_pooling_method(pooling_strategy)
            self.rec_align = nn.Linear(model.config.hidden_size, item_dim)
            self.rec_gate = GatedResidual(item_dim)
            self.rec_head = Feedforward(item_dim, item_dim)
            logger.info(f"Adding a linear layer with {self.n_classes} classes for classification.")
            self.classifier = nn.Linear(item_dim, self.n_classes)

    def forward(self, llm_inputs, reasoning=None):
        # concat the llm inputs and the reasoning
        if reasoning is not None:
            # llm_inputs and reasoning are yet tokenized. It's a list of strings
            llm_inputs = [f"{llm_input}\nUser Preference:\n{reason}" for llm_input, reason in
                          zip(llm_inputs, reasoning)]

        text_input_tokens = self.llm_tokenizer(
            llm_inputs,
            padding="longest",
            max_length=self.max_llm_prompt_len,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**text_input_tokens)
        outputs = self.pooling_layer(outputs.last_hidden_state, text_input_tokens["attention_mask"])
        outputs = self.rec_align(outputs)
        outputs = self.rec_gate(self.rec_head(outputs), outputs)
        if self.n_classes:
            outputs = self.classifier(outputs)
        return outputs

    @property
    def device(self):
        return list(self.parameters())[0].device

    def forward_with_reasoning(self, llm_inputs, labels=None):
        # this one aim to train the model with the reasoning generation

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = "left"
        text_input_tokens = self.llm_tokenizer(
            llm_inputs,
            padding="longest",
            max_length=self.max_llm_prompt_len,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            labels,
            padding="longest",
            max_length=self.max_output_txt_len,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        llm_tokens, input_part_target_len = concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )
        # do not apply loss to the padding
        targets = llm_tokens["input_ids"].masked_fill(llm_tokens["input_ids"] == self.llm_tokenizer.pad_token_id, -100)
        # do not apply loss to the text input (i.e. instruction and history)
        for i, l in enumerate(input_part_target_len):
            targets[i, :l] = -100

        outputs = self.model(
            input_ids=llm_tokens["input_ids"],
            attention_mask=llm_tokens["attention_mask"],
            return_dict=True,
            output_hidden_states=True,
            labels=targets,
        )

        lm_loss = outputs.loss

        return LLAMARecOutput(
            loss=lm_loss
        )

    def make_rec(self, user_embeddings, labels=None):

        user_embeddings = self.rec_align(user_embeddings)
        user_embeddings = self.rec_gate(self.rec_head(user_embeddings), user_embeddings)
        rec_logits = self.classifier(user_embeddings)

        if labels is not None:
            rec_loss = self.rec_loss(rec_logits, labels)
            return LLAMARecOutput(sims=rec_logits, loss=rec_loss)

        return LLAMARecOutput(sims=rec_logits)

    @torch.no_grad()
    def generate(self,
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
        text_input_tokens = self.llm_tokenizer(
            llm_inputs,
            padding="longest",
            max_length=self.max_llm_prompt_len,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        if labels is not None:
            out_tokens = self.llm_tokenizer(
                labels,
                padding=True,
                truncation=True,
                max_length=self.max_output_txt_len,
                return_tensors="pt",
            )
            max_new_tokens = out_tokens.input_ids.size(1)

        generation_outputs = self.model.generate(
            input_ids=text_input_tokens["input_ids"],
            attention_mask=text_input_tokens["attention_mask"],
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
        # keep generated text only
        token_ids = token_ids[:, text_input_tokens["input_ids"].size(1):]

        output_text = self.llm_tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return LLAMARecOutput(
            generated_text=output_text,
            last_hidden_states=self.get_last_embed(generation_outputs),
        )

    @torch.no_grad()
    def get_last_embed(self, generated):
        # get last token embeddings
        # last step, last layer, all batches, last token, dim
        token_embeds = generated.hidden_states[-1][-1][:, -1, :]
        return token_embeds  # bs,
