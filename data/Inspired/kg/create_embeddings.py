import torch
import json
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
else:
    print("MPS is available.")

# Load BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

# Load the entity dictionary
with open("entity_id2title.json", "r") as f:
    entity_dict = json.load(f)

with open("id2relation.json", "r") as f:
    id2relation = json.load(f)

# Add empty string for index 0 (padding)
entity_list = [""] + list(entity_dict.values())
print(entity_list[:5])

relation_list = list(id2relation.values())
print(relation_list[:5])

device = torch.device("mps" if torch.cuda.is_available() else "cpu")
model.to(device)


def avg_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Calculates the average pooling of the last hidden states.

    Args:
        last_hidden_states (Tensor): The hidden states to be pooled. Shape: (batch_size, sequence_length, hidden_size).
        attention_mask (Tensor): The attention mask tensor. Shape: (batch_size, sequence_length).

    Returns:
        Tensor: The tensor resulting from the average pooling operation. Shape: (batch_size, hidden_size).
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


@torch.inference_mode()
def encode_entity(entity_text):
    inputs = tokenizer(entity_text, return_tensors='pt', padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    pooled_output = avg_pooling(outputs.last_hidden_state, inputs['attention_mask'])
    return pooled_output.cpu()


# Encode all entities in the list
encoded_entities = [encode_entity(text) for text in tqdm(entity_list)]

# Stack encoded entities into a tensor
entity_embeddings = torch.stack(encoded_entities)
entity_embeddings = entity_embeddings.squeeze(1)
# convert to cpu
entity_embeddings = entity_embeddings.to('cpu')

# Save encoded entities
torch.save(entity_embeddings, 'bert_entity_embeddings.pt')

# Example usage: print encoded representation for entity at index 1
print(entity_embeddings.shape)


# for relation embedding
encoded_relations = [encode_entity(text) for text in tqdm(relation_list)]

# Stack encoded entities into a tensor
relation_embeddings = torch.stack(encoded_relations)
relation_embeddings = relation_embeddings.squeeze(1)
# convert to cpu
relation_embeddings = relation_embeddings.to('cpu')

# Save encoded entities
torch.save(relation_embeddings, 'bert_relation_embeddings.pt')

# Example usage: print encoded representation for entity at index 1
print(relation_embeddings.shape)
