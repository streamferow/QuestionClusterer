from Data.Raw import raw_data
from Data.Processed import processed_data
from transformers import AutoModel, AutoTokenizer
import torch, random, pandas as pd, numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(111)


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def texts_to_tensor(data_frame: pd.DataFrame, text_column='text', batch_size=8):
    model_name = "DeepPavlov/rubert-base-cased"
    tokenizer, model = load_model(model_name)

    embeddings_list = []
    for i in range(0, len(data_frame), batch_size):
        batch_texts = data_frame[text_column].iloc[i:i + batch_size].tolist()
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=1024, return_tensors="pt")

        with torch.no_grad():
            model_output = model(**inputs)

        embeddings = mean_pooling(model_output, inputs['attention_mask'])
        embeddings_list.append(embeddings)

    return torch.cat(embeddings_list, dim=0)


def save_tensor(path: str, tensor):
    torch.save(tensor, path)


def load_tensor(path):
    return torch.load(path, weights_only=False)


final_embeddings = load_tensor("/Users/ivan/Desktop/final_embeddings.pt")
