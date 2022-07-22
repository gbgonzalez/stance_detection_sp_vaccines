import torch
from model.preprocessing import tweet_preprocesing_embedding
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, RobertaTokenizer, XLMRobertaTokenizer
from .preprocessing import tweet_preprocesing_embedding

def dataloader_embedding(MAX_LEN, batch_size, X_train, y_train, model):

    train_inputs, train_masks = _preprocessing_for_embeddings(X_train, MAX_LEN, bert_model=model)
    train_labels = torch.tensor(y_train)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    return train_data, train_sampler, train_dataloader

def _preprocessing_for_embeddings(data, MAX_LEN, bert_model):
    input_ids = []
    attention_masks = []
    tokenizer = RobertaTokenizer.from_pretrained(bert_model, do_lower_case=True)
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=tweet_preprocesing_embedding(sent),  # Preprocess sentence
            add_special_tokens=True,
            max_length=MAX_LEN,
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True
        )

        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks