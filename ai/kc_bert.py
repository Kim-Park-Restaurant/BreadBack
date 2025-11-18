from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score
from data.preprocess_data import get_nsmc_dataset, get_komultitext_dataset
from datasets import DatasetDict

import torch
import numpy as np



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def preprocess_function(examples, tokenizer, max_length=512):
    texts = examples["text"]
    labels = examples["sentiment"]

    tokenized_text = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length
    )
    tokenized_text["labels"] = labels

    return tokenized_text


def kc_bert_nsmc(
    model_name,
    num_epochs,
    batch_size,
    learning_rate,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    df_nsmc = get_nsmc_dataset()
    tokenized_datasets = df_nsmc.map(
        preprocess_function, 
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": 512},
        remove_columns=["text", "sentiment"]
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return tokenized_datasets

if __name__ == "__main__":
    model_name = "beomi/kcbert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    test = kc_bert_nsmc(
        model_name=model_name,
        num_epochs=3,
        batch_size=8,
        learning_rate=2e-5
    )

