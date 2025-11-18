from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
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


def get_tokenized_datasets(tokenizer):
    df_nsmc = get_nsmc_dataset()
    tokenized_datasets = df_nsmc.map(
        preprocess_function, 
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": 128},
        remove_columns=["text", "sentiment"]
    )
    train_dataset = tokenized_datasets.train_test_split(test_size=0.2)
    tokenized_datasets = DatasetDict({
        "train": train_dataset["train"],
        "test": train_dataset["test"]
    })
    print(tokenized_datasets)

    return tokenized_datasets


def hp_space(trial):
    return {
        "learning_rate" : trial.suggest_float("learning_rate", 5e-6, 1e-5, log = True),
        "per_device_train_batch_size" : trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "per_device_eval_batch_size" : trial.suggest_categorical("per_device_eval_batch_size", [8, 16, 32]),
        "num_train_epochs" : trial.suggest_int("num_train_epochs", 1, 10),
        "weight_decay" : trial.suggest_float("weight_decay", 0.3, 0.5),
        "warmup_ratio" : trial.suggest_float("warmup_ratio", 0.0, 0.2),
        "max_grad_norm" : trial.suggest_float("max_grad_norm", 0.5, 1.0)
    }


def search_best_hyperparameters_kcbert_nsmc(
    model_name,
    num_epochs,
    batch_size,
    learning_rate,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = get_tokenized_datasets(tokenizer)


    def model_init():
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model.to(device)
        return model

    training_args = TrainingArguments(
        output_dir="./models/kc-bert-nsmc",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        bf16=True,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy"
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    best_run = trainer.hyperparameter_search(
        direction = "maximize",
        hp_space = hp_space,
        n_trials = 20,
        compute_objective = lambda metrics: metrics["eval_accuracy"]
    )

    return best_run


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'f1': f1,
        'recall': recall,
        'precision': precision
    }


if __name__ == "__main__":
    model_name = "beomi/kcbert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    bset_run = search_best_hyperparameters_kcbert_nsmc(
        model_name=model_name,
        num_epochs=3,
        batch_size=8,
        learning_rate=2e-5
    )

    print(bset_run)
