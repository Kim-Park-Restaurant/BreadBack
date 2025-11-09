import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from data.preprocess_data import create_datadict_from_csv

import sys
import os


def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["text"],  # 독립변수: 입력 텍스트
        truncation=True,
        padding=True,
        max_length=256
    )


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }


def fine_tune_korean_bert(
    model_name: str = "monologg/kobert",
    csv_path: str = "./data/final_df.csv",
    output_dir: str = "./models/finetuned-bert",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    dataset_dict = create_datadict_from_csv(csv_path, train_ratio=0.8, sample_ratio=0.5)
    print(f"Train dataset: {len(dataset_dict['train'])}")
    print(f"Test dataset: {len(dataset_dict['test'])}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    dataset_dict = dataset_dict.rename_column("sentiment", "labels")
    
    columns_to_remove = [col for col in dataset_dict["train"].column_names if col != "labels"]
    print(columns_to_remove)


    tokenized_datasets = dataset_dict.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=columns_to_remove
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy"
    )
    
    # Trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()

    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    eval_results = trainer.evaluate()
    print(f"평가 결과: {eval_results}")
    
    return trainer, model, tokenizer


if __name__ == "__main__":
    model_name = "monologg/kobert"
    
    fine_tune_korean_bert(
        model_name=model_name,
        csv_path="./data/final_df.csv",
        output_dir="./models/finetuned-bert",
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-5
    )

