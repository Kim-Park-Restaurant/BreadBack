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
from pathlib import Path

import torch
import numpy as np


def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=256
    )


def compute_metrics(eval_pred):
    """평가 메트릭 계산"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }


def fine_tune_kcbert_full(
    model_name: str,
    csv_path: str,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    train_ratio: float,
    sample_ratio: float
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 장치 설정
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"사용 장치: {device}")
    
    # 데이터셋 로드
    print("데이터셋 로드 중...")
    dataset_dict = create_datadict_from_csv(
        csv_path, 
        train_ratio=train_ratio, 
        sample_ratio=sample_ratio
    )
    print(f"Train dataset: {len(dataset_dict['train'])}")
    print(f"Test dataset: {len(dataset_dict['test'])}")
    
    # 레이블 검증
    train_labels = set(dataset_dict['train']['sentiment'])
    test_labels = set(dataset_dict['test']['sentiment'])
    
    print(f"모델 로드 중: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    
    for param in model.parameters():
        param.requires_grad = True
    
    model.to(device)
    dataset_dict = dataset_dict.rename_column("sentiment", "labels")
    
    columns_to_remove = [col for col in dataset_dict["train"].column_names if col != "labels"]
    
    tokenized_datasets = dataset_dict.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
    ).remove_columns(columns_to_remove)
    
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        max_length=256
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
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
    
    # 최종 평가
    print("최종 평가 중...")
    eval_results = trainer.evaluate()
    print(f"평가 결과: {eval_results}")
    
    return trainer, model, tokenizer


if __name__ == "__main__":
    model_name = "beomi/kcbert-base"
    
    fine_tune_kcbert_full(
        model_name=model_name,
        csv_path="./data/what_the.csv",
        output_dir="./models/finetuned-kcbert",
        num_epochs=3,
        batch_size=8,
        learning_rate=2e-5,
        train_ratio=0.8,
        sample_ratio=1.0
    )

