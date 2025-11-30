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
import os


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


def get_tokenized_datasets(tokenizer, train_dataset, test_dataset=None, seed=42):
    """
    데이터셋을 토크나이징
    
    Args:
        tokenizer: 토크나이저
        train_dataset: 학습 데이터셋
        test_dataset: 테스트 데이터셋 (None이면 train에서 분할)
        seed: 랜덤 시드 (test_dataset이 None일 때만 사용)
    """
    # train 데이터 토크나이징
    tokenized_train = train_dataset.map(
        preprocess_function, 
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": 128},
        remove_columns=train_dataset.column_names
    )
    
    # test 데이터 처리
    if test_dataset is not None:
        # 별도로 전달된 test 데이터 토크나이징
        tokenized_test = test_dataset.map(
            preprocess_function,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer, "max_length": 128},
            remove_columns=test_dataset.column_names
        )
    else:
        # test 데이터가 없으면 train에서 분할 (기존 동작 유지)
        train_test_split = tokenized_train.train_test_split(test_size=0.2, seed=seed)
        tokenized_train = train_test_split["train"]
        tokenized_test = train_test_split["test"]

    tokenized_datasets = DatasetDict({
        "train": tokenized_train,
        "test": tokenized_test
    })
    print(tokenized_datasets)

    return tokenized_datasets


def hp_space(trial):
    return {    
        "learning_rate" : trial.suggest_float("learning_rate", 1e-5, 5e-5, log = True),
        "per_device_train_batch_size" : trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
        "per_device_eval_batch_size" : trial.suggest_categorical("per_device_eval_batch_size", [8, 16]),
        "num_train_epochs" : trial.suggest_int("num_train_epochs", 1, 3),
        "weight_decay" : trial.suggest_float("weight_decay", 0.0, 0.1),
        "warmup_ratio" : trial.suggest_float("warmup_ratio", 0.0, 0.2),
        "max_grad_norm" : trial.suggest_float("max_grad_norm", 0.5, 1.0),
        "lr_scheduler_type" : trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine", "cosine_with_restarts"])  # 스케줄러 타입도 탐색
    }


def search_best_hyperparameters(
    model_name,
    num_epochs,
    batch_size,
    learning_rate,
    train_dataset,
    test_dataset,
    output_dir,
    seed=42
):
    tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = get_tokenized_datasets(tokenizer, train_dataset, test_dataset=test_dataset, seed=seed)


    def model_init():
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model.to(device)
        return model

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        bf16=True,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",  # 학습률 스케줄러: linear, cosine, cosine_with_restarts 등
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy"
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],  # test를 validation으로 사용
        data_collator=data_collator,
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


def train_kcbert(model_name, best_run, output_dir, train_dataset, test_dataset, seed=42):
    # 로컬 경로인 경우 절대 경로로 변환 (상대 경로 문제 해결)
    if os.path.exists(model_name) and os.path.isdir(model_name):
        model_name = os.path.abspath(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = get_tokenized_datasets(tokenizer, train_dataset, test_dataset=test_dataset, seed=seed)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=best_run.hyperparameters["num_train_epochs"],
        per_device_train_batch_size=best_run.hyperparameters["per_device_train_batch_size"],
        per_device_eval_batch_size=best_run.hyperparameters["per_device_eval_batch_size"],
        learning_rate=best_run.hyperparameters["learning_rate"],
        bf16=True,
        warmup_ratio=best_run.hyperparameters["warmup_ratio"],
        weight_decay=best_run.hyperparameters["weight_decay"],
        max_grad_norm=best_run.hyperparameters["max_grad_norm"],
        lr_scheduler_type=best_run.hyperparameters.get("lr_scheduler_type", "linear"),  # 하이퍼파라미터에서 가져오거나 기본값 사용
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        report_to="wandb"
    )   

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],  # test를 validation으로 사용
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )   
    trainer.train()

    trainer.save_model()

    # test 데이터셋 평가
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print(f"Test 평가 결과: {test_results}")

    return trainer


if __name__ == "__main__":
    from ai.shared_config import SHARED_SEED
    
    # 동일한 seed로 두 모델이 같은 데이터 사용
    print(f"공유 seed: {SHARED_SEED}")
    
    # komultitext 데이터셋 로드 및 8:2 분할 (두 모델이 동일한 데이터 사용)
    komultitext_dataset = get_komultitext_dataset()
    komultitext_split = komultitext_dataset.train_test_split(test_size=0.2, seed=SHARED_SEED)
    komultitext_train = komultitext_split["train"]
    komultitext_test = komultitext_split["test"]
    
    print(f"komultitext train 크기: {len(komultitext_train)}")
    print(f"komultitext test 크기: {len(komultitext_test)}")
    
    default_num_epochs = 3
    default_batch_size = 8
    default_learning_rate = 2e-5
    
    # kc_bert 학습
    fine_tune_model_name = "./nsmc_huggingface_train/kc-bert-nsmc-final/checkpoint-350"
    
    print("\n" + "="*80)
    print("KcBERT 하이퍼파라미터 검색")
    print("="*80)
    best_run_kcbert = search_best_hyperparameters(
        model_name=fine_tune_model_name,
        num_epochs=default_num_epochs,
        batch_size=default_batch_size,
        learning_rate=default_learning_rate,
        train_dataset=komultitext_train,  # train 데이터
        test_dataset=komultitext_test,  # test 데이터
        output_dir="./nsmc_huggingface_train/kc-bert-komultitext-hp-search",
        seed=SHARED_SEED  # 공유 seed 사용
    )

    print(f"최적 하이퍼파라미터: {best_run_kcbert.hyperparameters}")
    print(f"최적 성능: {best_run_kcbert.objective}")

    print("\n" + "="*80)
    print("KcBERT 최종 학습")
    print("="*80)
    trainer_kcbert = train_kcbert(
        fine_tune_model_name, 
        best_run_kcbert, 
        output_dir="./nsmc_huggingface_train/kc-bert-komultitext-final", 
        train_dataset=komultitext_train,  # train 데이터
        test_dataset=komultitext_test,  # test 데이터
        seed=SHARED_SEED  # 공유 seed 사용
    )
