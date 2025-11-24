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


def get_tokenized_datasets(tokenizer, dataset):
    tokenized_datasets = dataset.map(
        preprocess_function, 
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": 128},
        remove_columns=dataset.column_names
    )

    train_val_split = tokenized_datasets.train_test_split(test_size=0.3, seed=42)
    val_test_split = train_val_split["test"].train_test_split(test_size=0.5, seed=42)

    tokenized_datasets = DatasetDict({
        "train": train_val_split["train"],
        "validation": val_test_split["train"],
        "test": val_test_split["test"]
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
    dataset,
    output_dir
):
    tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = get_tokenized_datasets(tokenizer, dataset)


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
        eval_dataset=tokenized_datasets["validation"],
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


def train_kcbert(model_name, best_run, output_dir, dataset):
    # 로컬 경로인 경우 절대 경로로 변환 (상대 경로 문제 해결)
    if os.path.exists(model_name) and os.path.isdir(model_name):
        model_name = os.path.abspath(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = get_tokenized_datasets(tokenizer, dataset)

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
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )   
    trainer.train()

    trainer.save_model()

    # validation 데이터셋 평가
    eval_results = trainer.evaluate()
    print(f"Validation 평가 결과: {eval_results}")

    # test 데이터셋 평가
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print(f"Test 평가 결과: {test_results}")

    return trainer


if __name__ == "__main__":
    default_model_name = "beomi/kcbert-base"
    nsmc_dataset = get_nsmc_dataset()
    komultitext_dataset = get_komultitext_dataset()

    default_num_epochs = 3
    default_batch_size = 8
    default_learning_rate = 2e-5

    # best_run_nsmc = search_best_hyperparameters(
    #     model_name=default_model_name,
    #     num_epochs=default_num_epochs,
    #     batch_size=default_batch_size,
    #     learning_rate=default_learning_rate,
    #     dataset=komultitext_dataset,
    #     output_dir="./huggingface_nsmc_train/kc-bert-komultitext-hp-search"
    # )

    # print(f"최적 하이퍼파라미터: {best_run_nsmc.hyperparameters}")
    # print(f"최적 성능: {best_run_nsmc.objective}")
    # # TODO kcbert train
    # trainer = train_kcbert(default_model_name, best_run_nsmc, output_dir="./huggingface_nsmc_train/kc-bert-komultitext-final", dataset=komultitext_dataset)
    
    fine_tune_model_name = "./huggingface_nsmc_train/kc-bert-komultitext-final/checkpoint-194"
    best_run_komultitext = search_best_hyperparameters(
        model_name=fine_tune_model_name,
        num_epochs=default_num_epochs,
        batch_size=default_batch_size,
        learning_rate=default_learning_rate,
        dataset=nsmc_dataset,
        output_dir="./huggingface_nsmc_train/kc-bert-nsmc-hp-search"
    )

    print(f"최적 하이퍼파라미터: {best_run_komultitext.hyperparameters}")
    print(f"최적 성능: {best_run_komultitext.objective}")

    trainer = train_kcbert(fine_tune_model_name, best_run_komultitext, output_dir="./huggingface_nsmc_train/kc-bert-nsmc-final", dataset=nsmc_dataset)
