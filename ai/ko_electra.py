from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from data.preprocess_data import get_nsmc_dataset, get_komultitext_dataset
from ai.kc_bert import get_tokenized_datasets, hp_space

import torch
import numpy as np
import os


os.environ["WANDB_PROJECT"] = "finetuning"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


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


def search_best_hyperparameters(
    model_name,
    num_epochs,
    batch_size,
    learning_rate,
    dataset,
    output_dir
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = get_tokenized_datasets(tokenizer, dataset)

    def model_init():
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, trust_remote_code=True)
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
        metric_for_best_model="eval_accuracy",
        report_to="wandb"  # 하이퍼파라미터 검색 과정도 wandb에 기록
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


def train_ko_electra(model_name, best_run, output_dir, dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = get_tokenized_datasets(tokenizer, dataset)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, trust_remote_code=True)
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
        report_to="wandb"  # 최종 학습 과정도 wandb에 기록
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

    eval_results = trainer.evaluate()
    print(f"Validation 평가 결과: {eval_results}")

    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print(f"Test 평가 결과: {test_results}")


if __name__ == "__main__":
    model_name = "monologg/koelectra-base-v3-discriminator"
    default_epochs = 3
    default_batch_size = 8
    default_learning_rate = 1e-5

    nsmc_dataset = get_nsmc_dataset()
    komultitext_dataset = get_komultitext_dataset()

    # best_run_nsmc = search_best_hyperparameters(
    #     model_name=model_name,
    #     num_epochs=default_epochs,
    #     batch_size=default_batch_size,
    #     learning_rate=default_learning_rate,
    #     dataset=komultitext_dataset,
    #     output_dir="./huggingface_nsmc_train/ko-electra-komultitext-hp-search"
    # )

    # train_ko_electra(model_name, best_run_nsmc, output_dir="./huggingface_nsmc_train/ko-electra-komultitext-final", dataset=nsmc_dataset)

    model_name = "./huggingface_nsmc_train/ko-electra-komultitext-final/checkpoint-525"
    best_run_komultitext = search_best_hyperparameters(
        model_name=model_name,
        num_epochs=default_epochs,
        batch_size=default_batch_size,
        learning_rate=default_learning_rate,
        dataset=nsmc_dataset,
        output_dir="./huggingface_nsmc_train/ko-electra-nsmc-hp-search"
    )

    train_ko_electra(model_name, best_run_komultitext, output_dir="./huggingface_nsmc_train/ko-electra-nsmc-final", dataset=nsmc_dataset)


