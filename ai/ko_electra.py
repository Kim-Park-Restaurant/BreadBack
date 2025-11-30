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
    train_dataset,
    test_dataset,
    output_dir,
    seed=42
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = get_tokenized_datasets(tokenizer, train_dataset, test_dataset=test_dataset, seed=seed)

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


def train_ko_electra(model_name, best_run, output_dir, train_dataset, test_dataset, seed=42):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = get_tokenized_datasets(tokenizer, train_dataset, test_dataset=test_dataset, seed=seed)
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
        eval_dataset=tokenized_datasets["test"],  # test를 validation으로 사용
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model()

    # test 데이터셋 평가
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print(f"Test 평가 결과: {test_results}")


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
    
    default_epochs = 3
    default_batch_size = 8
    default_learning_rate = 1e-5
    
    # ko_electra 학습
    fine_tune_model_name = "./nsmc_huggingface_train/ko-electra-nsmc-final/checkpoint-700"
    
    print("\n" + "="*80)
    print("KoELECTRA 하이퍼파라미터 검색")
    print("="*80)
    best_run_ko_electra = search_best_hyperparameters(
        model_name=fine_tune_model_name,
        num_epochs=default_epochs,
        batch_size=default_batch_size,
        learning_rate=default_learning_rate,
        train_dataset=komultitext_train,  # train 데이터
        test_dataset=komultitext_test,  # test 데이터
        output_dir="./nsmc_huggingface_train/ko-electra-komultitext-hp-search",
        seed=SHARED_SEED  # 공유 seed 사용
    )

    print(f"최적 하이퍼파라미터: {best_run_ko_electra.hyperparameters}")
    print(f"최적 성능: {best_run_ko_electra.objective}")

    print("\n" + "="*80)
    print("KoELECTRA 최종 학습")
    print("="*80)
    train_ko_electra(
        fine_tune_model_name, 
        best_run_ko_electra, 
        output_dir="./nsmc_huggingface_train/ko-electra-komultitext-final", 
        train_dataset=komultitext_train,  # train 데이터
        test_dataset=komultitext_test,  # test 데이터
        seed=SHARED_SEED  # 공유 seed 사용
    )


