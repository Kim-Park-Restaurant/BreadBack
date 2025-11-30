from transformers import pipeline
from data.preprocess_data import get_komultitext_dataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from ai.shared_config import SHARED_SEED
import numpy as np

import torch


def get_komultitext_test_20():
    """komultitext dataset에서 20% 추출 (학습과 동일한 seed 사용)"""
    komultitext_dataset = get_komultitext_dataset()
    
    # 8:2 분할 (학습과 동일한 seed 사용)
    komultitext_split = komultitext_dataset.train_test_split(test_size=0.2, seed=SHARED_SEED)
    test_dataset = komultitext_split["test"]
    
    print(f"사용 seed: {SHARED_SEED}")
    print(f"테스트 데이터셋 크기: {len(test_dataset)}")
    print(f"전체 komultitext 데이터셋 크기: {len(komultitext_dataset)}")
    
    return test_dataset


def test_ko_bert(model_path: str, test_texts: list, test_labels: list = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    print(f"모델 로드 중: {model_path}\n")
    
    # 체크포인트에서 모델 로드, 토크나이저는 원본 모델에서 로드
    classifier = pipeline(
        "text-classification",
        model=model_path,
        tokenizer="beomi/kcbert-base",
        device=device,
        trust_remote_code=True
    )

    print("=== KcBERT Pipeline 테스트 결과 ===")
    predictions = []
    predicted_labels = []
    
    for i, text in enumerate(test_texts):
        result = classifier(text)
        # pipeline 결과는 [{'label': 'LABEL_0', 'score': 0.99}] 형태
        label = int(result[0]['label'].split('_')[1])  # LABEL_0 -> 0, LABEL_1 -> 1
        score = result[0]['score']
        predicted_labels.append(label)
        predictions.append({
            'text': text,
            'predicted_label': label,
            'score': score,
            'true_label': test_labels[i] if test_labels else None
        })
    
    # 정확도 계산 (라벨이 있는 경우)
    if test_labels:
        accuracy = accuracy_score(test_labels, predicted_labels)
        f1 = f1_score(test_labels, predicted_labels, average='weighted')
        recall = recall_score(test_labels, predicted_labels, average='weighted')
        precision = precision_score(test_labels, predicted_labels, average='weighted')
        
        print(f"\n=== KcBERT 평가 지표 ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
    
    return predictions


def test_ko_electra(model_path: str, test_texts: list, test_labels: list = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    print(f"모델 로드 중: {model_path}\n")
    
    # 체크포인트에서 모델 로드, 토크나이저는 원본 모델에서 로드
    classifier = pipeline(
        "text-classification",
        model=model_path,
        tokenizer="monologg/koelectra-base-v3-discriminator",
        device=device,
        trust_remote_code=True
    )

    print("=== KoELECTRA Pipeline 테스트 결과 ===")
    predictions = []
    predicted_labels = []
    
    for i, text in enumerate(test_texts):
        result = classifier(text)
        # pipeline 결과는 [{'label': 'LABEL_0', 'score': 0.99}] 형태
        label = int(result[0]['label'].split('_')[1])  # LABEL_0 -> 0, LABEL_1 -> 1
        score = result[0]['score']
        predicted_labels.append(label)
        predictions.append({
            'text': text,
            'predicted_label': label,
            'score': score,
            'true_label': test_labels[i] if test_labels else None
        })
    
    # 정확도 계산 (라벨이 있는 경우)
    if test_labels:
        accuracy = accuracy_score(test_labels, predicted_labels)
        f1 = f1_score(test_labels, predicted_labels, average='weighted')
        recall = recall_score(test_labels, predicted_labels, average='weighted')
        precision = precision_score(test_labels, predicted_labels, average='weighted')
        
        print(f"\n=== KoELECTRA 평가 지표 ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
    
    return predictions


def test_both_models_with_dataset(
    test_dataset,
    test_name: str,
    kcbert_model_path: str,
    ko_electra_model_path: str
):
    """두 모델 모두 테스트 (동일한 데이터 사용)"""
    # 텍스트와 라벨 추출
    test_texts = test_dataset["text"]
    test_labels = test_dataset["sentiment"]
    
    print(f"\n{'='*80}")
    print(f"테스트 시나리오: {test_name}")
    print(f"{'='*80}")
    print(f"총 {len(test_texts)}개의 테스트 샘플 사용")
    print(f"라벨 분포: {np.bincount(test_labels)}\n")
    
    print("="*80)
    print("KcBERT 모델 테스트")
    print("="*80)
    kcbert_results = test_ko_bert(kcbert_model_path, test_texts, test_labels)
    
    print("\n\n" + "="*80)
    print("KoELECTRA 모델 테스트")
    print("="*80)
    ko_electra_results = test_ko_electra(ko_electra_model_path, test_texts, test_labels)
    
    return {
        'test_name': test_name,
        'kcbert_results': kcbert_results,
        'ko_electra_results': ko_electra_results,
        'dataset_size': len(test_texts),
        'label_distribution': np.bincount(test_labels).tolist()
    }


def test_komultitext_20(
    kcbert_model_path: str = "./nsmc_huggingface_train/kc-bert-komultitext-final",
    ko_electra_model_path: str = "./nsmc_huggingface_train/ko-electra-komultitext-final"
):
    """komultitext 20%로 두 모델 검증 (학습과 동일한 seed 사용)"""
    # komultitext 20% 추출 (학습과 동일한 seed 사용)
    test_dataset = get_komultitext_test_20()
    
    # 두 모델 테스트
    result = test_both_models_with_dataset(
        test_dataset,
        "komultitext_20",
        kcbert_model_path,
        ko_electra_model_path
    )
    
    return result


if __name__ == "__main__":
    test_komultitext_20()
