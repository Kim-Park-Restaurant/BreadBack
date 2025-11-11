from transformers import pipeline

import torch


test_texts = [
    "와 진짜 이거 미쳤다 ㅋㅋㅋㅋ 완전 대박!"
]

def test_ko_bert(model_path: str = "./models/finetuned-kcbert/checkpoint-5637"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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
    
    for text in test_texts:
        result = classifier(text)
        print(f"\n입력: {text}")
        print(f"결과: {result}")


def test_ko_electra(model_path: str = "./models/finetuned-bert/checkpoint-6124"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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
    
    for text in test_texts:
        result = classifier(text)
        print(f"\n입력: {text}")
        print(f"결과: {result}")
    

def test_both_models():
    """두 모델 모두 테스트"""
    print("="*60)
    print("KoBERT 모델 테스트")
    print("="*60)
    test_ko_bert()
    
    print("\n\n" + "="*60)
    print("KoELECTRA 모델 테스트")
    print("="*60)
    test_ko_electra()


if __name__ == "__main__":
    test_both_models()
