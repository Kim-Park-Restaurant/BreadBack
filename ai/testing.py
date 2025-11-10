from transformers import pipeline

import torch


test_texts = [
    "와 진짜 이거 미쳤다 ㅋㅋㅋㅋ 완전 대박!",
    "아니 진심으로 짜증 나 죽겠네 ㅠㅠ 뭐야 이거",
    "오늘 본 영화 개쩔었음… 눈물 펑펑 ㅠㅠ",
    "헐 왜 이렇게 별로야… 시간 완전 낭비했음",
    "맛집 후기 보고 갔는데 개맛있음 ㅋㅋㅋ 완전 감동!"
]

def test_ko_bert(model_path: str = "./models/finetuned-kcbert/checkpoint-684"):
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


def test_ko_electra(model_path: str = "./models/finetuned-bert/checkpoint-3062"):
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
