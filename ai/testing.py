from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch


def test_kobert_with_pipeline(model_name: str = "monologg/kobert"):
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    try:
        classifier = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=device,
            trust_remote_code=True
        )
        
        # 테스트 문장들
        test_texts = [
            "정말 최고의 하루였어요",
            "오늘은 기분이 좋네요",
            "이건 좀 아닌것 같아요",
            "정말 감동적인 영화였습니다.",
            "별로 재미없었어요."
        ]
        
        print("=== Pipeline 테스트 결과 ===")
        predictions = []
        
        for text in test_texts:
            result = classifier(text)
            print(f"\n입력: {text}")
            print(f"결과: {result}")
            
            # 예측 결과 저장 (긍정/부정 판단)
            if isinstance(result, list):
                result = result[0]
            
            label = result.get('label', '')
            score = result.get('score', 0)
            
            # LABEL_0: 부정, LABEL_1: 긍정
            if label in ['LABEL_1', 'POSITIVE', '긍정', '1', 1] or (isinstance(label, str) and ('label_1' in label.lower() or 'positive' in label.lower())):
                predictions.append('긍정')
            elif label in ['LABEL_0', 'NEGATIVE', '부정', '0', 0] or (isinstance(label, str) and ('label_0' in label.lower() or 'negative' in label.lower())):
                predictions.append('부정')
            else:
                # 라벨이 명확하지 않으면 점수로 판단
                predictions.append('긍정' if score > 0.5 else '부정')
        
        # 긍정/부정 비율 계산
        print("\n" + "="*50)
        print("=== 예측 비율 통계 ===")
        total = len(predictions)
        positive_count = predictions.count('긍정')
        negative_count = predictions.count('부정')
        
        positive_ratio = (positive_count / total) * 100 if total > 0 else 0
        negative_ratio = (negative_count / total) * 100 if total > 0 else 0
        
        print(f"전체 예측: {total}개")
        print(f"긍정: {positive_count}개 ({positive_ratio:.1f}%)")
        print(f"부정: {negative_count}개 ({negative_ratio:.1f}%)")
            
    except Exception as e:
        print(f"Pipeline 사용 실패: {e}")
        print("\n토크나이저와 모델을 직접 사용합니다...")
        test_kobert_direct(model_name)


def test_kobert_direct(model_name: str = "monologg/kobert"):
    """토크나이저와 모델을 직접 사용한 테스트"""
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"사용 장치: {device}")
    print(f"모델 로드 중: {model_name}\n")
    
    # 토크나이저와 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        trust_remote_code=True
    )
    model.to(device)
    model.eval()
    
    # 테스트 문장들
    test_texts = [
        "정말 최고의 하루였어요",
        "오늘은 기분이 좋네요",
        "이건 좀 아닌것 같아요",
        "정말 감동적인 영화였습니다.",
        "별로 재미없었어요."
    ]
    
    print("=== 직접 테스트 결과 ===")
    predictions = []
    
    for text in test_texts:
        # 토크나이징
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 예측
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # 결과 출력
        label = "긍정" if probs[0][1] > probs[0][0] else "부정"
        confidence = float(probs[0][1] if label == "긍정" else probs[0][0])
        
        print(f"\n입력: {text}")
        print(f"예측: {label} (신뢰도: {confidence:.4f})")
        print(f"확률: 부정={probs[0][0]:.4f}, 긍정={probs[0][1]:.4f}")
        
        predictions.append(label)
    
    # 긍정/부정 비율 계산
    print("\n" + "="*50)
    print("=== 예측 비율 통계 ===")
    total = len(predictions)
    positive_count = predictions.count('긍정')
    negative_count = predictions.count('부정')
    
    positive_ratio = (positive_count / total) * 100 if total > 0 else 0
    negative_ratio = (negative_count / total) * 100 if total > 0 else 0
    
    print(f"전체 예측: {total}개")
    print(f"긍정: {positive_count}개 ({positive_ratio:.1f}%)")
    print(f"부정: {negative_count}개 ({negative_ratio:.1f}%)")


if __name__ == "__main__":
    test_kobert_with_pipeline()

