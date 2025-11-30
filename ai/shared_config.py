"""
두 모델이 동일한 데이터를 사용하도록 공유 seed 관리
"""
import random

# 공유 seed 생성 (한 번만 생성)
SHARED_SEED = random.randint(0, 2**31 - 1)

