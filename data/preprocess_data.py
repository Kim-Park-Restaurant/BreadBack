import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def sample_data_by_ratio(df: pd.DataFrame, sample_ratio: float = 0.1, random_state: int = 42):
    if sample_ratio <= 0 or sample_ratio > 1:
        raise ValueError("sample_ratio는 0과 1 사이의 값이어야 합니다.")
    
    if sample_ratio == 1.0:
        return df
    
    # stratify를 위해 sentiment 비율 유지하며 샘플링
    sampled_df = df.groupby('sentiment', group_keys=False).apply(
        lambda x: x.sample(frac=sample_ratio, random_state=random_state)
    ).reset_index(drop=True)
    
    return sampled_df


def create_datadict_from_csv(
    csv_path: str, 
    train_ratio: float = 0.8,
    sample_ratio: float = 1.0,
    random_state: int = 42
):
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # 특정 비율만 추출
    if sample_ratio < 1.0:
        print(f"전체 데이터: {len(df)}개")
        df = sample_data_by_ratio(df, sample_ratio=sample_ratio, random_state=random_state)
        print(f"추출된 데이터: {len(df)}개 ({sample_ratio*100:.1f}%)")
    
    train_df, test_df = train_test_split(
        df, 
        test_size=1 - train_ratio, 
        random_state=random_state,
        stratify=df['sentiment']
    )
    
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    return dataset_dict


if __name__ == "__main__":
    # final_df.csv 파일 경로
    csv_path = "./data/final_df.csv"
    
    # 데이터의 10%만 추출
    print("\n" + "="*50)
    dataset_dict_sample = create_datadict_from_csv(
        csv_path, 
        train_ratio=0.8, 
        sample_ratio=0.1
    )
    print("10% 샘플 데이터:")
    print(dataset_dict_sample)
