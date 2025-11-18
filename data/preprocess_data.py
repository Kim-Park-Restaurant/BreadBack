import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def get_nsmc_dataset():
    nsmc = load_dataset("Blpeng/nsmc")

    df_nsmc = pd.concat([
        pd.DataFrame(nsmc["train"]),
        pd.DataFrame(nsmc["test"])
    ])

    df_nsmc = df_nsmc.rename(columns={"document": "text", "label": "sentiment"})
    df_nsmc = df_nsmc.drop(["Unnamed: 0", "id"], axis = 1)

    return df_nsmc


def get_komultitext_dataset():
    df_hug_dc = load_dataset("Dasool/KoMultiText")["train"].to_pandas()
    df_hug_dc.rename(columns = {"comment": "text", "preference": "sentiment"}, inplace = True)

    df_hug_dc.drop(columns = ['profanity', 'gender', 'politics', 'nation',
       'race', 'region', 'generation', 'social_hierarchy', 'appearance',
       'others'], axis = 1, inplace = True)
    df_hug_dc.drop(
        df_hug_dc[df_hug_dc["preference"].isin([1, 2])].index,
        inplace=True
    )

    df_hug_dc["preference"] = df_hug_dc["preference"].replace({3 : 1, 4 : 1, 0 : 0})
    df_hug_dc = df_hug_dc[~df_hug_dc['comment'].apply(lambda x: bool(re.search(r'[A-Za-z]', x)))]

    df_pref_0 = df_hug_dc[df_hug_dc["preference"] == 0]
    df_pref_1 = df_hug_dc[df_hug_dc["preference"] == 1]
    
    pref_1_count = len(df_pref_1)
    pref_0_count = len(df_pref_0)
    
    if pref_0_count > pref_1_count:
        # preference 0인 샘플을 1의 개수만큼 랜덤 샘플링
        df_pref_0_sampled = df_pref_0.sample(n=pref_1_count, random_state=42)
        df_hug_dc = pd.concat([df_pref_0_sampled, df_pref_1], ignore_index=True)

    print(len(df_hug_dc[df_hug_dc["preference"] == 0]))
    print(len(df_hug_dc[df_hug_dc["preference"] == 1]))
    
    return df_hug_dc


def sample_data_by_ratio(df: pd.DataFrame, sample_ratio: float = 0.1, random_state: int = 42):
    if sample_ratio <= 0 or sample_ratio > 1:
        raise ValueError("sample_ratio는 0과 1 사이의 값이어야 합니다.")
    
    if sample_ratio == 1.0:
        return df
    
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
