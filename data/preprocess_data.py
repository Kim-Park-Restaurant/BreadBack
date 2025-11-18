from datasets import (
    Dataset, 
    DatasetDict, 
    load_dataset
)
from soynlp.normalizer import repeat_normalize

import pandas as pd
import re
import emoji


pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+')
url_pattern = re.compile(
    r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')


def clean_text(text):
    text = pattern.sub(' ', text)
    text = emoji.replace_emoji(text, replace='')
    text = url_pattern.sub('', text)
    text = text.strip()
    text = repeat_normalize(text, num_repeats=2)
    return text


def get_nsmc_dataset():
    nsmc = load_dataset("Blpeng/nsmc")

    df_nsmc = pd.concat([
        pd.DataFrame(nsmc["train"]),
        pd.DataFrame(nsmc["test"])
    ])

    df_nsmc = df_nsmc.rename(columns={"document": "text", "label": "sentiment"})
    df_nsmc = df_nsmc.drop(["Unnamed: 0", "id"], axis = 1)

    df_nsmc = df_nsmc.sample(frac = 0.1, random_state = 42)
    df_nsmc = Dataset.from_pandas(df_nsmc)
    return df_nsmc


def get_komultitext_dataset():
    df_hug_dc = load_dataset("Dasool/KoMultiText")["train"].to_pandas()
    df_hug_dc.rename(columns = {"comment": "text", "preference": "sentiment"}, inplace = True)

    df_hug_dc.drop(columns = ['profanity', 'gender', 'politics', 'nation',
       'race', 'region', 'generation', 'social_hierarchy', 'appearance',
       'others'], axis = 1, inplace = True)
    df_hug_dc.drop(
        df_hug_dc[df_hug_dc["sentiment"].isin([1, 2])].index,
        inplace=True
    )

    df_hug_dc["sentiment"] = df_hug_dc["sentiment"].replace({3 : 1, 4 : 1, 0 : 0})
    df_hug_dc = df_hug_dc[~df_hug_dc['text'].apply(lambda x: bool(re.search(r'[A-Za-z]', x)))]

    df_pref_0 = df_hug_dc[df_hug_dc["sentiment"] == 0]
    df_pref_1 = df_hug_dc[df_hug_dc["sentiment"] == 1]
    
    pref_1_count = len(df_pref_1)
    pref_0_count = len(df_pref_0)
    
    if pref_0_count > pref_1_count:
        df_pref_0_sampled = df_pref_0.sample(n=pref_1_count, random_state=42)
        df_hug_dc = pd.concat([df_pref_0_sampled, df_pref_1], ignore_index=True)

    print(len(df_hug_dc[df_hug_dc["sentiment"] == 0]))
    print(len(df_hug_dc[df_hug_dc["sentiment"] == 1]))
    
    df_hug_dc["text"] = df_hug_dc["text"].apply(clean_text)

    df_hug_dc = Dataset.from_pandas(df_hug_dc)

    return df_hug_dc



if __name__ == "__main__":
    df_nsmc = get_nsmc_dataset()
    df_hug_dc = get_komultitext_dataset()

    print(df_nsmc)
    print(df_nsmc.shape)

    print(df_hug_dc)
    print(df_hug_dc.shape)