import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_jsonl(filepath):
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return pd.DataFrame(records)

data_dir = Path("data/billsum_v4_1")
train_df = load_jsonl(data_dir / "us_train_data_final_OFFICIAL.jsonl")
test_df  = load_jsonl(data_dir / "us_test_data_final_OFFICIAL.jsonl")
ca_df    = load_jsonl(data_dir / "ca_test_data_final_OFFICIAL.jsonl")

def basic_stats(df, name):
    print(f"\n===== {name} =====")
    print(f"样本数量: {len(df)}")
    print(f"字段列表: {df.columns.tolist()}")
    print(f"缺失值:\n{df.isnull().sum()}")
    print(f"text 长度统计 (chars):\n{df['text'].str.len().describe()}")
    print(f"summary 长度统计 (chars):\n{df['summary'].str.len().describe()}")

basic_stats(train_df, "US Train")
basic_stats(test_df,  "US Test")
basic_stats(ca_df,    "CA Test")