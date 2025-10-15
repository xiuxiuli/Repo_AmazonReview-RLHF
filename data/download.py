from datasets import load_dataset
import pandas as pd

def download_data():
    # "title", "review" | split: train
    # 1. download
    raw_data = load_dataset("sentence-transformers/amazon-reviews")
    print(raw_data)
    print("Sample record: ", raw_data["train"][0])

    # 2. conver to df
    df = raw_data["train"].to_pandas()
    print(f"loaded {len(df):,} rows.")

    # 3. keep and rename filed
    df = df[["review", "title"]]
    df.rename(columns={"review": "review", "title": "summary"})


    return raw_data

def process_data(data):
    print()

if __name__ == "__main__":
    raw_data = download_data()
    processed = process_data(raw_data)