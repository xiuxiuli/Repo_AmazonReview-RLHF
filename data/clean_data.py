def clean_text(text: str):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", " ", text)                 # 去除HTML标签
    text = re.sub(r"http\S+|www\S+", " ", text)        # 去除URL
    text = re.sub(r"\s+", " ", text)                   # 多余空格
    text = re.sub(r"[^\x00-\x7F]+", " ", text)         # 非打印字符
    text = text.strip()                                # 去除首尾空格
    return text

def clean(json){

    # 2. conver to df
    df = raw_data.to_pandas()
    print(f"Loaded {len(df):,} rows.")

    # 3. rename fileds
    # df = df[["review", "title"]]
    df = df.rename(columns={"review": "review", "title": "summary"})

    # 4. remove too long/too short content
    df = df[df["review"].str.len() > 50]
    df = df[df["summary"].str.len().between(10, 200)]

    #print(df["summary"].str.len().describe())

    # 4. remove non-info summary
    bad_summaries = ["good", "ok", "nice", "great", "bad", "fine", "yes"]
    df = df[~df["summary"].str.lower().isin(bad_summaries)]

    # 4. remove other
    df["review"] = df["review"].apply(clean_text)
    df["summary"] = df["summary"].apply(clean_text)

    print(f"✅ After cleaning: {len(df):,} samples remaining.")
    # 统计
    print("\n📊 Dataset overview:")
    print(f"Total samples after cleaning: {len(df):,}")
    print(f"Average review length: {df['review'].str.len().mean():.1f}")
    print(f"Average summary length: {df['summary'].str.len().mean():.1f}")

    # 🔹 字符长度分布
    print("\n📈 Review length stats:")
    print(df["review"].str.len().describe())

    print("\n📈 Summary length stats:")
    print(df["summary"].str.len().describe())

    if save_jsonl:
        os.makedirs("data/processed", exist_ok=True)
        output_path = "data/processed/sft_train_cleaned.jsonl"
        df.to_json(output_path, orient="records", lines=True, force_ascii=False)
        print(f"✅ Saved cleaned dataset to: {output_path}")
    return df
}