import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gzip, re, json, statistics
from pathlib import Path
from utils import tool

def clean_text(text: str):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", " ", text)                 # remove HTML tag
    text = re.sub(r"http\S+|www\S+", " ", text)        # remove URL
    text = re.sub(r"\s+", " ", text)                   # remove extra space
    text = re.sub(r"[^\x00-\x7F]+", " ", text)         # remove non-ascii
    text = text.strip()                                # trim
    return text

def clean_record(record: dict) -> dict:
    """
    {'review', 'title'} => {'review', 'summary'}。
    """
    review = clean_text(record.get("review", ""))
    summary = clean_text(record.get("summary", record.get("title", "")))

    return {"review": review, "summary":summary}

def clean(config):
    subCfg = cfg["cleaning"]
    root_dir = tool.get_root_dir(config)
    output_dir, output_path = tool.get_dir_path(cfg, subCfg)

    source_file = subCfg["source_file"]

    min_review_len = subCfg["min_review_length"]
    min_summary_len = subCfg["min_summary_length"]
    max_summary_len = subCfg["max_summary_length"]
    bad_summaries = set(subCfg.get("bad_summaries", []))

    count_in, count_out = 0, 0
    review_lens, summary_lens = [], []

    drop_stats = {
                    "json_error": 0,
                    "empty_review_after_clean": 0,
                    "empty_summary_after_clean": 0,
                    "short_review": 0,
                    "summary_len_out_of_range": 0,
                    "bad_summary": 0
                }
    
    src_path = os.path.join(root_dir, source_file)

    with gzip.open(src_path , "rt", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            count_in += 1

            try: 
                record = json.loads(line)
            except json.JSONDecodeError:
                drop_stats["json_error"] += 1
                continue

            cleaned = clean_record(record)
            
            review = cleaned["review"]
            summary = cleaned["summary"]
            
            # if empty
            if not review:
                drop_stats["empty_review_after_clean"] += 1
                continue
            if not summary:
                drop_stats["empty_summary_after_clean"] += 1
                continue

            review_len = len(review.split())
            summary_len = len(summary.split())
            
            # len validate
            if len(review.split()) < min_review_len:
                drop_stats["short_review"] += 1
                continue
            if not (min_summary_len <= len(summary.split()) <= max_summary_len):
                drop_stats["summary_len_out_of_range"] += 1
                continue
            
            if summary.strip().lower() in bad_summaries:
                drop_stats["bad_summary"] += 1
                continue

            # ✅ keep track of lengths for statistics
            review_lens.append(review_len)
            summary_lens.append(summary_len)
            
            # validated success, write in
            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            count_out += 1

    # statistics - cleaned data
    kept_ratio = (count_out / count_in) if count_in else 0.0
    print("\n✅ Cleaning complete")
    print(f"📦 Input lines: {count_in:,}")
    print(f"✅ Kept lines : {count_out:,}  (keep ratio: {kept_ratio:.2%})")
    print("🗑️ Dropped by reason:")
    for k, v in drop_stats.items():
        print(f"   - {k}: {v:,}")

    # ✅ data len statistics
    if review_lens and summary_lens:
        print("\n📊 Length statistics (kept samples only):")
        print(f"📝 Avg review length:  {statistics.mean(review_lens):.1f} words")
        print(f"📝 Avg summary length: {statistics.mean(summary_lens):.1f} words")
        print(f"📈 Median review length:  {statistics.median(review_lens):.1f}")
        print(f"📈 Median summary length: {statistics.median(summary_lens):.1f}")
        print(f"📊 Review length range:  {min(review_lens)} ~ {max(review_lens)}")
        print(f"📊 Summary length range: {min(summary_lens)} ~ {max(summary_lens)}")
    
    print(f"\n💾 Saved cleaned data to: {output_path}")

if __name__ == "__main__":
    cfg = tool.load_yaml("config/data_config.yaml")
    clean(cfg)