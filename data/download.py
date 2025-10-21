import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset
import os, random, json
from utils import tool
import gzip

def download_stream_subset(cfg):
    print("🔹 Loading dataset from HuggingFace ...")

    subCfg = cfg["dataset"] 
    output_dir, output_path = tool.get_dir_path(cfg, subCfg)

    # "title", "review" | split: train
    dataset_name = subCfg["name"]
    sample_size = subCfg["size"]
    num_log_steps = subCfg["num_log_steps"]

    random.seed(subCfg["seed"])
    
    # 1. download full 87.5M
    stream = load_dataset(dataset_name, split=subCfg["split"], streaming=True)

    # 2. random sampling 40k
    reservoir = []
    for i, example in enumerate(stream):
        if i < sample_size:
            reservoir.append(example)
        else:
            j = random.randint(0, i)
            if j < sample_size:
                reservoir[j] = example
        if (i + 1) % num_log_steps == 0:
            print(f"📦 Processed {i+1:,} rows ... reservoir size: {len(reservoir)}")

    print(f"✅ Finished sampling {len(reservoir):,} samples from {i+1:,} total records.")
    
    # 3. save
    with gzip.open(output_path, "wt", encoding='utf-8') as f:
        for item in reservoir:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"💾 Saved random subset to: {output_path}")

if __name__ == "__main__":
    path = "config/data_config.yaml"
    cfg = tool.load_yaml(path, True)
    download_stream_subset(cfg)