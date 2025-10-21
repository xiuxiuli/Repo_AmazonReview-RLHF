import sys, os, random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gzip, json
from utils import tool
from tqdm import tqdm
from pathlib import Path

def partition(cfg):
    # load config
    subCfg = cfg["partition"]
    root_dir = tool.get_root_dir(cfg)
    src_path = os.path.join(root_dir, subCfg["source_file"])

    split_ratio = subCfg["split_ratio"]
    
    # load cleaned_data
    open_func = gzip.open if src_path.endswith(".gz") else open
    with open_func(src_path, "rt", encoding="utf-8") as f:
        data = [json.loads(line) for line in tqdm(f, desc="Reading lines", unit="lines")]

    n_total = len(data)
    print(f"✅ Total loaded samples: {n_total:,}")

    # shuffle 
    random.seed(subCfg["seed"])
    random.shuffle(data)

    # compute split size
    train_ratio = split_ratio.get("train", 0.8)
    val_ratio = split_ratio.get("val", 0.1)
    test_ratio = split_ratio.get("test", 0.1)

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_split = data[: n_train]
    val_split = data[n_train : (n_train + n_val)]
    test_split = data[(n_train + n_val):]

    print(f"📊 Split sizes:")
    print(f"   Train: {len(train_split):,}  ({train_ratio*100:.1f}%)")
    print(f"   Val:   {len(val_split):,}  ({val_ratio*100:.1f}%)")
    print(f"   Test:  {len(test_split):,}  ({test_ratio*100:.1f}%)")

    # save splits
    os.makedirs(os.path.join(root_dir, subCfg["output_subdir"]), exist_ok=True)
    out_files = subCfg["out_files"]

    save_details = {
        "train": (train_split, os.path.join(root_dir, subCfg['output_subdir'], out_files["train"])),
        "val": (val_split, os.path.join(root_dir, subCfg['output_subdir'], out_files["val"])),
        "test": (test_split, os.path.join(root_dir, subCfg['output_subdir'], out_files["test"]))
    }

    for name, (subset, path) in save_details.items():
        print(f"💾 Saving {name} set → {path}")
        save_jsonl(subset, path) 

# helper
def save_jsonl(data, path):
    isCompressed = path.endswith(".gz")

    open_func = gzip.open if isCompressed else open
    with open_func(path, "wt", encoding="utf-8") as f:
        for sample in tqdm(data, desc=f"Writing {os.path.basename(path)}", unit=" lines"):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    path = "config/data_config.yaml"
    cfg = tool.load_yaml(path)
    partition(cfg)