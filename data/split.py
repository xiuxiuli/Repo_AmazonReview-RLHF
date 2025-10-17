import sys, os, random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gzip, json
from utils import tool
from tqdm import tqdm
from pathlib import Path

def partition(cfg):
    # load config
    src_file = cfg['source_file']
    split_ratio = cfg["split_ratio"]
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    
    # load cleaned_data
    open_func = gzip.open if src_file.endswith(".gz") else open
    with open_func(src_file, "rt", encoding="utf-8") as f:
        data = [json.loads(line) for line in tqdm(f, desc="Reading lines", unit="lines")]

    n_total = len(data)
    print(f"✅ Total loaded samples: {n_total:,}")

    # shuffle 
    random.seed(cfg["seed"])
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
    out_files = cfg["out_files"]
    out_dir = cfg["output_dir"]

    save_details = {
        "train": (train_split, os.path.join(out_dir, out_files["train"])),
        "val": (val_split, os.path.join(out_dir, out_files["val"])),
        "test": (test_split, os.path.join(out_dir, out_files["test"]))
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
    partition(cfg["partition"])