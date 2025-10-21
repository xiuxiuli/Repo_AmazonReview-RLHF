import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import tool
import json, random, os
from pathlib import Path

def partition_trainset(cfg):
    subCfg = cfg["train_subset"]

    root_dir = tool.get_root_dir(cfg)
    src_path = os.path.join(root_dir, subCfg["source_file"])

    with open(src_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    random.seed(subCfg["seed"])
    random.shuffle(data)

    subsets = sorted(subCfg["subsets"], key=lambda x:int(x["size"]))

    output_dir = os.path.join(root_dir, subCfg["output_subdir"])
    os.makedirs(output_dir, exist_ok=True)

    for sub in subsets:
        size = sub["size"]
        out_path = os.path.join(output_dir, sub["out_file"])

        subset = data[:size]
        with open(out_path, "w", encoding="utf-8") as f:
            for item in subset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"💾 Saved {len(subset):,} → {out_path}")

if __name__ == "__main__":
    path = "config/data_config.yaml"
    cfg = tool.load_yaml(path)
    partition_trainset(cfg)
