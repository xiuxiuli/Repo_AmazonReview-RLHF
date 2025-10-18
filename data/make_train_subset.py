import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import tool
import json, random, os
from pathlib import Path

def partition_trainset(cfg):
    with open(cfg["source_file"], "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    random.seed(cfg["seed"])
    random.shuffle(data)

    output_dir = Path(cfg["out_dir"])
    subsets = sorted(cfg["subsets"], key=lambda x:int(x["size"]))

    for sub in subsets:
        size = sub["size"]
        out_path = output_dir / sub["out_file"]
        out_path.parent.mkdir(parents=True, exist_ok=True)

        subset = data[:size]
        with open(out_path, "w", encoding="utf-8") as f:
            for item in subset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"💾 Saved {len(subset):,} → {out_path}")

if __name__ == "__main__":
    path = "config/data_config.yaml"
    cfg = tool.load_yaml(path)
    partition_trainset(cfg["train_subset"])
