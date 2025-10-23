from utils import tool
from pathlib import Path
from train import sft_trainer, dpo_trainer, rlhf_trainer
import os

def mark_done(path):
    Path(path).joinpath(".done").touch()

def is_done(path):
    return Path(path).joinpath(".done").exists()

def main(pipe_cfg):
    stages = pipe_cfg["stages"]
    root_dir = tool.get_root_dir(pipe_cfg)
    
    for stage in stages:
        name = stage["name"]

        output_dir = os.path.join(root_dir, stage["output_dir"])

        cfg_path = stage["config"]
        cfg = tool.load_yaml(cfg_path)

        if is_done(output_dir):
            print(f"✅ Stage: {name.upper()} is completed - SKIP")
            continue
            
        print(f"\n 🚀 Start stage - {name.upper()}")

        if name.startswith("sft"):
            sft_trainer.run(cfg)
        elif name == "dpo":
            dpo_trainer.run(cfg)
        elif name == "rlhf":
            rlhf_trainer.run(cfg)

        mark_done(output_dir)

        print(f"✅ {name.upper()} Complete！\n")

if __name__ == "__main__":
    path = "config/pipeline.yaml"    
    pipe_cfg = tool.load_yaml(path, True)
    main(pipe_cfg)