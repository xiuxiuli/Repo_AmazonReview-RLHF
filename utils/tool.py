import yaml
from pathlib import Path
import os

def load_yaml(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config

def get_root_dir(cfg): 
    if "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ:
        root_dir = cfg["root"]["colab"]
    else: root_dir = cfg["root"]["local"]
    print(f"📁 Root dir set to: {root_dir}")
    return root_dir

def get_dir_path(cfg, subCfg):
    root_dir = get_root_dir(cfg)

    output_dir = os.path.join(root_dir, subCfg['output_subdir'])
    output_path= os.path.join(root_dir, subCfg['output_subdir'], subCfg["output_file"])
    
    os.makedirs(output_dir , exist_ok=True)

    return output_dir, output_path
    
