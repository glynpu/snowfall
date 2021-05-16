from pathlib import Path
import argparse
import yaml

def _load_espnet_model_config(config_file):
    config_file = Path(config_file)
    with config_file.open("r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    return argparse.Namespace(**args)
