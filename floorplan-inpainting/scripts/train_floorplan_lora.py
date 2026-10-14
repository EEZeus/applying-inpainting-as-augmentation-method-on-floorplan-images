#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.lora_inpainting.train import train_lora
from src.utils.config import load_yaml, resolve_path
from src.utils.reproducibility import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT / "configs/lora_training.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    seed_everything(config["seed"])
    print(
        json.dumps(
            train_lora(
                config,
                resolve_path(config["dataset"]["manifest"]),
                resolve_path(config["model"]["output_dir"]),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
