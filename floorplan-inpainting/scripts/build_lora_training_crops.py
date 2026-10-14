#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.augmentation.inputs import load_inputs
from src.lora_inpainting.data import build_training_crops
from src.utils.config import load_yaml, resolve_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT / "configs/lora_training.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    coco, _, train, image_dir = load_inputs(config)
    print(
        json.dumps(
            build_training_crops(
                config, coco, train, image_dir, resolve_path(config["dataset"]["output_dir"])
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
