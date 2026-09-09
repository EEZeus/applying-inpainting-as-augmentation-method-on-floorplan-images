#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.augmentation.inputs import load_inputs
from src.augmentation.review import apply_review
from src.utils.config import load_yaml, resolve_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/augmentation_b.yaml")
    parser.add_argument("--review", default="outputs/augmentation_b/review.csv")
    args = parser.parse_args()
    config = load_yaml(resolve_path(args.config))
    coco, _, train, image_dir = load_inputs(config)
    result = apply_review(resolve_path(config["output"]["dataset_dir"]), resolve_path(args.review), coco, train, image_dir)
    print(f"Approved training dataset: {len(result['images'])} images, {len(result['annotations'])} annotations")
