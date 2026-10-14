#!/usr/bin/env python3
"""Compose accepted Strategy C copy-paste edits into multi-replacement plans."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augmentation.inputs import load_inputs
from src.augmentation.lora_multiroom import build_multiroom_dataset
from src.utils.config import load_yaml, resolve_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT / "configs/augmentation_c.yaml")
    parser.add_argument("--replacement-percentage", type=float)
    parser.add_argument("--percentage-jitter", type=float)
    parser.add_argument("--candidate-dir", default="data/augmented_c/copy_paste")
    parser.add_argument("--output-dir", default="data/augmented_c/copy_paste_multi")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.replacement_percentage is not None:
        config["multi_room"]["replacement_percentage"] = args.replacement_percentage
    if args.percentage_jitter is not None:
        config["multi_room"]["random_jitter_percentage_points"] = args.percentage_jitter
    coco, _, train_ids, image_dir = load_inputs(config)
    result = build_multiroom_dataset(
        config,
        coco,
        train_ids,
        image_dir,
        resolve_path(args.candidate_dir),
        resolve_path(args.output_dir),
        strategy="c_multi",
        sample_prefix="c-copy-paste-multi",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
