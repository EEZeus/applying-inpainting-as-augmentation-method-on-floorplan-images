#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.augmentation.inputs import load_inputs
from src.augmentation.lora_multiroom import build_multiroom_dataset
from src.utils.config import load_yaml, resolve_path


def slug(v):
    return f"{v:g}".replace(".", "p")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=ROOT / "configs/augmentation_lora_multi.yaml")
    p.add_argument("--replacement-percentage", type=float)
    p.add_argument("--percentage-jitter", type=float)
    a = p.parse_args()
    c = load_yaml(a.config)
    if a.replacement_percentage is not None:
        c["multi_room"]["replacement_percentage"] = a.replacement_percentage
    if a.percentage_jitter is not None:
        c["multi_room"]["random_jitter_percentage_points"] = a.percentage_jitter
    coco, _, train, image_dir = load_inputs(c)
    pct = c["multi_room"]["replacement_percentage"]
    jit = c["multi_room"]["random_jitter_percentage_points"]
    output = resolve_path(c["output"]["root"]) / f"pct_{slug(pct)}_jitter_{slug(jit)}"
    print(
        json.dumps(
            build_multiroom_dataset(
                c, coco, train, image_dir, resolve_path(c["input"]["candidate_dir"]), output
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
