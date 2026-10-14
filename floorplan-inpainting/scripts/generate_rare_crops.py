#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.augmentation.inputs import load_inputs, original_train
from src.augmentation.rare_crops import build_rare_crops
from src.utils.config import load_yaml, resolve_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rare_crops.yaml")
    parser.add_argument("--real-only", action="store_true")
    parser.add_argument("--annotations", default="data/augmented_a/instances_train_aug_a.json")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    cfg = load_yaml(resolve_path(args.config))
    source, rare, train, image_dir = load_inputs(load_yaml(resolve_path("configs/augmentation_a.yaml")))
    rare_ids = {
        int(cid) for cid, value in rare["categories"].items() if value["role"] in cfg["include_roles"]
    }
    if args.real_only:
        coco = original_train(source, train)
        output = resolve_path(args.output_dir or "outputs/rare_crops/real")
    else:
        annotation_path = resolve_path(args.annotations)
        coco = json.loads(annotation_path.read_text())
        image_dir = annotation_path.parent
        output = resolve_path(args.output_dir or "data/augmented_a/crops")
    result = build_rare_crops(coco, image_dir, output, train, rare_ids, cfg)
    n_crops, n_annotations = len(result["images"]), len(result["annotations"])
    print(f"Wrote {n_crops} contextual crops and {n_annotations} annotations to {output}")


if __name__ == "__main__":
    main()
