#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pycocotools.coco import COCO

from src.data.class_agnostic import write_class_agnostic_coco
from src.data.coco_dataset import read_split
from src.utils.config import load_yaml, resolve_path


def main():
    parser = argparse.ArgumentParser(description="Create the class-agnostic view of the Phase 1 dataset")
    parser.add_argument("--source", default="data/processed/instances_deduplicated.json")
    parser.add_argument("--output", default="data/processed/instances_deduplicated_agnostic.json")
    args = parser.parse_args()
    source_path, output_path = resolve_path(args.source), resolve_path(args.output)
    converted = write_class_agnostic_coco(source_path, output_path)
    split_cfg = load_yaml(ROOT / "configs/dataset.yaml")
    split_dir = resolve_path(split_cfg["split_dir"])
    partitions = {name: set(read_split(split_dir / f"{name}.txt")) for name in ("train", "val", "test")}
    if any(partitions[a] & partitions[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Existing Phase 1 splits overlap")
    if set.union(*partitions.values()) != {int(image["id"]) for image in converted["images"]}:
        raise ValueError("Existing splits do not cover the converted dataset")
    COCO(str(output_path))
    counts = {
        name: {
            "images": len(ids),
            "annotations": sum(annotation["image_id"] in ids for annotation in converted["annotations"]),
        }
        for name, ids in partitions.items()
    }
    print(json.dumps({"output": str(output_path), "categories": converted["categories"], "splits": counts}, indent=2))


if __name__ == "__main__":
    main()
