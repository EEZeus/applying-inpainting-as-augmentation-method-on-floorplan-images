#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.augmentation.inputs import load_inputs
from src.augmentation.pipeline import validate_augmented_coco
from src.augmentation.validation import validate_full_provenance
from src.utils.config import load_yaml, resolve_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate COCO files, polygons, IDs, files, and train provenance")
    parser.add_argument("annotations")
    args = parser.parse_args()
    source, _, train, image_dir = load_inputs(load_yaml(resolve_path("configs/augmentation_a.yaml")))
    path = resolve_path(args.annotations)
    dataset = json.loads(path.read_text())
    if (path.parent / "run_manifest.json").exists():
        print(validate_full_provenance(dataset, source, image_dir, path.parent, train))
    else:
        validate_augmented_coco(dataset, path.parent, train)
    print(f"Validated {path}")
