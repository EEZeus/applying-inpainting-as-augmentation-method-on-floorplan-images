#!/usr/bin/env python3
"""Write before/after class occurrences and bounded sampling diagnostics."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.augmentation.inputs import load_inputs, original_train
from src.augmentation.rare_sampling import sampling_weights, write_sampling_report
from src.augmentation.reports import class_count_report
from src.utils.config import load_yaml, resolve_path


def main():
    coco, rare, train, _ = load_inputs(load_yaml(resolve_path("configs/augmentation_a.yaml")))
    original = original_train(coco, train)
    datasets = {}
    for name, relative in [
        ("Strategy A full", "data/augmented_a/instances_train_aug_a.json"),
        ("Strategy A crops", "data/augmented_a/crops/instances_train_aug_a_crops.json"),
        ("Strategy B automatic (semantics unverified)", "data/augmented_b/instances_train_aug_b.json"),
    ]:
        path = resolve_path(relative)
        if path.exists():
            datasets[name] = json.loads(path.read_text())
    cfg = load_yaml(resolve_path("configs/rare_sampling.yaml"))["sampling"]
    rare_ids = {
        int(cid) for cid, value in rare["categories"].items() if value["role"] in cfg["include_roles"]
    }
    for name, data in [
        ("sampling_weights", datasets.get("Strategy A full", original)),
        ("sampling_weights_original", original),
    ]:
        _, rows = sampling_weights(data, original, rare_ids, cfg)
        write_sampling_report(resolve_path(f"outputs/augmentation_a/{name}.csv"), rows)
    class_count_report(resolve_path("outputs/augmentation_a/class_counts.csv"), original, datasets)
    print("Wrote class counts and sampling weights (available datasets only)")


if __name__ == "__main__":
    main()
