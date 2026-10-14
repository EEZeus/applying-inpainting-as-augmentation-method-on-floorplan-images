"""Bounded image weights usable with torch WeightedRandomSampler in Phase 3."""

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def sampling_weights(coco, real_train_coco, rare_ids, config, image_ids=None):
    counts = Counter(a["category_id"] for a in real_train_coco["annotations"])
    per_image = defaultdict(Counter)
    for ann in coco["annotations"]:
        if ann["category_id"] in rare_ids:
            per_image[ann["image_id"]][ann["category_id"]] += 1
    ids = sorted(i["id"] for i in coco["images"]) if image_ids is None else list(image_ids)
    if set(ids) != {i["id"] for i in coco["images"]} or len(ids) != len(set(ids)):
        raise ValueError("Sampler image order must cover the dataset exactly once")
    mode = config.get("mode", "rare_weighted")
    weight_mode = config.get("class_weight_mode", "inverse_sqrt")
    if mode not in {"uniform", "rare_weighted"} or weight_mode not in {"inverse_sqrt", "inverse", "uniform"}:
        raise ValueError("Unknown sampling mode")
    multiplier, cap = float(config.get("lambda", 2.0)), float(config.get("max_image_weight", 5.0))
    if not np.isfinite([multiplier, cap]).all() or multiplier < 0 or cap < 1:
        raise ValueError("Sampling requires finite lambda >= 0 and cap >= 1")
    rows = []
    for image_id in ids:
        present = per_image[image_id]
        score = 0.0
        for category, count in present.items():
            if counts[category] == 0:
                raise ValueError("Rare class lacks real training support")
            if weight_mode == "inverse_sqrt":
                divisor = np.sqrt(counts[category])
            elif weight_mode == "inverse":
                divisor = counts[category]
            else:
                divisor = 1.0
            score += count / divisor
        rows.append(
            {
                "image_id": image_id,
                "rare_classes_present": ";".join(map(str, sorted(present))),
                "rare_instance_count": sum(present.values()),
                "base_weight": 1.0,
                "final_weight": min(cap, 1 + multiplier * score) if mode == "rare_weighted" else 1.0,
            }
        )
    weights = np.asarray([r["final_weight"] for r in rows], dtype=np.float64)
    # With weights in [1, cap], max probability <= cap/(N-1+cap).
    for row, weight in zip(rows, weights):
        row["probability"] = float(weight / weights.sum())
    return weights, rows


def make_sampler(weights, seed=42, num_samples=None):
    weights = torch.as_tensor(weights, dtype=torch.double)
    if weights.numel() == 0 or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Sampler requires positive finite weights")
    return WeightedRandomSampler(
        weights, num_samples or len(weights), replacement=True, generator=torch.Generator().manual_seed(seed)
    )


def write_sampling_report(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            [
                "image_id",
                "rare_classes_present",
                "rare_instance_count",
                "base_weight",
                "final_weight",
                "probability",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
