"""Train-only geometry screening for experimental common-to-rare conversion."""

import math
from collections import Counter
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .pipeline import Candidate, _atomic_json, run_augmentation
from .review import apply_review


def room_geometry(annotation):
    polygons = [Polygon(np.asarray(p).reshape(-1, 2)) for p in annotation["segmentation"]]
    shape = unary_union([p if p.is_valid else p.buffer(0) for p in polygons])
    width, height = annotation["bbox"][2:]
    return {
        "area": float(shape.area),
        "aspect_ratio": float(width / height),
        "width": float(width),
        "height": float(height),
        "solidity": float(shape.area / shape.convex_hull.area) if shape.area else 0.0,
    }


def plan_strategy_b(coco, train_ids, config):
    train = [a for a in coco["annotations"] if a["image_id"] in train_ids]
    names = {c["id"]: c["name"] for c in coco["categories"]}
    counts = Counter(a["category_id"] for a in train)
    sources = sorted(
        [a for a in train if a["category_id"] in config["classes"]["source_ids"]], key=lambda a: a["id"]
    )
    geometry = {a["id"]: room_geometry(a) for a in train}
    tolerance = config["geometry"]["tolerance"]
    ranges, diagnostics, pairs = {}, {}, []
    rng = np.random.default_rng(config["seed"])
    used_sources = set()
    for target in config["classes"]["target_ids"]:
        if target == 19:
            raise ValueError("Labor is excluded from Strategy B")
        examples = [geometry[a["id"]] for a in train if a["category_id"] == target]
        if not examples:
            continue
        bounds = {}
        for feature in examples[0]:
            low, high = np.quantile([g[feature] for g in examples], [0.05, 0.95])
            bounds[feature] = [max(0.0, float(low * (1 - tolerance))), float(high * (1 + tolerance))]
        ranges[target] = bounds
        eligible = [
            a
            for a in sources
            if a["category_id"] != target
            and all(low <= geometry[a["id"]][feature] <= high for feature, (low, high) in bounds.items())
        ]
        total_target = min(
            math.ceil(config["balancing"]["multiplier"] * counts[target]),
            config["balancing"]["max_target_instances"],
        )
        budget = max(0, total_target - counts[target])
        rng.shuffle(eligible)
        chosen = [a for a in eligible if a["id"] not in used_sources][:budget]
        used_sources.update(a["id"] for a in chosen)
        diagnostics[str(target)] = {
            "category_name": names[target],
            "real_count": counts[target],
            "target_total": total_target,
            "candidate_budget": budget,
            "geometry_eligible": len(eligible),
            "planned": len(chosen),
            "geometry_range": bounds,
        }
        pairs.extend((a, target) for a in chosen)
    pairs.sort(key=lambda pair: (pair[1], pair[0]["id"]))
    stride = max(Counter(a["image_id"] for a in coco["annotations"]).values())
    # Reserve disjoint ranges so A and B can later be mixed without ID collisions.
    max_im = max(i["id"] for i in coco["images"]) + 1_000_000
    max_ann = max(a["id"] for a in coco["annotations"]) + 1_000_000_000
    candidates = [
        Candidate(
            sample_id=f"b-ann{a['id']}-to{target}",
            candidate_index=i,
            source_image_id=a["image_id"],
            target_annotation_id=a["id"],
            category_id=target,
            category_name=names[target],
            variant_index=0,
            seed=config["seed"] + i,
            synthetic_image_id=max_im + i,
            annotation_id_start=max_ann + i * stride,
            strategy="b",
            source_geometry=geometry[a["id"]],
            target_geometry_range=ranges[target],
        )
        for i, (a, target) in enumerate(pairs)
    ]
    return (candidates, stride), diagnostics


def run_strategy_b(config, coco, rare_config, train_ids, image_dir, output_dir, qa_dir, **kwargs):
    plan, diagnostics = plan_strategy_b(coco, train_ids, config)
    _atomic_json(Path(qa_dir) / "geometry_screening.json", diagnostics)
    result = run_augmentation(
        config,
        coco,
        rare_config,
        train_ids,
        image_dir,
        output_dir,
        qa_dir,
        candidate_plan=plan,
        strategy="b",
        **kwargs,
    )
    apply_review(output_dir, Path(qa_dir) / "review.csv", coco, train_ids, image_dir)
    result["source_rare_instances"] = sum(d["real_count"] for d in diagnostics.values())
    result["planned_candidates_per_target"] = {d["category_name"]: d["planned"] for d in diagnostics.values()}
    result["acceptance_method"] = "automatic_geometry_and_pixel_qc"
    result["semantic_correctness"] = "unverified"
    _atomic_json(Path(qa_dir) / "summary.json", result)
    return result
