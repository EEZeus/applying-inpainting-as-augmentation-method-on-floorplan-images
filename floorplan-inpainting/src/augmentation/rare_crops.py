"""Contextual training crops with geometric clipping of every intersecting room."""
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from .crops import CropTransform
from .pipeline import _atomic_json, _save_png_atomic, validate_augmented_coco


def clip_annotation(annotation, transform):
    pieces = [Polygon(np.asarray(p).reshape(-1, 2)) for p in annotation["segmentation"]]
    geometry = unary_union([p if p.is_valid else p.buffer(0) for p in pieces])
    clipped = geometry.intersection(box(transform.x0, transform.y0, transform.x1, transform.y1))
    if clipped.is_empty or clipped.area <= 0:
        return None
    scale = transform.output_size / transform.padded_side
    clipped = affinity.translate(clipped, transform.pad_left - transform.x0, transform.pad_top - transform.y0)
    clipped = affinity.scale(clipped, xfact=scale, yfact=scale, origin=(0, 0))
    polygons = [clipped] if clipped.geom_type == "Polygon" else [p for p in clipped.geoms if p.geom_type == "Polygon"]
    if not polygons:
        return None
    if any(p.interiors for p in polygons):
        raise ValueError("Polygon with holes cannot be represented by COCO exterior polygons")
    clipped = unary_union(polygons)
    x0, y0, x1, y1 = clipped.bounds
    result = copy.deepcopy(annotation)
    result["segmentation"] = [np.asarray(p.exterior.coords)[:-1].ravel().tolist() for p in polygons]
    result["bbox"] = [x0, y0, x1 - x0, y1 - y0]
    result["area"] = float(clipped.area)
    result["source_annotation_id"] = annotation["id"]
    return result


def build_rare_crops(coco, image_dir, output_dir, train_ids, rare_ids, config,
                     filename="instances_train_aug_a_crops.json"):
    output_dir = Path(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    by_image = defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)
    context = max(float(config.get("context_fraction", 1.)), float(config.get("min_context_fraction", .75)))
    output = {"categories": copy.deepcopy(coco["categories"]), "images": [], "annotations": [],
              "info": {"description": "Rare-centered training crops; mix with full plans, not an independent geometry dataset"}}
    next_ann = 2_000_000_000
    for info in sorted(coco["images"], key=lambda im: im["id"]):
        source_id = info.get("source_image_id", info["id"])
        if source_id not in train_ids:
            raise ValueError("Non-training image passed to rare crop generation")
        anns = by_image[info["id"]]
        targets = [a for a in anns if a["category_id"] in rare_ids]
        if info.get("synthetic"):
            # A single crop per generated target, avoiding extra duplicate contexts.
            source_target = info["target_annotation_id"]
            source_annotations = sorted([a for a in coco["annotations"] if a["image_id"] == source_id], key=lambda a: a["id"])
            original_ordinal = next((i for i, a in enumerate(source_annotations) if a["id"] == source_target), None)
            if original_ordinal is None:
                raise ValueError("Synthetic crop target cannot be traced to original annotation")
            targets = [sorted(anns, key=lambda a: a["id"])[original_ordinal]]
        if not targets:
            continue
        with Image.open(Path(image_dir) / info["file_name"]) as handle:
            image = np.asarray(handle.convert("RGB"))
        for target in sorted(targets, key=lambda a: a["id"]):
            transform = CropTransform.from_bbox(target["bbox"], info["width"], info["height"],
                                                context, config.get("output_size", 512))
            image_id = 2_000_000 + len(output["images"])
            name = f"images/crop_im{info['id']}_ann{target['id']}.png"
            _save_png_atomic(output_dir / name, transform.forward_image(image))
            output["images"].append({"id": image_id, "file_name": name,
                "width": transform.output_size, "height": transform.output_size,
                "source_image_id": source_id, "parent_image_id": info["id"],
                "target_annotation_id": target["id"], "synthetic": info.get("synthetic", False),
                "strategy": info.get("strategy", "real"), "crop_transform": transform.to_dict()})
            target_retained = False
            for annotation in sorted(anns, key=lambda a: a["id"]):
                clipped = clip_annotation(annotation, transform)
                if clipped is None:
                    continue
                clipped.update(id=next_ann, image_id=image_id)
                next_ann += 1
                target_retained |= annotation["id"] == target["id"]
                output["annotations"].append(clipped)
            if not target_retained:
                raise ValueError("Target room lost during crop transform")
    validate_augmented_coco(output, output_dir, train_ids)
    _atomic_json(output_dir / filename, output)
    return output
