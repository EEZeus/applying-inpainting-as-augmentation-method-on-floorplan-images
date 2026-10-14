from collections import Counter

import numpy as np


def _mean_valid(values):
    values = values[values > -1]
    return float(np.mean(values)) if values.size else None


def per_class_mask_metrics(coco_gt, mask_evaluator, image_ids):
    """Per-class mask AP read from one accumulated COCOeval.

    Precision has dimensions [IoU, recall, category, area, maxDets]. Reusing it avoids
    repeated COCO.loadRes calls, which mutate the result dicts in place.
    """
    image_ids = set(image_ids)
    test_counts = Counter(
        ann["category_id"] for ann in coco_gt.dataset["annotations"] if ann["image_id"] in image_ids
    )
    category_index = {}
    precision = None
    iou_thresholds = None
    if mask_evaluator is not None:
        category_index = {
            category_id: index for index, category_id in enumerate(mask_evaluator.params.catIds)
        }
        precision = mask_evaluator.eval["precision"]
        iou_thresholds = np.asarray(mask_evaluator.params.iouThrs)
    rows = []
    for category in coco_gt.dataset["categories"]:
        category_id = category["id"]
        count = test_counts[category_id]
        ap = ap50 = None
        if count and precision is not None and category_id in category_index:
            category_precision = precision[:, :, category_index[category_id], 0, -1]
            ap = _mean_valid(category_precision)
            iou50 = np.flatnonzero(np.isclose(iou_thresholds, 0.5))
            if iou50.size:
                ap50 = _mean_valid(category_precision[iou50[0]])
        rows.append(
            {
                "category_id": category_id,
                "category_name": category["name"],
                "test_instances": count,
                "mask_AP": ap,
                "mask_AP50": ap50,
            }
        )
    return rows


def rare_aggregate_metrics(rows, rare_config, role="primary_rare"):
    """Macro and support-weighted rare metrics over evaluable configured classes."""
    allowed = {int(k) for k, value in rare_config["categories"].items() if value["role"] == role}
    selected = [row for row in rows if row["category_id"] in allowed and row["test_instances"]]
    result = {}
    for field, output in [("mask_AP", "rare_mask_AP"), ("mask_AP50", "rare_mask_AP50")]:
        valid = [row for row in selected if row[field] is not None]
        result[output] = sum(row[field] for row in valid) / len(valid) if valid else None
        support = sum(row["test_instances"] for row in valid)
        result[f"weighted_{output}"] = (
            sum(row[field] * row["test_instances"] for row in valid) / support if support else None
        )
    result["rare_evaluable_classes"] = len(selected)
    result["rare_test_instances"] = sum(row["test_instances"] for row in selected)
    return result


def rare_macro_ap(rows, rare_config):
    """Backward-compatible primary-rare mask AP accessor."""
    return rare_aggregate_metrics(rows, rare_config)["rare_mask_AP"]
