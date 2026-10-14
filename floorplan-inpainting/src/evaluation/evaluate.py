import csv
import gzip
import json
import time

import torch
from torch.utils.data import DataLoader

from src.data.coco_dataset import FloorplanCocoDataset, read_split
from src.data.transforms import DetectionTransform, collate_fn
from src.evaluation.coco_eval import compact_predictions, evaluate_coco_results
from src.evaluation.per_class_metrics import per_class_mask_metrics, rare_aggregate_metrics
from src.models.mask_rcnn import build_mask_rcnn
from src.training.engine import predict_coco
from src.utils.config import resolve_path, split_dir


def evaluate_checkpoint(
    dataset_cfg,
    model_cfg,
    rare_cfg,
    checkpoint,
    split="test",
    output_dir="outputs/metrics",
    dump_predictions=False,
    ocr=None,
):
    annotation_file = resolve_path(dataset_cfg["annotation_file"])
    ids = read_split(split_dir(dataset_cfg["split_dir"]) / f"{split}.txt")
    state = torch.load(checkpoint, map_location="cpu")
    text_head = state.get("text_head") if isinstance(state, dict) else None
    if text_head and ocr is None:
        raise ValueError("Checkpoint is text-aware; pass the OCR provider of its experiment config")
    dataset = FloorplanCocoDataset(
        resolve_path(dataset_cfg["image_dir"]),
        annotation_file,
        ids,
        DetectionTransform(),
        ocr=ocr if text_head else None,
    )
    category_ids = sorted(category["id"] for category in dataset.coco_data["categories"])
    if category_ids != dataset_cfg["category_ids"] or dataset_cfg["num_classes"] != len(category_ids) + 1:
        raise ValueError("Dataset category IDs and configured num_classes disagree")
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=model_cfg["num_workers"], collate_fn=collate_fn
    )
    device = torch.device(model_cfg["device"] if torch.cuda.is_available() else "cpu")
    model = build_mask_rcnn(
        dataset_cfg["num_classes"],
        model_cfg["min_size"],
        model_cfg["max_size"],
        pretrained=False,
        text_head=text_head,
    )
    if (
        isinstance(state, dict)
        and "num_classes" in state
        and state["num_classes"] != dataset_cfg["num_classes"]
    ):
        raise ValueError(
            f"Checkpoint has {state['num_classes']} classes "
            f"but dataset config requires {dataset_cfg['num_classes']}"
        )
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device)
    started = time.time()
    bbox_results, masks = predict_coco(model, loader, device)
    metrics, _, coco_gt, evaluators = evaluate_coco_results(
        annotation_file, bbox_results, masks, ids, return_evaluators=True
    )
    rows = per_class_mask_metrics(coco_gt, evaluators["mask"], ids)
    metrics.update(
        {
            "split": split,
            "class_agnostic": bool(dataset_cfg.get("class_agnostic", False)),
            "inference_seconds": time.time() - started,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }
    )
    if rare_cfg is not None:
        rare_metrics = rare_aggregate_metrics(rows, rare_cfg)
        metrics.update(rare_metrics)
        metrics["rare_class_macro_mask_AP"] = rare_metrics["rare_mask_AP"]
    else:
        metrics["class_agnostic_mask_AP"] = metrics["mask"]["AP"]
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / "baseline_per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if dump_predictions:
        # Raw COCO results for offline subset/bootstrap AP. COCO.loadRes added numpy
        # fields (area, id, ...) in place, so keep only the prediction fields.
        def clean(records, geometry):
            return [
                {
                    "image_id": int(r["image_id"]),
                    "category_id": int(r["category_id"]),
                    "score": float(r["score"]),
                    geometry: r[geometry] if geometry == "segmentation" else [float(v) for v in r[geometry]],
                }
                for r in records
            ]

        payload = {
            "image_ids": [int(i) for i in ids],
            "split": split,
            "annotation_file": str(annotation_file),
            "checkpoint": str(checkpoint),
            "bbox": clean(bbox_results, "bbox"),
            "segm": clean(masks, "segmentation"),
        }
        with gzip.open(output_dir / "coco_results.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
    predictions = compact_predictions(bbox_results, masks, model_cfg["score_threshold"], ids)
    return metrics, rows, predictions
