#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.evaluation.evaluate import evaluate_checkpoint
from src.data.coco_dataset import FloorplanCocoDataset, read_split
from src.data.transforms import DetectionTransform
from src.visualization.predictions import render_prediction_comparisons
from src.utils.config import load_yaml

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=ROOT / "outputs/checkpoints/best.pth")
parser.add_argument("--split", default="test", choices=["val", "test"])
parser.add_argument("--dataset-config", default=ROOT / "configs/dataset.yaml")
parser.add_argument("--model-config", default=ROOT / "configs/baseline.yaml")
parser.add_argument("--output-dir")
parser.add_argument("--figures-dir")
args = parser.parse_args()
dataset_cfg, model_cfg = load_yaml(args.dataset_config), load_yaml(args.model_config)
rare_cfg = None if dataset_cfg.get("class_agnostic") else load_yaml(ROOT / "configs/rare_classes.yaml")
metrics, rows, predictions = evaluate_checkpoint(
    dataset_cfg, model_cfg, rare_cfg, args.checkpoint, args.split,
    args.output_dir or model_cfg.get("metrics_dir", "outputs/metrics"),
)
dataset = FloorplanCocoDataset(ROOT / dataset_cfg["image_dir"], ROOT / dataset_cfg["annotation_file"],
                               read_split(ROOT / dataset_cfg["split_dir"] / f"{args.split}.txt"), DetectionTransform())
render_prediction_comparisons(
    dataset, predictions,
    ROOT / (args.figures_dir or model_cfg.get("figures_dir", "outputs/figures/predictions")),
    model_cfg["score_threshold"], 20,
)
print(json.dumps({"metrics": metrics, "per_class": rows}, ensure_ascii=False, indent=2))
