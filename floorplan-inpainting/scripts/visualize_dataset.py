#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.coco_dataset import read_split
from src.utils.config import load_yaml, resolve_path
from src.visualization.annotations import render_dataset_annotations

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["train", "val", "test"], default="train")
parser.add_argument("--limit", type=int, default=20)
args = parser.parse_args()
cfg = load_yaml(ROOT / "configs/dataset.yaml")
render_dataset_annotations(
    resolve_path(cfg["annotation_file"]),
    resolve_path(cfg["image_dir"]),
    ROOT / "outputs/figures/annotations",
    read_split(resolve_path(cfg["split_dir"]) / f"{args.split}.txt"),
    args.limit,
)
