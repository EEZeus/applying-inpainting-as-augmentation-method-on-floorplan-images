#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.train import train
from src.utils.config import load_yaml

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=None, help="Use 1-2 for a smoke run")
parser.add_argument("--dataset-config", default=ROOT / "configs/dataset.yaml")
parser.add_argument("--model-config", default=ROOT / "configs/baseline.yaml")
args = parser.parse_args()
print(json.dumps(train(load_yaml(args.dataset_config), load_yaml(args.model_config), args.epochs), indent=2))
