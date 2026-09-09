#!/usr/bin/env python3
"""Download model weights only; no dataset images are read or uploaded."""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from huggingface_hub import snapshot_download
from src.utils.config import load_yaml, resolve_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/augmentation_a.yaml")
    args = parser.parse_args()
    cfg = load_yaml(resolve_path(args.config))["generation"]
    print(snapshot_download(cfg["model_name"], revision=cfg.get("model_revision"),
                            cache_dir=str(resolve_path(cfg["cache_dir"])),
                            allow_patterns=["*.json", "*.txt", "*.safetensors"],
                            ignore_patterns=["*.non_ema.*"]))
