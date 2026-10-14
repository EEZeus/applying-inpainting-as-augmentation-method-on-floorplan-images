#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.augmentation.review import build_review
from src.utils.config import resolve_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/augmented_b")
    parser.add_argument("--review", default="outputs/augmentation_b/review.csv")
    args = parser.parse_args()
    rows = build_review(resolve_path(args.dataset_dir), resolve_path(args.review))
    approved = sum(r["status"] == "approved" for r in rows)
    print(f"Audit contains {len(rows)} automatic decisions; {approved} approved")
