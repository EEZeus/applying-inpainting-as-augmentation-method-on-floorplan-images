#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.split import search_group_split, write_split_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=20000)
    args = parser.parse_args()
    coco = json.loads((ROOT / "data/processed/instances_deduplicated.json").read_text(encoding="utf-8"))
    cfg = {
        "seed": args.seed,
        "search_iterations": args.iterations,
        "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "size_weight": 4.0,
        "coverage_weight": 4.0,
        "rare_coverage_weight": 3.0,
        "primary_rare_ids": [11, 12, 13, 14, 15, 16],
    }
    assignment, candidates = search_group_split(coco, cfg)
    manifest = write_split_outputs(
        coco, assignment, candidates, cfg, ROOT / "data/splits", ROOT / "outputs/splits", [22, 24, 25, 26]
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
