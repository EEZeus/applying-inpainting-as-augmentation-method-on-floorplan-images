#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augmentation.pipeline import run_augmentation
from src.augmentation.inputs import load_inputs
from src.data.coco_dataset import read_split
from src.utils.config import load_yaml, resolve_path
from src.utils.reproducibility import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Generate structure-preserving Strategy A samples")
    parser.add_argument("--config", default=ROOT / "configs/augmentation_a.yaml")
    parser.add_argument("--limit", type=int, help="Run only the first N planned candidates")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--model-cache")
    parser.add_argument("--output-dir")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--qa-dir")
    args = parser.parse_args()

    config = load_yaml(args.config)
    if config["generation"]["backend"] != "stable_diffusion_inpaint":
        raise ValueError("Phase 2 supports only backend=stable_diffusion_inpaint")
    if args.resolution is not None:
        config["crop"]["output_size"] = args.resolution
    if args.model_cache is not None:
        config["generation"]["cache_dir"] = args.model_cache
    if args.num_workers is not None:
        config["runtime"]["num_workers"] = args.num_workers

    config["generation"]["cache_dir"] = str(resolve_path(config["generation"]["cache_dir"], ROOT))
    output_dir = resolve_path(args.output_dir or config["output"]["dataset_dir"], ROOT)
    qa_dir = resolve_path(args.qa_dir or config["output"]["qa_dir"], ROOT)
    coco, rare, train_ids, image_dir = load_inputs(config)
    seed_everything(int(config["seed"]))
    summary = run_augmentation(
        config, coco, rare, train_ids, image_dir,
        output_dir, qa_dir, limit=args.limit,
        num_workers=args.num_workers, retry_errors=args.retry_errors,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
