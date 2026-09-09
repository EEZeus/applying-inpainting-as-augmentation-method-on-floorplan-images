#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.augmentation.inputs import load_inputs
from src.augmentation.strategy_b import run_strategy_b, plan_strategy_b
from src.augmentation.pipeline import _atomic_json
from src.utils.config import load_yaml, resolve_path
from src.utils.reproducibility import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Generate automatically QC-accepted experimental Strategy B samples")
    parser.add_argument("--config", default=ROOT / "configs/augmentation_b.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--model-cache")
    parser.add_argument("--output-dir")
    parser.add_argument("--qa-dir")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="Write geometry screening without loading diffusion")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.resolution:
        config["crop"]["output_size"] = args.resolution
    config["generation"]["cache_dir"] = str(resolve_path(args.model_cache or config["generation"]["cache_dir"]))
    coco, rare, train, image_dir = load_inputs(config)
    qa_dir = resolve_path(args.qa_dir or config["output"]["qa_dir"])
    if args.plan_only:
        plan, diagnostics = plan_strategy_b(coco, train, config)
        _atomic_json(qa_dir / "geometry_screening.json", diagnostics)
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
        return
    seed_everything(config["seed"])
    summary = run_strategy_b(config, coco, rare, train, image_dir,
        resolve_path(args.output_dir or config["output"]["dataset_dir"]), qa_dir,
        limit=args.limit, num_workers=args.num_workers, retry_errors=args.retry_errors)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
