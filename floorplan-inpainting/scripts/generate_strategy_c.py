#!/usr/bin/env python3
"""Generate isolated Strategy C control and exemplar-inpainting datasets."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augmentation.inputs import load_inputs
from src.augmentation.pipeline import _atomic_json
from src.augmentation.strategy_c import MODES, plan_strategy_c, run_strategy_c
from src.utils.config import load_yaml, resolve_path
from src.utils.reproducibility import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Generate exemplar-guided Strategy C datasets")
    parser.add_argument("--config", default=ROOT / "configs/augmentation_c.yaml")
    parser.add_argument("--mode", choices=[*MODES, "all"], default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--model-cache")
    parser.add_argument("--dataset-root")
    parser.add_argument("--qa-root")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.resolution:
        config["crop"]["output_size"] = args.resolution
    config["generation"]["cache_dir"] = str(resolve_path(
        args.model_cache or config["generation"]["cache_dir"]))
    coco, rare, train, image_dir = load_inputs(config)
    modes = list(MODES) if args.mode == "all" else [args.mode]
    dataset_root = resolve_path(args.dataset_root or config["output"]["dataset_root"])
    qa_root = resolve_path(args.qa_root or config["output"]["qa_root"])
    results = {}
    for mode in modes:
        output_dir, qa_dir = dataset_root / mode, qa_root / mode
        if args.plan_only:
            (_, _), diagnostics = plan_strategy_c(coco, train, config, mode)
            _atomic_json(qa_dir / "pairing_diagnostics.json", diagnostics)
            results[mode] = diagnostics
            continue
        seed_everything(int(config["seed"]))
        results[mode] = run_strategy_c(
            config, coco, rare, train, image_dir, output_dir, qa_dir, mode,
            limit=args.limit, num_workers=args.num_workers,
            retry_errors=args.retry_errors)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if any(value.get("errors", 0) for value in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
