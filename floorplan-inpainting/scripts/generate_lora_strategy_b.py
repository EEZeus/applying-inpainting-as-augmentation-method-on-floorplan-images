#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.augmentation.inputs import load_inputs
from src.augmentation.lora_strategy_b import plan_lora_strategy_b, run_lora_strategy_b
from src.augmentation.pipeline import _atomic_json
from src.utils.config import load_yaml, resolve_path
from src.utils.reproducibility import seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=ROOT / "configs/augmentation_lora_b.yaml")
    p.add_argument("--limit", type=int)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--retry-errors", action="store_true")
    a = p.parse_args()
    c = load_yaml(a.config)
    c["generation"]["cache_dir"] = str(resolve_path(c["generation"]["cache_dir"]))
    c["generation"]["lora_path"] = str(resolve_path(c["generation"]["lora_path"]))
    coco, rare, train, image_dir = load_inputs(c)
    output = resolve_path(c["output"]["dataset_dir"])
    qa = resolve_path(c["output"]["qa_dir"])
    if a.plan_only:
        _, diagnostics = plan_lora_strategy_b(coco, train, c)
        _atomic_json(qa / "geometry_screening.json", diagnostics)
        print(json.dumps(diagnostics, indent=2))
        return
    seed_everything(c["seed"])
    result = run_lora_strategy_b(
        c, coco, rare, train, image_dir, output, qa, limit=a.limit, retry_errors=a.retry_errors
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
