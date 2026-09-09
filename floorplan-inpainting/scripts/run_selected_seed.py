#!/usr/bin/env python3
"""Run an extra seed for a fixed control or automatically selected A/B method."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "e0": "e0_original.yaml", "e1": "e1_oversampling.yaml", "e2": "e2_strategy_a.yaml",
    "e3": "e3_strategy_a_weighted.yaml", "e4": "e4_strategy_a_crops.yaml",
    "e5": "e5_strategy_a_weighted_crops.yaml", "e6": "e6_strategy_b.yaml",
    "e7": "e7_strategy_b_weighted.yaml",
}


def score(track, experiment_id):
    path = ROOT / f"outputs/experiments/{track}/{experiment_id}/seed_42/result.json"
    if not path.exists():
        return float("-inf")
    metrics = json.loads(path.read_text())["test_metrics"]
    value = metrics.get("rare_mask_AP") if track == "multiclass" else metrics["mask"]["AP"]
    return float("-inf") if value is None else value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=["e0", "e1", "best_a", "best_b"])
    parser.add_argument("--track", required=True, choices=["multiclass", "agnostic"])
    parser.add_argument("--seed", required=True, type=int, choices=[123, 2026])
    args = parser.parse_args()
    candidates = {"best_a": ["e2", "e3", "e4", "e5"], "best_b": ["e6", "e7"]}
    experiment_id = args.role if args.role in {"e0", "e1"} else max(candidates[args.role], key=lambda item: score(args.track, item))
    if score(args.track, experiment_id) == float("-inf"):
        raise SystemExit("Single-seed comparison must finish before extra seeds")
    command = [sys.executable, str(ROOT / "scripts/run_experiment.py"), "--config",
               str(ROOT / "configs/experiments" / CONFIGS[experiment_id]), "--track", args.track,
               "--seed", str(args.seed), "--resume"]
    raise SystemExit(subprocess.run(command, cwd=ROOT).returncode)
