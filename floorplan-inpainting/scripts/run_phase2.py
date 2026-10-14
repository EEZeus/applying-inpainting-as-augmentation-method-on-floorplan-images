#!/usr/bin/env python3
"""Unattended Phase 2: generate A/B, derive crops, validate, and report. No training."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    commands = [
        ["inspect_safe_masks.py", "--limit", "30"],
        ["generate_strategy_a.py", "--retry-errors"],
        ["generate_rare_crops.py"],
        ["generate_strategy_b.py", "--retry-errors"],
        ["validate_augmented_dataset.py", "data/augmented_a/instances_train_aug_a.json"],
        ["validate_augmented_dataset.py", "data/augmented_a/crops/instances_train_aug_a_crops.json"],
        ["validate_augmented_dataset.py", "data/augmented_b/instances_train_aug_b.json"],
        ["report_augmentation.py"],
    ]
    for script, *args in commands:
        retries = 2 if script in {"generate_strategy_a.py", "generate_strategy_b.py"} else 0
        for attempt in range(retries + 1):
            result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], cwd=ROOT)
            if result.returncode == 0:
                break
            if attempt == retries:
                raise SystemExit(result.returncode)
            print(
                f"Retrying transient failures in {script} ({attempt + 1}/{retries}); "
                "completed samples are reused",
                flush=True,
            )
