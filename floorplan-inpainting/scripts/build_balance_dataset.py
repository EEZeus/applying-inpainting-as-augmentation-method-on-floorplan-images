#!/usr/bin/env python3
"""Follow-up X1: compose accepted copy-paste edits of one dose into multi-room plans.

Every accepted (frequent target, rare donor) edit of a source plan goes into
one synthetic variant of that plan (replacement_percentage 100 in the dose
config), so a plan swapped in by the paired_variant sampler carries all of
its conversions.

    python scripts/build_balance_dataset.py --config configs/augmentation_c_balance_b50.yaml
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augmentation.inputs import load_inputs
from src.augmentation.lora_multiroom import build_multiroom_dataset
from src.augmentation.pipeline import _atomic_json
from src.utils.config import load_yaml, resolve_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(resolve_path(args.config))
    coco, _, train_ids, image_dir = load_inputs(config)
    root = resolve_path(config["output"]["dataset_root"])
    tag = root.name
    manifest = build_multiroom_dataset(
        config,
        coco,
        train_ids,
        image_dir,
        root / "copy_paste",
        root / "copy_paste_multi",
        strategy="c_multi",
        sample_prefix=f"c-balance-{tag}",
        canonical_order=True,
    )
    real = Counter(int(a["category_id"]) for a in coco["annotations"] if int(a["image_id"]) in train_ids)
    converted = Counter(c["target_category_id"] for g in manifest["groups"] for c in g["conversions"])
    summary = {
        "dose": tag,
        "synthetic_plans": manifest["synthetic_plans"],
        "converted_rooms": int(sum(converted.values())),
        "per_class": {
            str(c): {
                "real": real[c],
                "converted": converted.get(c, 0),
                "total": real[c] + converted.get(c, 0),
            }
            for c in sorted(int(x) for x in config["classes"]["target_ids"])
        },
        "donor_reuse_cap": int(config["selection"]["max_donor_reuse"]),
        "target_per_class": int(config["balancing"]["max_target_instances"]),
    }
    _atomic_json(root / "balance_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
