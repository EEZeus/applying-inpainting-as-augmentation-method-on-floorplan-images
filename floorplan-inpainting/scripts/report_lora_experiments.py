#!/usr/bin/env python3
"""Isolated E12/E13 report; never mutates earlier experiment reports."""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.coco_dataset import read_split

METHODS = {
    "e0": "Original",
    "e1": "Original + rare-weighted sampling",
    "e12": "LoRA common-to-rare, uniform",
    "e13": "LoRA common-to-rare + rare-weighted",
}


def load(track, eid):
    root = ROOT / ("outputs/experiments" if eid in {"e0", "e1"} else "outputs/experiments_lora")
    path = root / track / eid / "seed_42/result.json"
    return json.loads(path.read_text()) if path.is_file() else None


def nested(value, *keys):
    for key in keys:
        if value is None:
            return None
        value = value.get(key)
    return value


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    output = ROOT / "outputs/final_lora"
    results = {e: load("multiclass", e) for e in METHODS}
    rows = []
    agnostic = []
    for eid, method in METHODS.items():
        m, a = results[eid], load("agnostic", eid)
        rows.append(
            {
                "Experiment": eid.upper(),
                "Method": method,
                "Mask AP": nested(m, "test_metrics", "mask", "AP"),
                "Mask AP50": nested(m, "test_metrics", "mask", "AP50"),
                "Rare AP": nested(m, "test_metrics", "rare_mask_AP"),
                "Rare AP50": nested(m, "test_metrics", "rare_mask_AP50"),
            }
        )
        agnostic.append(
            {
                "Experiment": eid.upper(),
                "Method": method,
                "Mask AP": nested(a, "test_metrics", "mask", "AP"),
                "Mask AP50": nested(a, "test_metrics", "mask", "AP50"),
                "Mask AP75": nested(a, "test_metrics", "mask", "AP75"),
            }
        )
    write(output / "experiment_summary.csv", rows)
    write(output / "experiment_summary_agnostic.csv", agnostic)
    original = json.loads((ROOT / "data/processed/instances_deduplicated.json").read_text())
    train = set(read_split(ROOT / "data/splits/train.txt"))
    real = Counter(a["category_id"] for a in original["annotations"] if a["image_id"] in train)
    count_rows = list(csv.DictReader((ROOT / "data/augmented_lora_b/before_after_class_counts.csv").open()))
    write(output / "augmentation_counts.csv", count_rows)
    per_class = []
    for category in original["categories"]:
        row = {"ID": category["id"], "Class": category["name"], "Train Real": real[category["id"]]}
        for eid in METHODS:
            match = next(
                (
                    x
                    for x in (results[eid] or {}).get("per_class", [])
                    if int(x["category_id"]) == category["id"]
                ),
                None,
            )
            row[f"{eid.upper()} AP"] = match.get("mask_AP") if match else None
        per_class.append(row)
    write(output / "per_class_results.csv", per_class)

    def delta(left, right, key="rare_mask_AP"):
        a = nested(results[left], "test_metrics", key)
        b = nested(results[right], "test_metrics", key)
        return None if a is None or b is None else b - a

    comparisons = [
        ("E0 original → E12 LoRA augmentation", "e0", "e12"),
        ("E1 weighted original → E13 LoRA + weighting", "e1", "e13"),
        ("E12 uniform → E13 weighted", "e12", "e13"),
    ]
    lines = [
        "# LoRA inpainting augmentation result summary",
        "",
        "This report is isolated from existing E0–E11 outputs.",
        "",
        "## Comparisons",
        "",
    ]
    for label, left, right in comparisons:
        value = delta(left, right)
        lines.append(f"- {label}: " + ("pending." if value is None else f"rare mask AP change {value:+.4f}."))
    lines += [
        "",
        "Classical LoRA crop variants are repeated appearances of real geometries, not independent rooms. "
        "Only automatically QC-accepted common-to-rare conversions are counted as synthetic "
        "target-class geometries. "
        "Semantic correctness remains unverified.",
    ]
    (output / "result_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "completed_multiclass": sum(load("multiclass", e) is not None for e in ("e12", "e13")),
                "completed_agnostic": sum(load("agnostic", e) is not None for e in ("e12", "e13")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
