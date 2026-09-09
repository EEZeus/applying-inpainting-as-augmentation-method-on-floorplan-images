#!/usr/bin/env python3
"""Write a Strategy C report without modifying the completed Phase 3 report."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.coco_dataset import read_split


METHODS = {
    "e0": "Original (existing Phase 3)",
    "e8": "C0 multi-room rare-exemplar copy-paste",
    "e9": "C1 rare-exemplar feather blend",
    "e10": "C2 exemplar-seeded diffusion inpainting",
    "e11": "C2 inpainting + rare-aware sampling",
}


def result(track, experiment_id, seed=42):
    root = ROOT / ("outputs/experiments" if experiment_id == "e0" else "outputs/experiments_c_multi")
    path = root / track / experiment_id / f"seed_{seed}" / "result.json"
    return json.loads(path.read_text()) if path.exists() else None


def nested(value, *keys):
    for key in keys:
        if value is None:
            return None
        value = value.get(key)
    return value


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def metadata(path):
    latest = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[row["sample_id"]] = row
    return list(latest.values())


def main():
    output = ROOT / "outputs/final_c_multi"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows, agnostic_rows, all_results = [], [], {}
    for experiment_id, method in METHODS.items():
        multiclass = result("multiclass", experiment_id)
        agnostic = result("agnostic", experiment_id)
        all_results[experiment_id] = multiclass
        rows.append({"Experiment": experiment_id.upper(), "Method": method,
                     "Mask AP": nested(multiclass, "test_metrics", "mask", "AP"),
                     "Mask AP50": nested(multiclass, "test_metrics", "mask", "AP50"),
                     "Rare AP": nested(multiclass, "test_metrics", "rare_mask_AP"),
                     "Rare AP50": nested(multiclass, "test_metrics", "rare_mask_AP50")})
        agnostic_rows.append({"Experiment": experiment_id.upper(), "Method": method,
                              "Mask AP": nested(agnostic, "test_metrics", "mask", "AP"),
                              "Mask AP50": nested(agnostic, "test_metrics", "mask", "AP50"),
                              "Mask AP75": nested(agnostic, "test_metrics", "mask", "AP75")})
    write_csv(output / "experiment_summary.csv", rows)
    write_csv(output / "experiment_summary_agnostic.csv", agnostic_rows)

    original = json.loads((ROOT / "data/processed/instances_deduplicated.json").read_text())
    train_ids = set(read_split(ROOT / "data/splits/train.txt"))
    test_ids = set(read_split(ROOT / "data/splits/test.txt"))
    train = Counter(a["category_id"] for a in original["annotations"] if a["image_id"] in train_ids)
    test = Counter(a["category_id"] for a in original["annotations"] if a["image_id"] in test_ids)
    per_class = []
    for category in original["categories"]:
        row = {"ID": category["id"], "Class": category["name"],
               "Train Real": train[category["id"]], "Test Real": test[category["id"]]}
        for experiment_id in METHODS:
            match = next((item for item in (all_results[experiment_id] or {}).get("per_class", [])
                          if int(item["category_id"]) == int(category["id"])), None)
            row[f"{experiment_id.upper()} AP"] = match.get("mask_AP") if match else None
        per_class.append(row)
    write_csv(output / "per_class_results.csv", per_class)

    augmentation = []
    for mode in ("copy_paste", "feather_blend", "diffusion_inpaint"):
        records = metadata(ROOT / "data/augmented_c" / mode / "metadata.jsonl")
        for category in sorted({r.get("category_name") for r in records}):
            selected = [r for r in records if r.get("category_name") == category]
            accepted = [r for r in selected if r.get("accepted")]
            mean = lambda key: (float(np.mean([r[key] for r in accepted if r.get(key) is not None]))
                                if any(r.get(key) is not None for r in accepted) else None)
            augmentation.append({"Mode": mode, "Class": category, "Attempted": len(selected),
                                 "Accepted": len(accepted), "Acceptance Rate":
                                 len(accepted) / len(selected) if selected else None,
                                 "Mean Donor Coverage": mean("donor_coverage_fraction"),
                                 "Mean Donor Dark Retention": mean("donor_dark_retention_fraction"),
                                 "Mean Outside Change": mean("outside_change_ratio"),
                                 "Mean Boundary Change": mean("protected_boundary_change_ratio")})
    if augmentation:
        write_csv(output / "augmentation_statistics.csv", augmentation)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    values = [0 if row["Rare AP"] is None else row["Rare AP"] for row in rows]
    bars = ax.bar(range(len(rows)), values, color=["#777777", "#4c78a8", "#72b7b2", "#f58518", "#e45756"])
    for bar, row in zip(bars, rows):
        if row["Rare AP"] is None:
            ax.text(bar.get_x() + bar.get_width()/2, 0, "NA", ha="center", va="bottom")
    ax.set_xticks(range(len(rows)), [row["Experiment"] for row in rows])
    ax.set_ylabel("Primary-rare mask AP"); ax.set_title("Strategy C controlled comparison")
    fig.tight_layout(); fig.savefig(figures / "rare_ap_strategy_c.png", dpi=180); plt.close(fig)

    def delta(left, right, metric="rare_mask_AP"):
        a = nested(all_results[left], "test_metrics", metric)
        b = nested(all_results[right], "test_metrics", metric)
        return None if a is None or b is None else b - a
    comparisons = [("E8 multi-room copy-paste → E10 diffusion inpainting", "e8", "e10"),
                   ("E9 feather blend → E10 diffusion inpainting", "e9", "e10"),
                   ("E10 uniform → E11 rare-aware sampling", "e10", "e11"),
                   ("E0 original → E10 diffusion inpainting", "e0", "e10")]
    lines = ["# Strategy C result summary", "",
             "This report is isolated from `outputs/final`; completed E0-E7 results remain unchanged.", "",
             "## Controlled comparisons", ""]
    for label, left, right in comparisons:
        change = delta(left, right)
        lines.append(f"- {label}: " + ("pending." if change is None else f"rare mask AP change {change:+.4f}."))
    lines += ["", "E8 and E9 are non-diffusion controls. E10 isolates exemplar-seeded inpainting; "
              "E11 tests whether its result is additive with rare-aware sampling. All generated pixels "
              "and both donor and target provenance are restricted to the frozen training split."]
    (output / "result_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    strategy_c_ids = ["e8", "e9", "e10", "e11"]
    print(json.dumps({
        "existing_e0_multiclass_available": result("multiclass", "e0") is not None,
        "existing_e0_agnostic_available": result("agnostic", "e0") is not None,
        "completed_strategy_c_multiclass": sum(
            result("multiclass", i) is not None for i in strategy_c_ids),
        "completed_strategy_c_agnostic": sum(
            result("agnostic", i) is not None for i in strategy_c_ids),
    }, indent=2))


if __name__ == "__main__":
    main()
