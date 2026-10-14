"""Aggregate Phase 3 outputs without inventing results for unfinished runs."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from src.data.coco_dataset import read_split
from src.utils.config import resolve_path, split_dir

METHODS = {
    "e0": ("Original", "No", "No", "No", "No"),
    "e1": ("Oversampling", "No", "No", "Yes", "No"),
    "e2": ("Strategy A", "Yes", "No", "No", "No"),
    "e3": ("Strategy A + weighted sampling", "Yes", "No", "Yes", "No"),
    "e4": ("Strategy A + rare-centered crops", "Yes", "No", "No", "Yes"),
    "e5": ("Strategy A + weighted sampling + rare-centered crops", "Yes", "No", "Yes", "Yes"),
    "e6": ("Strategy B", "Yes", "Yes", "No", "No"),
    "e7": ("Strategy B + weighted sampling", "Yes", "Yes", "Yes", "No"),
}


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _result(root, track, experiment_id, seed=42):
    path = root / track / experiment_id / f"seed_{seed}" / "result.json"
    return json.loads(path.read_text()) if path.exists() else None


def _value(result, *keys):
    value = result
    for key in keys:
        if value is None:
            return None
        value = value.get(key)
    return value


def _plot_bar(path, labels, values, title, ylabel):
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.15), 4.8))
    positions = np.arange(len(labels))
    display = [0 if value is None else value for value in values]
    bars = ax.bar(positions, display, color="#4878a8")
    for bar, value in zip(bars, values):
        if value is None:
            ax.text(bar.get_x() + bar.get_width() / 2, 0, "NA", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _metadata_records(path):
    latest = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[row["sample_id"]] = row
    return list(latest.values())


def _copy_qa_example(records, accepted, output):
    row = next(
        (row for row in records if bool(row.get("accepted")) == accepted and row.get("qa_row_path")), None
    )
    if row and Path(row["qa_row_path"]).exists():
        Image.open(row["qa_row_path"]).convert("RGB").save(output)


def _prediction_montage(experiment_root, best_a, output):
    methods = ["e0", "e1", best_a, "e6"]
    panels = []
    for experiment_id in methods:
        paths = sorted(
            (experiment_root / "multiclass" / experiment_id / "seed_42" / "predictions/comparisons").glob(
                "comparison_*.jpg"
            )
        )
        if paths:
            image = Image.open(paths[0]).convert("RGB")
            image.thumbnail((700, 450))
            panels.append((METHODS[experiment_id][0], image.copy()))
    if len(panels) < 2:
        return
    width, height = max(i.width for _, i in panels), max(i.height for _, i in panels) + 35
    canvas = Image.new("RGB", (width, height * len(panels)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(panels):
        y = index * height
        draw.text((8, y + 5), title, fill="black")
        canvas.paste(image, (0, y + 30))
    canvas.save(output)


def generate_final_report(experiment_root="outputs/experiments", output_dir="outputs/final"):
    experiment_root, output_dir = resolve_path(experiment_root), resolve_path(output_dir)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    original = json.loads(resolve_path("data/processed/instances_deduplicated.json").read_text())
    train_ids, test_ids = (set(read_split(split_dir() / f"{name}.txt")) for name in ("train", "test"))
    categories = {c["id"]: c["name"] for c in original["categories"]}
    train_counts = Counter(a["category_id"] for a in original["annotations"] if a["image_id"] in train_ids)
    test_counts = Counter(a["category_id"] for a in original["annotations"] if a["image_id"] in test_ids)
    class_counts_path = resolve_path("outputs/augmentation_a/class_counts.csv")
    class_count_rows = list(csv.DictReader(class_counts_path.open())) if class_counts_path.exists() else []

    main_rows, agnostic_rows = [], []
    results = {}
    for experiment_id, method in METHODS.items():
        result = _result(experiment_root, "multiclass", experiment_id)
        agnostic = _result(experiment_root, "agnostic", experiment_id)
        results[experiment_id] = result
        common = {
            "Method": method[0],
            "Synthetic Appearance": method[1],
            "New Geometry": method[2],
            "Rare-aware Sampling": method[3],
            "Rare-centered Crops": method[4],
        }
        main_rows.append(
            {
                **common,
                "Mask AP": _value(result, "test_metrics", "mask", "AP"),
                "Mask AP50": _value(result, "test_metrics", "mask", "AP50"),
                "Rare AP": _value(result, "test_metrics", "rare_mask_AP"),
                "Rare AP50": _value(result, "test_metrics", "rare_mask_AP50"),
            }
        )
        agnostic_rows.append(
            {
                **common,
                "Mask AP": _value(agnostic, "test_metrics", "mask", "AP"),
                "Mask AP50": _value(agnostic, "test_metrics", "mask", "AP50"),
                "Mask AP75": _value(agnostic, "test_metrics", "mask", "AP75"),
                "BBox AP": _value(agnostic, "test_metrics", "bbox", "AP"),
                "BBox AP50": _value(agnostic, "test_metrics", "bbox", "AP50"),
                "BBox AP75": _value(agnostic, "test_metrics", "bbox", "AP75"),
            }
        )
    main_fields = [
        "Method",
        "Synthetic Appearance",
        "New Geometry",
        "Rare-aware Sampling",
        "Rare-centered Crops",
        "Mask AP",
        "Mask AP50",
        "Rare AP",
        "Rare AP50",
    ]
    _write_csv(output_dir / "experiment_summary.csv", main_rows, main_fields)
    _write_csv(output_dir / "experiment_summary_agnostic.csv", agnostic_rows, list(agnostic_rows[0]))

    per_class = []
    for category_id, name in categories.items():
        row = {
            "ID": category_id,
            "Class": name,
            "Train Real": train_counts[category_id],
            "Test Real": test_counts[category_id],
        }
        values = []
        for experiment_id in METHODS:
            result = results[experiment_id]
            match = next(
                (r for r in (result or {}).get("per_class", []) if int(r["category_id"]) == category_id), None
            )
            value = None if match is None else match["mask_AP"]
            row[f"{experiment_id.upper()} AP"] = value
            values.append(value)
        valid = [value for value in values if value is not None]
        row["Best AP"] = max(valid) if valid else None
        row["Delta Best vs E0"] = (
            row["Best AP"] - values[0] if row["Best AP"] is not None and values[0] is not None else None
        )
        per_class.append(row)
    _write_csv(output_dir / "per_class_results.csv", per_class, list(per_class[0]))

    exposure_rows = []
    for track in ("multiclass", "agnostic"):
        for experiment_id, method in METHODS.items():
            path = experiment_root / track / experiment_id / "seed_42/class_exposure_by_epoch.csv"
            rows = list(csv.DictReader(path.open())) if path.exists() else []
            grouped = defaultdict(list)
            for row in rows:
                grouped[int(row["category_id"])].append(row)
            dataset_labels = {
                "e0": ["Original"],
                "e1": ["Original"],
                "e2": ["Strategy A full"],
                "e3": ["Strategy A full"],
                "e4": ["Strategy A full", "Strategy A crops"],
                "e5": ["Strategy A full", "Strategy A crops"],
                "e6": ["Strategy B automatic (semantics unverified)"],
                "e7": ["Strategy B automatic (semantics unverified)"],
            }[experiment_id]
            for category_id, name in categories.items():
                sources = [
                    r
                    for r in class_count_rows
                    if r["dataset"] in dataset_labels and int(r["category_id"]) == category_id
                ]
                seen = grouped.get(category_id, [])

                def average(field, seen=seen):
                    return sum(float(r[field]) for r in seen) / len(seen) if seen else None

                exposure_rows.append(
                    {
                        "Track": track,
                        "Method": method[0],
                        "Class": name,
                        "Original Train Instances": train_counts[category_id],
                        "Synthetic A Occurrences": sum(
                            int(r["synthetic_a_appearance_occurrences"]) for r in sources
                        ),
                        "Synthetic B Instances": sum(
                            int(r["new_synthetic_b_room_instances"]) for r in sources
                        ),
                        "Average Effective Instances Seen Per Epoch": average("total_occurrences_seen"),
                        "Test Instances": test_counts[category_id],
                        "Average Real Seen": average("real_occurrences_seen"),
                        "Average Synthetic A Seen": average("synthetic_A_occurrences_seen"),
                        "Average Synthetic B Seen": average("synthetic_B_occurrences_seen"),
                    }
                )
    _write_csv(output_dir / "class_exposure_summary.csv", exposure_rows, list(exposure_rows[0]))

    augmentation_rows = []
    strategy_records = {}
    for strategy in ("a", "b"):
        records = _metadata_records(resolve_path(f"data/augmented_{strategy}/metadata.jsonl"))
        strategy_records[strategy] = records
        for category in sorted({r.get("category_name") for r in records}):
            selected = [r for r in records if r.get("category_name") == category]
            accepted = [r for r in selected if r.get("accepted")]

            def mean(key, accepted=accepted):
                values = [float(r[key]) for r in accepted if r.get(key) is not None]
                return sum(values) / len(values) if values else None

            augmentation_rows.append(
                {
                    "Strategy": strategy.upper(),
                    "Class": category,
                    "Attempted": len(selected),
                    "Accepted": len(accepted),
                    "Rejected": len(selected) - len(accepted),
                    "Acceptance Rate": len(accepted) / len(selected),
                    "Generation Failure Rate": sum(r.get("status") == "error" for r in selected)
                    / len(selected),
                    "Mean Outside Change Ratio": mean("outside_change_ratio"),
                    "Mean Boundary Change Ratio": mean("protected_boundary_change_ratio"),
                    "Mean Mutable Fraction": mean("mutable_fraction"),
                }
            )
    _write_csv(output_dir / "augmentation_statistics.csv", augmentation_rows, list(augmentation_rows[0]))

    seed_rows = []
    for track in ("multiclass", "agnostic"):
        for experiment_id, method in METHODS.items():
            found = []
            for seed_dir in sorted((experiment_root / track / experiment_id).glob("seed_*")):
                path = seed_dir / "result.json"
                if path.exists():
                    found.append(json.loads(path.read_text()))
            for metric_name, keys in {
                "mask_AP": ("mask", "AP"),
                "mask_AP50": ("mask", "AP50"),
                "rare_mask_AP": (None, "rare_mask_AP"),
            }.items():
                values = []
                for result in found:
                    metrics = result["test_metrics"]
                    value = metrics.get(keys[1]) if keys[0] is None else metrics[keys[0]].get(keys[1])
                    if value is not None:
                        values.append(value)
                seed_rows.append(
                    {
                        "Track": track,
                        "Method": method[0],
                        "Metric": metric_name,
                        "Seeds Completed": len(values),
                        "Mean": np.mean(values) if values else None,
                        "Std": np.std(values, ddof=1) if len(values) > 1 else None,
                    }
                )
    _write_csv(output_dir / "seed_summary.csv", seed_rows, list(seed_rows[0]))

    _plot_bar(
        figures / "class_distribution_original.png",
        [categories[i] for i in categories],
        [train_counts[i] for i in categories],
        "Original training class distribution",
        "Instances",
    )
    primary = set(range(11, 17))
    method_exposure = []
    for method in [m[0] for m in METHODS.values()]:
        selected = [
            r
            for r in exposure_rows
            if r["Track"] == "multiclass"
            and r["Method"] == method
            and next(cid for cid, name in categories.items() if name == r["Class"]) in primary
        ]
        values = [r["Average Effective Instances Seen Per Epoch"] for r in selected]
        method_exposure.append(sum(values) if values and all(v is not None for v in values) else None)
    _plot_bar(
        figures / "effective_class_exposure_by_method.png",
        [m[0] for m in METHODS.values()],
        method_exposure,
        "Primary-rare exposure by method",
        "Occurrences seen per epoch",
    )
    _plot_bar(
        figures / "figure3_before_after_minority_exposure.png",
        [m[0] for m in METHODS.values()],
        method_exposure,
        "Before/after primary-rare exposure",
        "Occurrences seen per epoch",
    )
    _plot_bar(
        figures / "rare_AP_by_method.png",
        [r["Method"] for r in main_rows],
        [r["Rare AP"] for r in main_rows],
        "Primary-rare mask AP",
        "AP",
    )
    _plot_bar(
        figures / "rare_AP50_by_method.png",
        [r["Method"] for r in main_rows],
        [r["Rare AP50"] for r in main_rows],
        "Primary-rare mask AP50",
        "AP50",
    )
    _plot_bar(
        figures / "per_class_AP_delta.png",
        [r["Class"] for r in per_class if r["ID"] in primary],
        [r["Delta Best vs E0"] for r in per_class if r["ID"] in primary],
        "Best primary-rare AP change from E0",
        "AP difference",
    )
    _plot_bar(
        figures / "strategy_a_vs_oversampling.png",
        [main_rows[i]["Method"] for i in (1, 2, 3, 4, 5)],
        [main_rows[i]["Rare AP"] for i in (1, 2, 3, 4, 5)],
        "Strategy A versus oversampling",
        "Rare AP",
    )
    _plot_bar(
        figures / "strategy_a_vs_strategy_b.png",
        [main_rows[i]["Method"] for i in (3, 5, 6, 7)],
        [main_rows[i]["Rare AP"] for i in (3, 5, 6, 7)],
        "Appearance versus new geometry",
        "Rare AP",
    )
    _plot_bar(
        figures / "synthetic_acceptance_by_class.png",
        [f"{r['Strategy']} {r['Class']}" for r in augmentation_rows],
        [r["Acceptance Rate"] for r in augmentation_rows],
        "Synthetic acceptance by class",
        "Rate",
    )
    _copy_qa_example(strategy_records["a"], True, figures / "figure1_strategy_a_pipeline.png")
    _copy_qa_example(strategy_records["b"], True, figures / "figure2_strategy_b_pipeline.png")
    rejected = [r for values in strategy_records.values() for r in values if not r.get("accepted")]
    _copy_qa_example(rejected, False, figures / "figure5_failure_case.png")
    complete_a = [(i, _value(results[i], "test_metrics", "rare_mask_AP")) for i in ("e2", "e3", "e4", "e5")]
    complete_a = [(i, v) for i, v in complete_a if v is not None]
    best_a = max(complete_a, key=lambda item: item[1])[0] if complete_a else "e5"
    _prediction_montage(experiment_root, best_a, figures / "figure4_test_predictions.png")

    completed = sum(result is not None for result in results.values())
    lines = [
        "# Phase 3 result summary",
        "",
        f"Completed multiclass single-seed experiments: {completed}/8. "
        "Missing results remain `NA`; no claim is inferred from an unfinished run.",
        "",
        "## Scientific interpretation",
        "",
        "- Oversampling changes effective frequency, but not appearance or geometry diversity.",
        "- Strategy A changes appearance within existing real polygons; "
        "its synthetic occurrences are not independent geometries.",
        "- Weighted/crop Strategy A regimes increase minority exposure "
        "while retaining the same underlying rare-room geometries.",
        "- Strategy B adds converted polygon geometries, "
        "but its semantic correctness is automatically screened and remains unverified.",
        "- The class-agnostic track measures room instance segmentation independent of room labels. "
        "Rare-class AP is not defined there; source-category exposure remains logged for comparison.",
        "",
        "## Required comparisons",
        "",
    ]
    comparisons = [
        ("RQ1 / E0 vs E1", "e0", "e1"),
        ("RQ4 / E0 vs E2", "e0", "e2"),
        ("RQ2 / E1 vs E3", "e1", "e3"),
        ("E2 vs E3", "e2", "e3"),
        ("RQ3 / E3 vs E5", "e3", "e5"),
        ("RQ5 / E5 vs E6", "e5", "e6"),
    ]
    for label, left, right in comparisons:
        a, b = (
            _value(results[left], "test_metrics", "rare_mask_AP"),
            _value(results[right], "test_metrics", "rare_mask_AP"),
        )
        lines.append(
            f"- {label}: "
            + (f"rare mask AP change {b - a:+.4f}." if a is not None and b is not None else "pending.")
        )
    lines += [
        "",
        "The primary rare aggregate is an unweighted macro average over classes 11–16 "
        "with nonzero test support. "
        "Individual AP values must be interpreted alongside support in `per_class_results.csv`.",
    ]
    (output_dir / "result_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "completed_multiclass": completed,
        "completed_agnostic": sum(_result(experiment_root, "agnostic", i) is not None for i in METHODS),
    }
