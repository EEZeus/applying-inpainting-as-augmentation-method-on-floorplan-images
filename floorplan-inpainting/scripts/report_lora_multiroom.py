#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def slug(v):
    return f"{v:g}".replace(".", "p")


def load(path):
    return json.loads(path.read_text()) if path.is_file() else None


def metric(result, key):
    return (result or {}).get("test_metrics", {}).get(key)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replacement-percentage", type=float, default=5)
    p.add_argument("--percentage-jitter", type=float, default=2)
    a = p.parse_args()
    variant = f"pct_{slug(a.replacement_percentage)}_jitter_{slug(a.percentage_jitter)}"
    out = ROOT / "outputs/final_lora_multi" / variant
    out.mkdir(parents=True, exist_ok=True)
    specs = [
        ("e0", "Original", ROOT / "outputs/experiments"),
        ("e12", "Single-room LoRA", ROOT / "outputs/experiments_lora"),
        (
            "e14",
            f"Multi-room rare-exemplar LoRA {a.replacement_percentage}% ± {a.percentage_jitter} pp",
            ROOT / "outputs/experiments_lora_multi_semantic_v2" / variant,
        ),
        (
            "e15",
            "Multi-room rare-exemplar LoRA + rare weighting",
            ROOT / "outputs/experiments_lora_multi_semantic_v2" / variant,
        ),
    ]
    rows = []
    for eid, name, base in specs:
        result = load(base / "multiclass" / eid / "seed_42/result.json")
        rows.append(
            {
                "Experiment": eid.upper(),
                "Method": name,
                "Mask AP": metric(result, "mask").get("AP") if metric(result, "mask") else None,
                "Rare AP": metric(result, "rare_mask_AP"),
                "Rare AP50": metric(result, "rare_mask_AP50"),
            }
        )
    with (out / "experiment_summary.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    manifest = load(ROOT / "data/augmented_lora_multi_semantic_v2" / variant / "multiroom_manifest.json")
    lines = [
        "# Multi-room LoRA result summary",
        "",
        f"Requested replacement policy: {a.replacement_percentage}% with uniform "
        f"±{a.percentage_jitter} percentage-point jitter.",
        "",
        "Every conversion is common (1–10) to rare (11–16). "
        "Geometry/QC limits may lower the achieved percentage.",
        "",
    ]
    for row in rows:
        lines.append(
            f"- {row['Experiment']} rare AP: {row['Rare AP'] if row['Rare AP'] is not None else 'pending'}"
        )
    (out / "result_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "variant": variant,
                "dataset_ready": manifest is not None,
                "completed": sum(r["Rare AP"] is not None for r in rows[-2:]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
