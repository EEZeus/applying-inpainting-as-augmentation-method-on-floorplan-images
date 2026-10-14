"""LoRA-based, automatically QC-gated common-to-rare conversion branch."""

from dataclasses import replace
from pathlib import Path

from .pipeline import _atomic_json, run_augmentation
from .strategy_b import plan_strategy_b


def plan_lora_strategy_b(coco, train_ids, config):
    (planned, stride), diagnostics = plan_strategy_b(coco, train_ids, config)
    # Separate deterministic ID namespace from all earlier strategies.
    candidates = [
        replace(
            c,
            sample_id=f"lora-b-ann{c.target_annotation_id}-to{c.category_id}",
            strategy="lora_b",
            synthetic_image_id=c.synthetic_image_id + 5_000_000,
            annotation_id_start=c.annotation_id_start + 5_000_000_000,
        )
        for c in planned
    ]
    return (candidates, stride), diagnostics


def run_lora_strategy_b(config, coco, rare, train_ids, image_dir, output_dir, qa_dir, **kwargs):
    from src.utils.config import resolve_path

    from .lora_audit import apply_automatic_decisions

    plan, diagnostics = plan_lora_strategy_b(coco, train_ids, config)
    _atomic_json(Path(qa_dir) / "geometry_screening.json", diagnostics)
    result = run_augmentation(
        config,
        coco,
        rare,
        train_ids,
        image_dir,
        output_dir,
        qa_dir,
        candidate_plan=plan,
        strategy="lora_b",
        **kwargs,
    )
    integration = apply_automatic_decisions(
        output_dir,
        resolve_path(config["output"]["decision_audit_csv"]),
        coco,
        train_ids,
        image_dir,
        resolve_path(config["dataset"]["lora_training_manifest"]),
    )
    result.update(
        human_approval_used=False,
        decision_method="automatic_geometry_pixel_and_structural_qc",
        semantic_correctness="unverified",
        integration=integration,
        planned_candidates_per_target={v["category_name"]: v["planned"] for v in diagnostics.values()},
    )
    _atomic_json(Path(qa_dir) / "summary.json", result)
    return result
