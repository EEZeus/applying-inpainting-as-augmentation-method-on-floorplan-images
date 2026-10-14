"""Automatic acceptance audit for experimental Strategy B (no human gate)."""

import csv
import json
import os
from pathlib import Path

from .pipeline import (
    Candidate,
    _atomic_json,
    build_augmented_coco,
    file_digest,
    latest_metadata,
    read_metadata,
    run_fingerprint,
)

FIELDS = [
    "sample_id",
    "source_image_id",
    "annotation_id",
    "source_category",
    "target_category",
    "source_geometry",
    "target_geometry_range",
    "seed",
    "image_path",
    "image_sha256",
    "status",
    "decision_method",
    "semantic_correctness",
    "notes",
]


def read_review(path):
    if not Path(path).exists():
        return {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs in review sheet")
    if any(r["status"] not in {"approved", "rejected"} for r in rows):
        raise ValueError("Automatic audit status must be approved or rejected")
    return {r["sample_id"]: r for r in rows}


def build_review(dataset_dir, path):
    dataset_dir, path = Path(dataset_dir), Path(path)
    records = latest_metadata(read_metadata(dataset_dir / "metadata.jsonl"))
    rows = []
    for record in records.values():
        accepted = record["status"] == "accepted"
        rows.append(
            {
                "sample_id": record["sample_id"],
                "source_image_id": record["source_image_id"],
                "annotation_id": record["target_annotation_id"],
                "source_category": record["source_category_id"],
                "target_category": record["category_id"],
                "source_geometry": json.dumps(record["source_geometry"]),
                "target_geometry_range": json.dumps(record["target_geometry_range"]),
                "seed": record["seed"],
                "image_path": str((dataset_dir / record["output_file"]).resolve())
                if accepted
                else record.get("raw_generated_path", ""),
                "image_sha256": record.get("output_sha256", ""),
                "status": "approved" if accepted else "rejected",
                "decision_method": "automatic_geometry_and_pixel_qc",
                "semantic_correctness": "unverified",
                "notes": "Experimental class conversion; no human approval required"
                if accepted
                else record.get("rejection_reason", "QC failed"),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return rows


def apply_review(dataset_dir, review_path, coco, train_ids, image_dir):
    dataset_dir = Path(dataset_dir)
    build_review(dataset_dir, review_path)
    rows = read_review(review_path)
    manifest = json.loads((dataset_dir / "run_manifest.json").read_text())
    candidates = [Candidate(**c) for c in manifest["candidates"]]
    known = {c.sample_id: c for c in candidates}
    records = latest_metadata(read_metadata(dataset_dir / "metadata.jsonl"))
    approved = []
    for key, row in rows.items():
        if key not in known or key not in records:
            raise ValueError(f"Unknown review sample {key}")
        if row["status"] != "approved":
            continue
        record, candidate = records[key], known[key]
        if record["status"] != "accepted" or candidate.strategy != "b":
            raise ValueError("Only QC-passing Strategy B candidates can be approved")
        if (
            int(row["target_category"]) != candidate.category_id
            or int(row["annotation_id"]) != candidate.target_annotation_id
            or int(row["source_image_id"]) != candidate.source_image_id
            or int(row["seed"]) != candidate.seed
        ):
            raise ValueError("Review provenance was modified")
        image = dataset_dir / record["output_file"]
        if file_digest(image) != row["image_sha256"] or row["image_sha256"] != record["output_sha256"]:
            raise ValueError("Approved image hash mismatch")
        approved.append(record)
    # Recheck the original inputs used by the generation run.
    cfg = manifest["config"]
    cfg["input_hashes"] = {
        str(i["id"]): file_digest(Path(image_dir) / i["file_name"])
        for i in coco["images"]
        if i["id"] in train_ids
    }
    if run_fingerprint(coco, candidates, manifest["annotation_stride"], cfg) != manifest["fingerprint"]:
        raise ValueError("Generation sources changed since review")
    output = build_augmented_coco(
        coco,
        train_ids,
        candidates,
        approved,
        image_dir,
        dataset_dir,
        manifest["annotation_stride"],
        "instances_train_aug_b.json",
    )
    output["info"] = {
        **output.get("info", {}),
        "acceptance_method": "automatic_geometry_and_pixel_qc",
        "semantic_correctness": "unverified",
        "experimental": True,
    }
    _atomic_json(dataset_dir / "instances_train_aug_b.json", output)
    # Compatibility with the previous planned filename: approved now means automatic QC.
    _atomic_json(dataset_dir / "instances_train_aug_b_approved.json", output)
    total = len(rows)
    decided = total
    _atomic_json(
        Path(review_path).parent / "review_summary.json",
        {
            "candidates": total,
            "approved": len(approved),
            "decided": decided,
            "rejected": total - len(approved),
            "pending": total - decided,
            "approval_rate": len(approved) / decided if decided else None,
            "decision_method": "automatic_geometry_and_pixel_qc",
            "semantic_correctness": "unverified",
        },
    )
    return output
