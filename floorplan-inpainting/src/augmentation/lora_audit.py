"""Automatic, tamper-evident decision audit for LoRA conversion candidates."""
import csv
import json
import os
from collections import Counter
from pathlib import Path

from .pipeline import Candidate, _atomic_json, build_augmented_coco, file_digest, latest_metadata, read_metadata, run_fingerprint

FIELDS = ["sample_id", "source_image_id", "annotation_id", "source_category", "target_category",
          "source_geometry", "target_geometry_range", "seed", "image_path", "image_sha256",
          "status", "decision_method", "semantic_correctness", "qc_failure_reason"]


def build_audit(dataset_dir, audit_path):
    """Derive decisions exclusively from immutable geometry/structural/pixel QC records."""
    dataset_dir, audit_path = Path(dataset_dir), Path(audit_path)
    records = latest_metadata(read_metadata(dataset_dir / "metadata.jsonl")); rows = []
    for record in sorted(records.values(), key=lambda r: r["candidate_index"]):
        accepted = record["status"] == "accepted"
        rows.append({"sample_id": record["sample_id"], "source_image_id": record["source_image_id"],
                     "annotation_id": record["target_annotation_id"], "source_category": record["source_category_id"],
                     "target_category": record["category_id"], "source_geometry": json.dumps(record["source_geometry"], sort_keys=True),
                     "target_geometry_range": json.dumps(record["target_geometry_range"], sort_keys=True), "seed": record["seed"],
                     "image_path": str((dataset_dir / record["output_file"]).resolve()) if accepted else record.get("raw_generated_path", ""),
                     "image_sha256": record.get("output_sha256", ""), "status": "accepted" if accepted else "rejected",
                     "decision_method": "automatic_geometry_pixel_and_structural_qc", "semantic_correctness": "unverified",
                     "qc_failure_reason": record.get("rejection_reason", "") or ""})
    audit_path.parent.mkdir(parents=True, exist_ok=True); temporary = audit_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, audit_path)
    _atomic_json(audit_path.parent / "decision_audit_summary.json", {"candidates": len(rows),
                 "automatically_accepted": sum(r["status"] == "accepted" for r in rows),
                 "automatically_rejected": sum(r["status"] == "rejected" for r in rows), "human_approval_used": False})
    return rows


def apply_automatic_decisions(dataset_dir, audit_path, coco, train_ids, image_dir, lora_manifest=None):
    """Integrate all and only candidates that passed the automatic checks."""
    dataset_dir = Path(dataset_dir); rows = {r["sample_id"]: r for r in build_audit(dataset_dir, audit_path)}
    manifest = json.loads((dataset_dir / "run_manifest.json").read_text(encoding="utf-8"))
    candidates = [Candidate(**c) for c in manifest["candidates"]]; known = {c.sample_id: c for c in candidates}
    records = latest_metadata(read_metadata(dataset_dir / "metadata.jsonl")); accepted_records = []
    for key, row in rows.items():
        candidate, record = known[key], records[key]
        if row["status"] != "accepted": continue
        if candidate.strategy != "lora_b" or record["status"] != "accepted": raise ValueError("Automatic audit disagrees with QC metadata")
        if file_digest(dataset_dir / record["output_file"]) != record["output_sha256"]: raise ValueError("Generated image hash mismatch")
        accepted_records.append(record)
    cfg = manifest["config"]
    cfg["input_hashes"] = {str(i["id"]): file_digest(Path(image_dir) / i["file_name"]) for i in coco["images"] if i["id"] in train_ids}
    if run_fingerprint(coco, candidates, manifest["annotation_stride"], cfg) != manifest["fingerprint"]: raise ValueError("Generation source or LoRA changed before integration")
    output = build_augmented_coco(coco, train_ids, candidates, accepted_records, image_dir, dataset_dir, manifest["annotation_stride"],
                                  filename="instances_train_augmented.json", include_originals=True)
    output["info"] = {**output.get("info", {}), "strategy": "lora_b", "decision_method": "automatic_geometry_pixel_and_structural_qc",
                      "human_approval_used": False, "semantic_correctness": "unverified", "automatically_accepted_synthetic_geometries": len(accepted_records),
                      "classical_variants_are_independent_geometries": False}
    _atomic_json(dataset_dir / "instances_train_augmented.json", output)
    real = Counter(a["category_id"] for a in coco["annotations"] if a["image_id"] in train_ids); variants = Counter()
    if lora_manifest and Path(lora_manifest).is_file():
        lm=json.loads(Path(lora_manifest).read_text()); variants.update(r["category_id"] for r in lm["records"] if r["is_classical_variant"])
    synthetic=Counter(r["category_id"] for r in accepted_records); names={c["id"]:c["name"] for c in coco["categories"]}
    with (dataset_dir/"before_after_class_counts.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.writer(handle); writer.writerow(["category_id","category_name","real_train_instances","classical_lora_crop_variants","automatically_accepted_synthetic_geometries","final_train_instances"])
        for cid in sorted(names): writer.writerow([cid,names[cid],real[cid],variants[cid],synthetic[cid],real[cid]+synthetic[cid]])
    summary={"automatically_accepted":len(accepted_records),"automatically_rejected":sum(r["status"]=="rejected" for r in rows.values()),
             "human_approval_used":False,"synthetic_by_class":{names[k]:v for k,v in sorted(synthetic.items())},
             "output_json":str(dataset_dir/"instances_train_augmented.json")}
    _atomic_json(Path(audit_path).parent/"integration_summary.json",summary); return summary
