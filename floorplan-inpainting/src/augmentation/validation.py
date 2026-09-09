"""Independent checks against original pixels and unchanged COCO annotations."""
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .masks import build_safe_mask
from .pipeline import file_digest, latest_metadata, read_metadata, validate_augmented_coco


def validate_full_provenance(dataset, source, source_dir, output_dir, train_ids):
    output_dir, source_dir = Path(output_dir), Path(source_dir)
    validate_augmented_coco(dataset, output_dir, train_ids)
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    config = manifest["config"]
    candidates = {c["synthetic_image_id"]: c for c in manifest["candidates"]}
    records = latest_metadata(read_metadata(output_dir / "metadata.jsonl"))
    originals = {im["id"]: im for im in source["images"]}
    source_anns, output_anns = defaultdict(list), defaultdict(list)
    for annotation in source["annotations"]:
        source_anns[annotation["image_id"]].append(annotation)
    for annotation in dataset["annotations"]:
        output_anns[annotation["image_id"]].append(annotation)
    original_pixels = {}
    verified = 0
    for info in dataset["images"]:
        source_id = info.get("source_image_id", info["id"])
        source_info = originals[source_id]
        source_path = source_dir / source_info["file_name"]
        actual_anns = sorted(output_anns[info["id"]], key=lambda a: a["id"])
        expected_anns = sorted(source_anns[source_id], key=lambda a: a["id"])
        if not info.get("synthetic"):
            if actual_anns != expected_anns or file_digest(output_dir / info["file_name"]) != file_digest(source_path):
                raise ValueError("Original training image/annotations were changed")
            continue
        candidate = candidates[info["id"]]
        record = records[candidate["sample_id"]]
        if record["status"] != "accepted":
            raise ValueError("QC-failed sample entered a dataset")
        expected = []
        for ordinal, ann in enumerate(expected_anns):
            duplicate = copy.deepcopy(ann)
            duplicate.update(id=candidate["annotation_id_start"] + ordinal, image_id=info["id"])
            if candidate["strategy"] in {"b", "c", "lora_b", "lora_multi"} and ann["id"] == candidate["target_annotation_id"]:
                duplicate["category_id"] = candidate["category_id"]
            expected.append(duplicate)
        if expected != actual_anns:
            raise ValueError("Synthetic annotations differ from permitted transformation")
        if file_digest(output_dir / info["file_name"]) != record["output_sha256"]:
            raise ValueError("Generated file hash changed")
        if source_id not in original_pixels:
            # Retain only one source image to bound full-plan memory use.
            original_pixels.clear()
            with Image.open(source_path) as handle:
                original_pixels[source_id] = np.asarray(handle.convert("RGB"))
        original = original_pixels[source_id]
        target = next(a for a in expected_anns if a["id"] == candidate["target_annotation_id"])
        mask = build_safe_mask(original, target, [a for a in expected_anns if a["id"] != target["id"]], config["mask"])
        with Image.open(output_dir / info["file_name"]) as handle:
            generated = np.asarray(handle.convert("RGB"))
        if not np.array_equal(original[~mask.safe_mask], generated[~mask.safe_mask]):
            raise ValueError("Pixels outside safe mask changed")
        if not np.array_equal(original[mask.room_mask & ~mask.eroded_mask], generated[mask.room_mask & ~mask.eroded_mask]):
            raise ValueError("Protected boundary pixels changed")
        verified += 1
    return {"verified_synthetic_images": verified, "outside_change_ratio": 0., "boundary_change_ratio": 0.}
