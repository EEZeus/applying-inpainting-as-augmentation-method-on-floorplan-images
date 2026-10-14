import csv
import hashlib
import json
from pathlib import Path

from .grouping import image_to_project

DUPLICATE_GROUPS = ((17, 22), (19, 24), (20, 25, 26))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deduplicate_coco(annotation_file, image_dir, output_file, manifest_file):
    with Path(annotation_file).open(encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {image["id"]: image for image in coco["images"]}
    projects = image_to_project()
    removed = set()
    rows = []
    for group in DUPLICATE_GROUPS:
        canonical, *duplicates = group
        hashes = {sha256(Path(image_dir) / images[i]["file_name"]) for i in group}
        if len(hashes) != 1:
            raise ValueError(f"Expected byte-identical images in group {group}")
        removed.update(duplicates)
        rows.append(
            {
                "canonical_image_id": canonical,
                "duplicate_image_ids": ";".join(map(str, duplicates)),
                "filename": images[canonical]["file_name"],
                "project": projects[canonical],
                "reason": "exact_sha256_duplicate",
            }
        )

    cleaned = dict(coco)
    cleaned["images"] = [im for im in coco["images"] if im["id"] not in removed]
    cleaned["annotations"] = [ann for ann in coco["annotations"] if ann["image_id"] not in removed]
    output_file, manifest_file = Path(output_file), Path(manifest_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(cleaned, handle, ensure_ascii=False)
    with manifest_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return {
        "images_before": len(coco["images"]),
        "images_after": len(cleaned["images"]),
        "annotations_before": len(coco["annotations"]),
        "annotations_after": len(cleaned["annotations"]),
        "removed_image_ids": sorted(removed),
    }
