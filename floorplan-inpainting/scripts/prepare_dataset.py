#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.deduplication import deduplicate_coco
from src.data.geometry import shape_statistics
from src.data.grouping import validate_groups


def audit(annotation_file, image_dir):
    coco = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    facts = {"images": len(coco["images"]), "annotations": len(coco["annotations"]),
             "categories": len(coco["categories"]), "category_ids": sorted(c["id"] for c in coco["categories"]),
             "all_polygon": all(isinstance(a["segmentation"], list) for a in coco["annotations"]),
             "missing_files": [im["file_name"] for im in coco["images"] if not (Path(image_dir) / im["file_name"]).is_file()]}
    validate_groups(im["id"] for im in coco["images"])
    expected = {"images": 60, "annotations": 1709, "categories": 19, "category_ids": list(range(1, 20)),
                "all_polygon": True, "missing_files": []}
    differences = {k: {"expected": expected[k], "actual": facts[k]} for k in expected if facts[k] != expected[k]}
    return facts, differences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default=ROOT.parent / "area_segmentation_coco/annotations/instances.json")
    parser.add_argument("--images", default=ROOT.parent / "area_segmentation_coco/images")
    args = parser.parse_args()
    facts, differences = audit(args.annotations, args.images)
    report = deduplicate_coco(args.annotations, args.images, ROOT / "data/processed/instances_deduplicated.json",
                              ROOT / "outputs/dataset/exact_duplicate_manifest.csv")
    shape_statistics(ROOT / "data/processed/instances_deduplicated.json", ROOT / "outputs/dataset/room_shape_statistics.csv")
    audit_report = {"verified": facts, "differences_from_known_facts": differences, "deduplication": report}
    (ROOT / "outputs/dataset/audit_report.json").write_text(json.dumps(audit_report, indent=2), encoding="utf-8")
    print(json.dumps(audit_report, indent=2))


if __name__ == "__main__": main()

