#!/usr/bin/env python3
"""Create split-specific COCO views for the official SAM 3 trainer."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _ids(path: Path) -> set[int]:
    return {int(value) for value in path.read_text(encoding="utf-8").splitlines() if value.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/processed/instances_deduplicated.json")
    parser.add_argument("--images", type=Path, default=ROOT.parent / "area_segmentation_coco/images")
    parser.add_argument("--splits", type=Path, default=ROOT / "data/splits")
    parser.add_argument("--output", type=Path, default=ROOT / "data/sam3_floorplan")
    args = parser.parse_args()

    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    images_by_id = {int(image["id"]): image for image in coco["images"]}
    split_ids = {name: _ids(args.splits / f"{name}.txt") for name in ("train", "val", "test")}
    if any(split_ids[a] & split_ids[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("SAM 3 export aborted: train/val/test image IDs overlap")

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split, allowed in split_ids.items():
        unknown = allowed - images_by_id.keys()
        if unknown:
            raise ValueError(f"{split} contains unknown image IDs: {sorted(unknown)}")
        split_dir = args.output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        selected_images = [images_by_id[image_id] for image_id in sorted(allowed)]
        selected_annotations = [ann for ann in coco["annotations"] if int(ann["image_id"]) in allowed]

        for image in selected_images:
            source = (args.images / image["file_name"]).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = split_dir / image["file_name"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_symlink() and destination.resolve() == source:
                continue
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            destination.symlink_to(source)

        exported = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "images": selected_images,
            "annotations": selected_annotations,
            # SAM 3's COCO loader uses category names as text concept prompts.
            "categories": coco["categories"],
        }
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(exported, ensure_ascii=False), encoding="utf-8"
        )
        summary[split] = {"images": len(selected_images), "annotations": len(selected_annotations)}

    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
