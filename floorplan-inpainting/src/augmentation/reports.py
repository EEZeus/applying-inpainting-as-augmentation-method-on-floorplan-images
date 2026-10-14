import csv
from collections import Counter
from pathlib import Path


def class_count_report(path, original_train, datasets):
    real = Counter(a["category_id"] for a in original_train["annotations"])
    source_annotations = {}
    for annotation in original_train["annotations"]:
        source_annotations.setdefault(annotation["image_id"], []).append(annotation)
    for annotations in source_annotations.values():
        annotations.sort(key=lambda item: item["id"])

    def target_annotation_ids(coco, images):
        """Return the IDs of the one changed annotation in each synthetic image.

        Full-plan copies keep the source annotation order; crops keep ``source_annotation_id``.
        """
        by_image = {}
        for annotation in coco["annotations"]:
            by_image.setdefault(annotation["image_id"], []).append(annotation)
        targets = set()
        for image_id, annotations in by_image.items():
            info = images[image_id]
            if info.get("strategy") not in {"a", "b"}:
                continue
            target_id = info.get("target_annotation_id")
            crop_match = next(
                (
                    annotation
                    for annotation in annotations
                    if annotation.get("source_annotation_id") == target_id
                ),
                None,
            )
            if crop_match is not None:
                targets.add(crop_match["id"])
                continue

            originals = source_annotations.get(info.get("source_image_id"), [])
            target_positions = [
                index for index, annotation in enumerate(originals) if annotation["id"] == target_id
            ]
            ordered = sorted(annotations, key=lambda item: item["id"])
            if len(target_positions) != 1 or len(ordered) != len(originals):
                raise ValueError(f"Cannot resolve target annotation provenance for image {image_id}")
            targets.add(ordered[target_positions[0]]["id"])
        return targets

    rows = []
    for dataset_name, coco in {"Original": original_train, **datasets}.items():
        images = {im["id"]: im for im in coco["images"]}
        total = Counter(a["category_id"] for a in coco["annotations"])
        unchanged_base = Counter(
            annotation["category_id"]
            for annotation in coco["annotations"]
            if not images[annotation["image_id"]].get("strategy")
        )
        deliberate_targets = target_annotation_ids(coco, images)
        a_occurrences, b_instances = Counter(), Counter()
        for ann in coco["annotations"]:
            info = images[ann["image_id"]]
            if ann["id"] not in deliberate_targets:
                continue
            if info.get("strategy") == "a":
                a_occurrences[ann["category_id"]] += 1
            elif info.get("strategy") == "b":
                b_instances[ann["category_id"]] += 1
        for category in coco["categories"]:
            cid = category["id"]
            rows.append(
                {
                    "dataset": dataset_name,
                    "category_id": cid,
                    "class": category["name"],
                    "real_instances": real[cid],
                    "synthetic_a_appearance_occurrences": a_occurrences[cid],
                    "new_synthetic_b_room_instances": b_instances[cid],
                    "effective_training_occurrences": total[cid],
                    "other_context_occurrences": total[cid]
                    - unchanged_base[cid]
                    - a_occurrences[cid]
                    - b_instances[cid],
                }
            )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
