#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augmentation.inputs import load_inputs
from src.augmentation.pipeline import plan_candidates, prepare_candidate
from src.utils.config import load_yaml, resolve_path


def main():
    parser = argparse.ArgumentParser(description="Preview Strategy A room and safe masks without a GPU")
    parser.add_argument("--config", default=ROOT / "configs/augmentation_a.yaml")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    config = load_yaml(args.config)
    coco, rare, train_ids, image_dir = load_inputs(config)
    candidates, _ = plan_candidates(coco, train_ids, rare, config)
    unique = []
    seen = set()
    for candidate in candidates:
        if candidate.target_annotation_id not in seen:
            unique.append(candidate)
            seen.add(candidate.target_annotation_id)
    images = {int(item["id"]): item for item in coco["images"]}
    annotations, annotation_by_id = {}, {}
    for image_id in images:
        annotations[image_id] = []
    for annotation in coco["annotations"]:
        annotations[int(annotation["image_id"])].append(annotation)
        annotation_by_id[int(annotation["id"])] = annotation
    indices = {"images": images, "annotations": annotations, "annotation_by_id": annotation_by_id}
    output_dir = resolve_path(config["output"]["qa_dir"], ROOT) / "mask_previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    for candidate in unique[: args.limit]:
        item = prepare_candidate(candidate, indices, image_dir, config)
        result, transform = item.mask_result, item.transform
        sl = (slice(transform.y0, transform.y1), slice(transform.x0, transform.x1))
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
        for axis, value, title in zip(
            axes,
            [item.source_crop, result.room_mask[sl], result.structure_mask[sl], result.safe_mask[sl]],
            ["Source crop", "Room mask", "Protected structure", "Safe mask"],
        ):
            axis.imshow(value, cmap=None if value.ndim == 3 else "gray")
            axis.set_title(title)
            axis.axis("off")
        fig.suptitle(
            f"ann={candidate.target_annotation_id} {candidate.category_name} "
            f"erosion={result.erosion_pixels}px mutable={result.mutable_fraction:.3f}"
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"mask_ann_{candidate.target_annotation_id}.png", dpi=120)
        plt.close(fig)
    print(f"Wrote {min(args.limit, len(unique))} previews to {output_dir}")


if __name__ == "__main__":
    main()
