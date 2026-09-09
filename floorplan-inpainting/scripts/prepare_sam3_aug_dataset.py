#!/usr/bin/env python3
"""Build a SAM 3 COCO dataset whose TRAIN split is an augmented set.

train/  <- one of the augmented training COCOs (real 35 + synthetic), file
           names flattened to basenames and symlinked from the augmentation dir
val/    <- frozen real val split   (data/splits/val.txt  over instances_deduplicated.json)
test/   <- frozen real test split  (data/splits/test.txt over instances_deduplicated.json)

So training sees the augmentation and evaluation stays on untouched real images -
the same contract as the Mask R-CNN experiment matrix, now with a SAM 3 backbone.
"""
import argparse
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

AUG = {
    "strategy_c_inpaint": "data/augmented_c/diffusion_inpaint/instances_train_aug_c.json",
    "lora_b": "data/augmented_lora_b/instances_train_augmented.json",
    "copy_paste": "data/augmented_c/copy_paste_multi/instances_train_augmented.json",
}


def _ids(path: Path):
    return {int(v) for v in path.read_text(encoding="utf-8").split()}


def _to_class_agnostic(coco: dict) -> dict:
    coco["categories"] = [{"id": 1, "name": "room", "supercategory": "room"}]
    for ann in coco["annotations"]:
        ann["category_id"] = 1
    coco["info"] = {**coco.get("info", {}), "class_agnostic": True}
    return coco


def _link(dest: Path, source: Path):
    if not source.is_file():
        raise FileNotFoundError(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() and dest.resolve() == source.resolve():
        return
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    dest.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aug", required=True, choices=sorted(AUG))
    parser.add_argument("--class-agnostic", action="store_true")
    parser.add_argument("--real-annotations", type=Path,
                        default=ROOT / "data/processed/instances_deduplicated.json")
    parser.add_argument("--real-images", type=Path,
                        default=ROOT.parent / "area_segmentation_coco/images")
    parser.add_argument("--splits", type=Path, default=ROOT / "data/splits")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    aug_json = ROOT / AUG[args.aug]
    aug_dir = aug_json.parent  # file_names are like "full_images/xxx.png" under here
    out = args.out or (ROOT / f"data/sam3_{args.aug}{'_agnostic' if args.class_agnostic else ''}")
    out.mkdir(parents=True, exist_ok=True)

    real = json.loads(args.real_annotations.read_text(encoding="utf-8"))
    aug = json.loads(aug_json.read_text(encoding="utf-8"))
    if [c["name"] for c in sorted(real["categories"], key=lambda c: c["id"])] != \
       [c["name"] for c in sorted(aug["categories"], key=lambda c: c["id"])]:
        raise ValueError("augmented and real category lists differ")
    if args.class_agnostic:
        real = _to_class_agnostic(real)
        aug = _to_class_agnostic(aug)

    summary = {}

    # --- train: the augmented set, basenames symlinked from the augmentation dir
    train_dir = out / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    train_images = []
    for info in aug["images"]:
        src = (aug_dir / info["file_name"]).resolve()
        base = Path(info["file_name"]).name
        _link(train_dir / base, src)
        width, height = info.get("width"), info.get("height")
        if not width or not height:
            with Image.open(src) as im:
                width, height = im.size
        # keep only standard COCO image fields for the SAM 3 loader
        train_images.append({"id": int(info["id"]), "file_name": base,
                             "width": int(width), "height": int(height)})
    train_image_ids = {int(i["id"]) for i in train_images}
    train_anns = [a for a in aug["annotations"] if int(a["image_id"]) in train_image_ids]
    (train_dir / "_annotations.coco.json").write_text(json.dumps({
        "info": aug.get("info", {}), "licenses": aug.get("licenses", []),
        "images": train_images, "annotations": train_anns, "categories": aug["categories"],
    }, ensure_ascii=False), encoding="utf-8")
    summary["train"] = {"images": len(train_images), "annotations": len(train_anns)}

    # --- val / test: frozen real splits over the real (optionally agnostic) COCO
    real_by_id = {int(i["id"]): i for i in real["images"]}
    for split in ("val", "test"):
        allowed = _ids(args.splits / f"{split}.txt")
        unknown = allowed - real_by_id.keys()
        if unknown:
            raise ValueError(f"{split} has unknown image IDs: {sorted(unknown)}")
        split_dir = out / split
        split_dir.mkdir(parents=True, exist_ok=True)
        images = [real_by_id[i] for i in sorted(allowed)]
        for info in images:
            _link(split_dir / info["file_name"], (args.real_images / info["file_name"]).resolve())
        anns = [a for a in real["annotations"] if int(a["image_id"]) in allowed]
        (split_dir / "_annotations.coco.json").write_text(json.dumps({
            "info": real.get("info", {}), "licenses": real.get("licenses", []),
            "images": images, "annotations": anns, "categories": real["categories"],
        }, ensure_ascii=False), encoding="utf-8")
        summary[split] = {"images": len(images), "annotations": len(anns)}

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "class_agnostic": args.class_agnostic, **summary}, indent=2))


if __name__ == "__main__":
    main()
