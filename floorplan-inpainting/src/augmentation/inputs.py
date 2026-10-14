import json

from src.data.coco_dataset import read_split
from src.utils.config import load_yaml, resolve_path


def load_inputs(config):
    dataset = config["dataset"]
    # Phase 1 config and frozen split files are the authority for generation inputs.
    canonical = load_yaml(resolve_path("configs/dataset.yaml"))
    annotation_path = resolve_path(dataset["annotation_file"]).resolve()
    if annotation_path != resolve_path(canonical["annotation_file"]).resolve():
        raise ValueError("Augmentation must use the Phase 1 deduplicated annotation file")
    split_dir = resolve_path(canonical["split_dir"])
    if resolve_path(dataset["train_split"]).resolve() != (split_dir / "train.txt").resolve():
        raise ValueError("Augmentation must use the frozen training split")
    train = set(read_split(split_dir / "train.txt"))
    val = set(read_split(split_dir / "val.txt"))
    test = set(read_split(split_dir / "test.txt"))
    if train & val or train & test or val & test:
        raise ValueError("Phase 1 split files overlap")
    manifest = json.loads(resolve_path("outputs/splits/split_manifest.json").read_text())
    for name, ids in [("train", train), ("val", val), ("test", test)]:
        if ids != set(manifest["image_ids"][name]):
            raise ValueError("Split files do not match Phase 1 manifest")
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    if train | val | test != {im["id"] for im in coco["images"]}:
        raise ValueError("Frozen splits do not cover the dataset exactly")
    if resolve_path(dataset["image_dir"]).resolve() != resolve_path(canonical["image_dir"]).resolve():
        raise ValueError("Source image directory differs from Phase 1")
    return (
        coco,
        load_yaml(resolve_path(dataset["rare_classes_file"])),
        train,
        resolve_path(dataset["image_dir"]),
    )


def original_train(coco, train):
    return {
        "images": [i for i in coco["images"] if i["id"] in train],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in train],
        "categories": coco["categories"],
    }
