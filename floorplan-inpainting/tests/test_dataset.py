from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")
from src.data.coco_dataset import FloorplanCocoDataset, read_split
from src.data.transforms import DetectionTransform

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "data/processed/instances_deduplicated.json").exists(),
    reason="cluster dataset artifacts are not available",
)


def dataset():
    return FloorplanCocoDataset(
        ROOT.parent / "area_segmentation_coco/images",
        ROOT / "data/processed/instances_deduplicated.json",
        read_split(ROOT / "data/splits/train.txt"),
        DetectionTransform(),
    )


def test_dataset_item_fields_and_alignment():
    image, target = dataset()[0]
    assert set(target) == {"boxes", "labels", "masks", "image_id", "area", "iscrowd"}
    assert tuple(target["masks"].shape[1:]) == tuple(image.shape[1:])
    assert len(target["boxes"]) == len(target["labels"]) == len(target["masks"])
    assert target["boxes"].shape[1] == 4


def test_category_labels_remain_contiguous():
    ds = dataset()
    labels = {ann["category_id"] for anns in ds.annotations.values() for ann in anns}
    assert labels == set(range(1, 20))
