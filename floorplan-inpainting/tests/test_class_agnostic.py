import copy
import json
from pathlib import Path

import pytest

from src.data.class_agnostic import make_class_agnostic_coco, validate_class_agnostic
from src.data.coco_dataset import FloorplanCocoDataset, read_split
from src.models.mask_rcnn import build_mask_rcnn

ROOT = Path(__file__).resolve().parents[1]
requires_dataset = pytest.mark.skipif(
    not (ROOT / "data/processed/instances_deduplicated.json").exists(),
    reason="cluster dataset artifacts are not available",
)


@requires_dataset
def test_class_agnostic_conversion_changes_only_category_metadata_and_labels():
    source = json.loads((ROOT / "data/processed/instances_deduplicated.json").read_text())
    original = copy.deepcopy(source)
    converted = make_class_agnostic_coco(source)
    assert validate_class_agnostic(source, converted)
    assert source == original
    assert len(converted["images"]) == 56
    assert len(converted["annotations"]) == 1633
    assert {annotation["category_id"] for annotation in converted["annotations"]} == {1}
    assert [annotation["id"] for annotation in converted["annotations"]] == [
        annotation["id"] for annotation in source["annotations"]
    ]
    assert [annotation["segmentation"] for annotation in converted["annotations"]] == [
        annotation["segmentation"] for annotation in source["annotations"]
    ]


@requires_dataset
def test_class_agnostic_dataset_uses_unchanged_split_ids():
    dataset = FloorplanCocoDataset(
        ROOT.parent / "area_segmentation_coco/images",
        ROOT / "data/processed/instances_deduplicated_agnostic.json",
        read_split(ROOT / "data/splits/train.txt"),
    )
    assert len(dataset) == 35
    assert {annotation["category_id"] for annotations in dataset.annotations.values() for annotation in annotations} == {1}
    assert sum(map(len, dataset.annotations.values())) == 887


@requires_dataset
def test_original_multiclass_dataset_loader_remains_supported():
    dataset = FloorplanCocoDataset(
        ROOT.parent / "area_segmentation_coco/images",
        ROOT / "data/processed/instances_deduplicated.json",
    )
    assert len(dataset.coco_data["categories"]) == 19


def test_class_agnostic_model_has_background_and_one_foreground_class():
    model = build_mask_rcnn(2, min_size=64, max_size=128, pretrained=False)
    assert model.roi_heads.box_predictor.cls_score.out_features == 2
    assert model.roi_heads.mask_predictor.mask_fcn_logits.out_channels == 2
