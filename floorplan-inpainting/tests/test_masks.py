import pytest

pytest.importorskip("numpy")
pytest.importorskip("cv2")
pytest.importorskip("torch")
import numpy as np
import torch
from pycocotools import mask as mask_utils

from src.data.coco_dataset import FloorplanCocoDataset
from src.evaluation.coco_eval import compact_predictions, output_to_coco


def test_polygon_rasterizes_correctly():
    mask = FloorplanCocoDataset.rasterize([[1, 1, 8, 1, 8, 8, 1, 8]], 10, 10)
    assert mask.shape == (10, 10)
    assert mask.dtype == np.uint8
    assert mask[5, 5] == 1 and mask[0, 0] == 0


def test_boxes_match_masks():
    mask = FloorplanCocoDataset.rasterize([[2, 3, 7, 3, 7, 9, 2, 9]], 12, 12)
    ys, xs = np.where(mask)
    assert [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1] == [2, 3, 8, 10]


def test_predictions_are_compacted_to_rle():
    output = {
        "boxes": torch.tensor([[2.0, 3.0, 8.0, 10.0]]),
        "labels": torch.tensor([12]),
        "scores": torch.tensor([0.9]),
        "masks": torch.from_numpy(FloorplanCocoDataset.rasterize([[2, 3, 7, 3, 7, 9, 2, 9]], 12, 12)).float()[
            None, None
        ],
    }
    boxes, masks = output_to_coco(7, output)
    assert "counts" in masks[0]["segmentation"]
    rle = dict(masks[0]["segmentation"])
    rle["counts"] = rle["counts"].encode("ascii")
    assert mask_utils.decode(rle).shape == (12, 12)
    compact = dict(compact_predictions(boxes, masks, image_ids=[7, 8]))
    assert compact[7]["segmentations"] and compact[8]["segmentations"] == []
