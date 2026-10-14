from types import SimpleNamespace

import numpy as np

from src.evaluation.per_class_metrics import per_class_mask_metrics, rare_aggregate_metrics


def test_per_class_metrics_are_extracted_from_accumulated_precision():
    # [IoU=2, recall=3, category=2, area=1, maxDets=1]
    precision = np.full((2, 3, 2, 1, 1), -1.0)
    precision[:, :, 0, 0, 0] = [[0.2, 0.4, 0.6], [0.4, 0.6, 0.8]]
    precision[:, :, 1, 0, 0] = 0.9
    evaluator = SimpleNamespace(
        params=SimpleNamespace(catIds=[1, 2], iouThrs=np.array([0.5, 0.75])),
        eval={"precision": precision},
    )
    coco = SimpleNamespace(
        dataset={
            "categories": [{"id": 1, "name": "one"}, {"id": 2, "name": "two"}, {"id": 3, "name": "empty"}],
            "annotations": [{"image_id": 7, "category_id": 1}, {"image_id": 7, "category_id": 2}],
        }
    )
    rows = per_class_mask_metrics(coco, evaluator, [7])
    assert np.isclose(rows[0]["mask_AP"], 0.5)
    assert np.isclose(rows[0]["mask_AP50"], 0.4)
    assert np.isclose(rows[1]["mask_AP"], 0.9)
    assert rows[2]["test_instances"] == 0 and rows[2]["mask_AP"] is None


def test_rare_aggregate_uses_primary_support_and_reports_weighted_secondary():
    rows = [
        {"category_id": 11, "test_instances": 1, "mask_AP": 0.2, "mask_AP50": 0.4},
        {"category_id": 12, "test_instances": 3, "mask_AP": 0.6, "mask_AP50": 0.8},
        {"category_id": 17, "test_instances": 2, "mask_AP": 1.0, "mask_AP50": 1.0},
        {"category_id": 16, "test_instances": 0, "mask_AP": None, "mask_AP50": None},
    ]
    config = {
        "categories": {
            11: {"role": "primary_rare"},
            12: {"role": "primary_rare"},
            16: {"role": "primary_rare"},
            17: {"role": "low_support"},
        }
    }
    metrics = rare_aggregate_metrics(rows, config)
    assert np.isclose(metrics["rare_mask_AP"], 0.4)
    assert np.isclose(metrics["weighted_rare_mask_AP"], 0.5)
    assert metrics["rare_evaluable_classes"] == 2 and metrics["rare_test_instances"] == 4
