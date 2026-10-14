import contextlib
import io

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def output_to_coco(image_id, output, score_threshold=0.0):
    """Encode one model output immediately so dense float masks can be freed."""
    bbox_results, mask_results = [], []
    for box, label, score, mask in zip(output["boxes"], output["labels"], output["scores"], output["masks"]):
        score = float(score)
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = map(float, box)
        common = {"image_id": image_id, "category_id": int(label), "score": score}
        bbox_results.append({**common, "bbox": [x1, y1, x2 - x1, y2 - y1]})
        # One mask at a time: all float masks of several large plans can take tens of GB.
        binary = np.asfortranarray((mask[0].detach().cpu().numpy() >= 0.5).astype(np.uint8))
        rle = mask_utils.encode(binary)
        rle["counts"] = rle["counts"].decode("ascii")
        mask_results.append({**common, "segmentation": rle})
    return bbox_results, mask_results


def predictions_to_coco(predictions, score_threshold=0.0):
    bbox_results, mask_results = [], []
    for image_id, output in predictions:
        boxes, masks = output_to_coco(image_id, output, score_threshold)
        bbox_results.extend(boxes)
        mask_results.extend(masks)
    return bbox_results, mask_results


def _evaluate(coco_gt, results, image_ids, iou_type, category_ids=None):
    if not results:
        return None
    coco_dt = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, iou_type)
    evaluator.params.imgIds = list(image_ids)
    if category_ids is not None:
        evaluator.params.catIds = list(category_ids)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return evaluator


def evaluate_coco_results(annotation_file, bbox, masks, image_ids, return_evaluators=False):
    coco_gt = COCO(str(annotation_file))
    bbox_eval = _evaluate(coco_gt, bbox, image_ids, "bbox")
    mask_eval = _evaluate(coco_gt, masks, image_ids, "segm")

    def metrics(evaluator):
        if evaluator is None:
            return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
        return {
            "AP": float(evaluator.stats[0]),
            "AP50": float(evaluator.stats[1]),
            "AP75": float(evaluator.stats[2]),
        }

    result = ({"bbox": metrics(bbox_eval), "mask": metrics(mask_eval)}, (bbox, masks), coco_gt)
    if return_evaluators:
        return (*result, {"bbox": bbox_eval, "mask": mask_eval})
    return result


def evaluate_coco(annotation_file, predictions, image_ids, score_threshold=0.0):
    """Compatibility wrapper for callers that already hold dense predictions."""
    bbox, masks = predictions_to_coco(predictions, score_threshold)
    return evaluate_coco_results(annotation_file, bbox, masks, image_ids)


def compact_predictions(bbox_results, mask_results, score_threshold=0.5, image_ids=None):
    """Create visualization records containing RLEs rather than dense masks."""
    grouped = {
        image_id: {"boxes": [], "labels": [], "scores": [], "segmentations": []}
        for image_id in (image_ids or [])
    }
    for bbox, mask in zip(bbox_results, mask_results):
        if bbox["score"] < score_threshold:
            continue
        item = grouped.setdefault(
            bbox["image_id"], {"boxes": [], "labels": [], "scores": [], "segmentations": []}
        )
        x, y, width, height = bbox["bbox"]
        item["boxes"].append([x, y, x + width, y + height])
        item["labels"].append(bbox["category_id"])
        item["scores"].append(bbox["score"])
        item["segmentations"].append(mask["segmentation"])
    return sorted(grouped.items())
